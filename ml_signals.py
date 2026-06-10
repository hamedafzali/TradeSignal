import copy
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

RETRAIN_EVERY = int(os.getenv("RETRAIN_EVERY_SECONDS", "7200"))  # 2-hour fallback
RETRAIN_OUTCOME_THRESHOLD = int(os.getenv("RETRAIN_OUTCOME_THRESHOLD", "25"))
# Minimum resolved live outcomes (per direction) before blending them into
# training. Below this, a 5x-weighted handful of trades dominates the GBM loss
# and the model whipsaws on individual outcomes instead of learning patterns.
MIN_LIVE_BLEND = int(os.getenv("MIN_LIVE_BLEND_OUTCOMES", "30"))
# Champion/challenger: a retrained model is rejected if its holdout buy-AUC is
# worse than the current model's by more than this margin (~2x the AUC standard
# error at the typical ~1,200-sample calibration set).
PROMOTION_AUC_MARGIN = float(os.getenv("PROMOTION_AUC_MARGIN", "0.05"))
AI_SIGNAL_THRESHOLD = float(os.getenv("AI_SIGNAL_THRESHOLD", "0.65"))

# ── Sector ETF map ───────────────────────────────────────────────────────────
# Maps symbol → sector SPDR ETF for relative strength features.
# Crypto and unlisted symbols map to None (feature gracefully absent).
_SECTOR_MAP: dict[str, str | None] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "TSLA": "XLK",
    "GOOG": "XLK", "GOOGL": "XLK", "META": "XLK", "AMZN": "XLK",
    "JPM": "XLF", "GS": "XLF", "BAC": "XLF", "WFC": "XLF", "MS": "XLF",
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE",
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "BTC-USD": None, "ETH-USD": None, "BNB-USD": None, "SOL-USD": None,
}

# ── Market context cache ──────────────────────────────────────────────────────
_mkt_cache: dict = {"ts": 0.0, "spy": None, "vix": None}
_mkt_last_good: dict = {"spy": None, "vix": None, "ts": 0.0}
_MKT_TTL = 300        # seconds — live cache TTL
_MKT_FALLBACK_MAX = 4 * 3600  # 4h — don't use stale data older than this
_sector_cache: dict[str, dict] = {}  # etf_ticker → {ts, df}


def _get_market_ctx(period: str = "730d") -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """SPY (1h) + VIX (1d) with 5-min in-memory cache.
    On download failure falls back to the last successful fetch so regime
    classification never returns 'unknown' due to a transient network error."""
    now = time.time()
    if now - _mkt_cache["ts"] < _MKT_TTL and _mkt_cache["spy"] is not None:
        return _mkt_cache["spy"], _mkt_cache["vix"]
    spy, vix = None, None
    try:
        spy = yf.download("SPY", period=period, interval="1h", progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.droplevel(1)
        if spy.empty:
            spy = None
        else:
            _mkt_last_good["spy"] = spy
            _mkt_last_good["ts"]  = now
    except Exception as e:
        logger.debug(f"[ML] SPY fetch failed: {e}")
    try:
        vix = yf.download("^VIX", period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
        if vix.empty:
            vix = None
        else:
            _mkt_last_good["vix"] = vix
            _mkt_last_good["ts"]  = now
    except Exception as e:
        logger.debug(f"[ML] VIX fetch failed: {e}")
    # Fall back to last known good data only if it is fresh enough
    fallback_ok = (now - _mkt_last_good["ts"]) < _MKT_FALLBACK_MAX
    if spy is None and _mkt_last_good["spy"] is not None and fallback_ok:
        spy = _mkt_last_good["spy"]
        logger.debug("[ML] SPY: using last known good data (fetch failed)")
    if vix is None and _mkt_last_good["vix"] is not None and fallback_ok:
        vix = _mkt_last_good["vix"]
        logger.debug("[ML] VIX: using last known good data (fetch failed)")
    _mkt_cache.update({"ts": now, "spy": spy, "vix": vix})
    return spy, vix


def get_predict_market_context() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Public export — bot.py uses this so feature capture shares the same cache."""
    return _get_market_ctx()


def _get_sector_ctx(symbol: str, period: str = "730d") -> pd.DataFrame | None:
    """Fetch the sector SPDR ETF for a symbol. Returns None for crypto/unknown."""
    etf = _SECTOR_MAP.get(symbol.upper())
    if etf is None:
        return None
    now = time.time()
    cached = _sector_cache.get(etf)
    if cached and now - cached["ts"] < _MKT_TTL:
        return cached.get("df")
    try:
        df = yf.download(etf, period=period, interval="1h", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        sector_df = df if not df.empty else None
    except Exception as e:
        logger.debug(f"[ML] Sector ETF {etf} fetch failed: {e}")
        sector_df = None
    _sector_cache[etf] = {"ts": now, "df": sector_df}
    return sector_df


def _align(series: pd.Series, target: pd.Index) -> pd.Series:
    """Reindex series to target using ffill, normalising timezones."""
    try:
        s = series.copy()
        if getattr(s.index, "tz", None) != getattr(target, "tz", None):
            if getattr(s.index, "tz", None) is None:
                s.index = s.index.tz_localize("UTC")
            if getattr(target, "tz", None) is not None:
                s.index = s.index.tz_convert(target.tz)
            else:
                s.index = s.index.tz_localize(None)
        return s.reindex(target, method="ffill")
    except Exception:
        return pd.Series(np.nan, index=target, dtype=float)


def build_features(df: pd.DataFrame,
                   spy_df: pd.DataFrame | None = None,
                   vix_df: pd.DataFrame | None = None,
                   sector_df: pd.DataFrame | None = None) -> pd.DataFrame:
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    open_  = df["Open"].squeeze() if "Open" in df.columns else close

    feat = pd.DataFrame(index=df.index)

    # ── Direction features ────────────────────────────────────────────────────
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(com=13, min_periods=14).mean()
    al = loss.ewm(com=13, min_periods=14).mean()
    feat["rsi"] = 100 - (100 / (1 + ag / al))

    # MACD histogram (normalized by price to be scale-invariant)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - sig) / (close + 1e-9)

    # Bollinger Band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_pos"] = (close - sma20) / (2 * std20 + 1e-9)

    # Volume ratio (current vs 20-bar average)
    vol_ma = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / (vol_ma + 1e-9)

    # ── ATR base (used by multiple features below) ────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feat["atr_pct"] = atr / (close + 1e-9)

    # ── Volatility regime features (key for TP/SL reachability) ──────────────
    # ATR percentile vs 90-bar rolling distribution — is vol expanding or contracting?
    feat["atr_percentile"] = atr.rolling(90, min_periods=20).rank(pct=True)

    # Garman-Klass volatility (uses OHLC — more efficient than close-only)
    # Measures realized volatility including intrabar range
    log_hl = (np.log(high / (low + 1e-9))) ** 2
    log_co = (np.log(close / (open_ + 1e-9))) ** 2
    feat["gk_vol"] = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(14).mean()

    # Volatility-of-volatility: is vol regime stable or chaotic?
    feat["vol_of_vol"] = feat["atr_pct"].rolling(20).std()

    # ── Momentum quality features ─────────────────────────────────────────────
    # Momentum over two horizons (removed roc_3, roc_5 — subsumed)
    feat["roc_10"] = close.pct_change(10)
    feat["roc_20"] = close.pct_change(20)

    # Momentum consistency: what fraction of bars in last 10 moved in positive direction?
    # High consistency = persistent trend; low = choppy
    feat["mom_consistency"] = close.pct_change().rolling(10).apply(
        lambda x: float((x > 0).mean()), raw=False
    )

    # Momentum acceleration: is the move getting faster or slower?
    feat["mom_accel"] = close.pct_change(5) - close.pct_change(20)

    # ── Price structure ───────────────────────────────────────────────────────
    # VWAP spread: proxy for institutional vs retail positioning
    typical = (high + low + close) / 3
    vwap_approx = (typical * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    feat["vwap_spread"] = (close - vwap_approx) / (atr + 1e-9)

    # ── Time features (cyclical encoding — intraday patterns are real) ────────
    try:
        idx = feat.index
        hour = idx.hour if hasattr(idx, "hour") else pd.Series(0, index=idx)
        dow  = idx.dayofweek if hasattr(idx, "dayofweek") else pd.Series(0, index=idx)
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feat["dow_sin"]  = np.sin(2 * np.pi * dow / 5)
        feat["dow_cos"]  = np.cos(2 * np.pi * dow / 5)
    except Exception:
        pass

    # ── Market context (optional — gracefully absent if fetch fails) ──────────
    if spy_df is not None and not spy_df.empty:
        try:
            spy_c = _align(spy_df["Close"].squeeze(), feat.index)
            feat["rel_str_1h"] = close.pct_change(1) - spy_c.pct_change(1)
            feat["rel_str_8h"] = close.pct_change(8) - spy_c.pct_change(8)
            spy_ema50 = spy_c.ewm(span=50, adjust=False).mean()
            feat["spy_regime"] = (spy_c >= spy_ema50).astype(float) * 2 - 1
            spy_sma20 = spy_c.rolling(20).mean()
            spy_std20 = spy_c.rolling(20).std()
            feat["spy_bb_pos"] = (spy_c - spy_sma20) / (2 * spy_std20 + 1e-9)
        except Exception as e:
            logger.debug(f"[ML] SPY feature error: {e}")

    if vix_df is not None and not vix_df.empty:
        try:
            vix_c = _align(vix_df["Close"].squeeze(), feat.index)
            vix_ma = vix_c.rolling(20, min_periods=1).mean()
            feat["vix_norm"] = vix_c / (vix_ma + 1e-9)
        except Exception as e:
            logger.debug(f"[ML] VIX feature error: {e}")

    # ── Sector ETF relative strength (optional — absent for crypto) ───────────
    # Compares symbol move vs its sector ETF to isolate stock-specific strength.
    # Absent (column not added) when sector ETF is unknown/unavailable.
    if sector_df is not None and not sector_df.empty:
        try:
            sec_c = _align(sector_df["Close"].squeeze(), feat.index)
            feat["sector_rel_str_1h"] = close.pct_change(1) - sec_c.pct_change(1)
            feat["sector_rel_str_8h"] = close.pct_change(8) - sec_c.pct_change(8)
        except Exception as e:
            logger.debug(f"[ML] Sector feature error: {e}")

    return feat


def build_labels(df: pd.DataFrame, forward: int = 8, threshold: float = 0.008) -> pd.Series:
    """Simple forward-return labels — kept for bootstrap compatibility."""
    close = df["Close"].squeeze()
    ret = close.shift(-forward) / close - 1
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[ret > threshold] = 1
    labels[ret < -threshold] = -1
    return labels


def build_labels_tpsl(df: pd.DataFrame,
                      sl_mult: float = 1.5, tp_mult: float = 2.5,
                      outcome_bars: int = 24) -> tuple["pd.Series", "pd.Series"]:
    """
    TP/SL-aligned labels matching the live outcome resolution logic exactly.
    Returns (y_buy, y_sell) — 0/1 series where 1 = "this direction wins".
    Rows whose outcome window is incomplete (warmup head, final outcome_bars
    tail, bad ATR) are NaN so callers' dropna() removes them — previously they
    trained as fake "don't fire" negatives, biasing the model against firing
    on exactly the newest data.
    """
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    n = len(df)
    # numpy arrays: the nested loop below does ~150K scalar reads on 730d data —
    # pandas .iloc there costs seconds per symbol, numpy costs milliseconds
    close_a = close.to_numpy(dtype=float)
    high_a  = high.to_numpy(dtype=float)
    low_a   = low.to_numpy(dtype=float)
    atr_a   = atr.to_numpy(dtype=float)
    yb = np.full(n, np.nan)
    ys = np.full(n, np.nan)

    for i in range(14, n - outcome_bars):
        atr_val = atr_a[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue  # stays NaN — dropped by callers, not trained as a negative
        entry = close_a[i]
        tp_buy  = entry + atr_val * tp_mult
        sl_buy  = entry - atr_val * sl_mult
        tp_sell = entry - atr_val * tp_mult
        sl_sell = entry + atr_val * sl_mult

        buy_result = sell_result = 0
        for j in range(i + 1, i + 1 + outcome_bars):
            h = high_a[j]
            l = low_a[j]
            # BUY outcome
            if buy_result == 0:
                if h >= tp_buy and l <= sl_buy:
                    buy_result = 0   # ambiguous
                elif h >= tp_buy:
                    buy_result = 1   # TP hit
                elif l <= sl_buy:
                    buy_result = -1  # SL hit — resolved, stop looking
            # SELL outcome
            if sell_result == 0:
                if l <= tp_sell and h >= sl_sell:
                    sell_result = 0
                elif l <= tp_sell:
                    sell_result = 1   # TP hit for sell
                elif h >= sl_sell:
                    sell_result = -1  # SL hit for sell
            if buy_result != 0 and sell_result != 0:
                break

        yb[i] = 1.0 if buy_result  == 1 else 0.0
        ys[i] = 1.0 if sell_result == 1 else 0.0

    return (pd.Series(yb, index=df.index, dtype=float),
            pd.Series(ys, index=df.index, dtype=float))


class StockModel:
    def __init__(self, symbol: str):
        self.symbol = symbol

        # ── 1h model: used for bias/confirmation on hourly candles ──────────
        self.clf_buy  = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.clf_sell = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.cal_buy:  CalibratedClassifierCV | None = None
        self.cal_sell: CalibratedClassifierCV | None = None
        self.scaler       = StandardScaler()
        self.feature_cols: list[str] = []
        self.trained      = False

        # ── 5m model: trained on 5m candles to confirm 5m rule signals ──────
        # The 1h model learned hourly patterns and cannot reliably confirm signals
        # that fired on 5m bars — RSI/MACD/EMA values are statistically different
        # across timeframes. The 5m model uses the same feature set but learns
        # the distribution of those features at 5m resolution.
        self.clf_buy_5m  = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.clf_sell_5m = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.cal_buy_5m:  CalibratedClassifierCV | None = None
        self.cal_sell_5m: CalibratedClassifierCV | None = None
        self.scaler_5m       = StandardScaler()
        self.feature_cols_5m: list[str] = []
        self.trained_5m       = False

        # ── Meta-labeling classifiers (Phase 6) ──────────────────────────────
        # Secondary classifiers trained to predict "is the primary classifier
        # correct?" — trained on calibration data with primary_prob appended as
        # an extra feature. Separates direction from confidence quality.
        self.clf_meta_buy:    GradientBoostingClassifier | None = None
        self.clf_meta_sell:   GradientBoostingClassifier | None = None
        self.clf_meta_buy_5m:  GradientBoostingClassifier | None = None
        self.clf_meta_sell_5m: GradientBoostingClassifier | None = None
        self.trained_meta    = False
        self.trained_meta_5m = False

        self.trained_at   = 0.0
        self.train_samples    = 0
        self.outcome_samples  = 0
        self.outcome_count_at_train = 0
        self.last_blend_count = 0  # live outcomes actually blended in last train()
        self.val_auc_buy:  float | None = None  # holdout AUC from last train()
        self.val_auc_sell: float | None = None
        self.last_promotion = ""  # "promoted" | "rejected" from last train_gated()
        self._path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "rb") as f:
                d = pickle.load(f)
            self.__dict__.update(d)
            logger.info(f"[ML] Loaded cached model for {self.symbol}")
        except Exception:
            pass

    def _save(self):
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        with open(self._path, "wb") as f:
            pickle.dump(data, f)

    def needs_retrain(self, current_outcome_count: int = 0) -> bool:
        if not self.trained:
            return True
        # Below the blend threshold, train() skips live outcomes entirely, so a
        # retrain reproduces the same synthetic-only model — pure waste
        # (~8s of yfinance downloads per symbol per cycle).
        if current_outcome_count < MIN_LIVE_BLEND:
            return False
        new_outcomes = current_outcome_count - self.outcome_count_at_train
        if new_outcomes >= RETRAIN_OUTCOME_THRESHOLD:
            return True
        return (time.time() - self.trained_at) > RETRAIN_EVERY

    def train_gated(self, outcome_data: list[dict] | None = None) -> bool:
        """
        Champion/challenger wrapper around train(): keeps the retrained model
        only if its holdout buy-AUC is not materially worse than the current
        model's. Retrains previously deployed unconditionally, so one bad
        retrain (bad data window, yfinance glitch) silently replaced a good
        model with no detection. Use this for all scheduled/automatic retrains;
        bootstrap and first-time training go through train() directly.
        """
        champion = None
        champion_auc = self.val_auc_buy if self.trained else None
        if self.trained and champion_auc is not None:
            # train() refits estimators in place — a shallow snapshot would be
            # corrupted by the retrain, so deep-copy the persisted state.
            champion = copy.deepcopy(
                {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
            )
        ok = self.train(outcome_data)
        if not ok:
            return False
        self.last_promotion = "promoted"
        if (champion is not None
                and self.val_auc_buy is not None
                and self.val_auc_buy < champion_auc - PROMOTION_AUC_MARGIN):
            logger.warning(
                f"[ML] {self.symbol}: challenger REJECTED — holdout AUC "
                f"{self.val_auc_buy:.3f} vs champion {champion_auc:.3f} "
                f"(margin {PROMOTION_AUC_MARGIN})"
            )
            self.__dict__.update(champion)
            self.last_promotion = "rejected"
            self._save()  # restore champion pickle (train() saved the challenger)
        return True

    def train(self, outcome_data: list[dict] | None = None, _log: bool = True) -> bool:
        """
        Train on up to 730 days of hourly data, then blend in real outcome
        samples (gated by MIN_LIVE_BLEND per direction).
        outcome_data: list of {"features": {col: val, ...}, "label": int}
        """
        logger.info(f"[ML] Training {self.symbol}...")
        try:
            # 730d is yfinance's max for 1h bars: ~3,200 samples for stocks vs
            # ~390 from 60d — and covers multiple market regimes instead of
            # only the most recent one.
            df = yf.download(self.symbol, period="730d", interval="1h", progress=False, auto_adjust=True)
            if len(df) < 200:
                logger.warning(f"[ML] Too little data for {self.symbol}")
                return False

            spy_df, vix_df = _get_market_ctx()
            sector_df = _get_sector_ctx(self.symbol)
            features = build_features(df, spy_df=spy_df, vix_df=vix_df, sector_df=sector_df)

            # TP/SL-aligned labels: train buy/sell classifiers independently
            # with objectives that exactly match live outcome resolution
            y_buy_raw, y_sell_raw = build_labels_tpsl(df)
            combined = pd.concat(
                [features, y_buy_raw.rename("y_buy"), y_sell_raw.rename("y_sell")],
                axis=1
            ).dropna()

            if len(combined) < 100:
                return False

            self.feature_cols = [c for c in combined.columns
                                  if c not in ("y_buy", "y_sell")]
            X = combined[self.feature_cols].values
            y_buy  = combined["y_buy"].values
            y_sell = combined["y_sell"].values

            # Blend real outcomes using sample_weight (5x) instead of row duplication.
            # Row duplication causes GBM to memorise exact feature vectors at those
            # outcome bars — sample_weight applies the same boost via the loss function
            # without creating near-duplicate rows that overfit.
            self.last_blend_count = 0
            # Per-direction gate: blend BUY outcomes only once there are enough of
            # them to constitute a sample, same for SELL. With ~98% BUY signals,
            # clf_sell would otherwise be "trained" on 1-2 live trades.
            if outcome_data:
                n_buy_outcomes  = sum(1 for it in outcome_data if abs(it["label"]) == 1)
                n_sell_outcomes = sum(1 for it in outcome_data if abs(it["label"]) == 2)
                blend_buy  = n_buy_outcomes  >= MIN_LIVE_BLEND
                blend_sell = n_sell_outcomes >= MIN_LIVE_BLEND
                if not (blend_buy or blend_sell):
                    logger.info(
                        f"[ML] {self.symbol}: {len(outcome_data)} live outcomes below "
                        f"blend threshold ({MIN_LIVE_BLEND}/direction) — training synthetic-only"
                    )
                    outcome_data = None
                else:
                    outcome_data = [it for it in outcome_data
                                    if (blend_buy and abs(it["label"]) == 1)
                                    or (blend_sell and abs(it["label"]) == 2)]

            # ── Temporal split with embargo, BEFORE blending live outcomes ──────
            # Labels look ahead `outcome_bars` (24) bars, so the last 24 rows of
            # the train portion share outcome windows with the first calibration
            # rows — the embargo gap removes that leakage. Live outcomes are
            # appended to the TRAIN portion after the split; previously they were
            # appended to X before splitting and always landed in the calibration
            # tail, so they calibrated probabilities but never trained the
            # classifiers at all.
            _EMBARGO = 24  # = build_labels_tpsl outcome_bars
            n_cal = max(20, len(X) // 5)
            if n_cal + _EMBARGO < len(X) - 50:
                X_main_raw, X_cal_raw = X[:-(n_cal + _EMBARGO)], X[-n_cal:]
                yb_main, yb_cal       = y_buy[:-(n_cal + _EMBARGO)],  y_buy[-n_cal:]
                ys_main, ys_cal       = y_sell[:-(n_cal + _EMBARGO)], y_sell[-n_cal:]
            else:
                X_main_raw, X_cal_raw = X, X
                yb_main, yb_cal       = y_buy,  y_buy
                ys_main, ys_cal       = y_sell, y_sell

            sw_buy_main  = np.ones(len(X_main_raw))
            sw_sell_main = np.ones(len(X_main_raw))

            if outcome_data:
                extra_rows, extra_buy, extra_sell = [], [], []
                sw_buy_extra, sw_sell_extra = [], []
                for item in outcome_data:
                    row = [item["features"].get(col, 0.0) for col in self.feature_cols]
                    lbl = item["label"]
                    extra_rows.append(row)
                    # label scheme (from get_outcome_training_data):
                    #  +1 = buy was correct   → teach clf_buy to fire (y_buy=1), clf_sell neutral (0, low weight)
                    #  -1 = buy was wrong     → teach clf_buy NOT to fire (y_buy=0), clf_sell neutral (0, low weight)
                    #  +2 = sell was correct  → teach clf_sell to fire (y_sell=1), clf_buy neutral (0, low weight)
                    #  -2 = sell was wrong    → teach clf_sell NOT to fire (y_sell=0), clf_buy neutral (0, low weight)
                    if lbl == 1:
                        extra_buy.append(1);  extra_sell.append(0)
                        sw_buy_extra.append(5.0); sw_sell_extra.append(0.5)
                    elif lbl == -1:
                        extra_buy.append(0);  extra_sell.append(0)
                        sw_buy_extra.append(5.0); sw_sell_extra.append(0.5)
                    elif lbl == 2:
                        extra_buy.append(0);  extra_sell.append(1)
                        sw_buy_extra.append(0.5); sw_sell_extra.append(5.0)
                    else:  # -2
                        extra_buy.append(0);  extra_sell.append(0)
                        sw_buy_extra.append(0.5); sw_sell_extra.append(5.0)
                if extra_rows:
                    n_real = len(extra_rows)
                    X_main_raw   = np.vstack([X_main_raw, extra_rows])
                    yb_main      = np.concatenate([yb_main, extra_buy])
                    ys_main      = np.concatenate([ys_main, extra_sell])
                    sw_buy_main  = np.concatenate([sw_buy_main,  sw_buy_extra])
                    sw_sell_main = np.concatenate([sw_sell_main, sw_sell_extra])
                    self.last_blend_count = n_real
                    logger.info(f"[ML] {self.symbol}: blended {n_real} real outcomes into training set (5x per-classifier sample_weight)")

            X_main = self.scaler.fit_transform(X_main_raw)   # fit ONLY on train
            X_cal  = self.scaler.transform(X_cal_raw)         # transform without refit

            self.clf_buy.fit(X_main, yb_main, sample_weight=sw_buy_main)
            self.clf_sell.fit(X_main, ys_main, sample_weight=sw_sell_main)

            # Log top-5 features by importance so we can detect noise dominance
            imp = self.clf_buy.feature_importances_
            top5 = sorted(zip(self.feature_cols, imp), key=lambda x: -x[1])[:5]
            logger.info(f"[ML] {self.symbol} 1h top features: " +
                        " | ".join(f"{n}={v:.3f}" for n, v in top5))

            # ── Feature importance pruning: keep features >= 0.3× mean importance ──
            # Drops noise features identified by the primary classifier.
            # Re-trains on pruned set so the scaler and meta-classifiers are consistent.
            try:
                mean_imp = float(imp.mean())
                active_mask = imp >= 0.3 * mean_imp
                active_cols = [col for col, keep in zip(self.feature_cols, active_mask) if keep]
                dropped = [col for col, keep in zip(self.feature_cols, active_mask) if not keep]
                if len(active_cols) >= 10 and len(dropped) > 0:
                    logger.info(f"[ML] {self.symbol} pruning {len(dropped)} low-importance features: {dropped}")
                    # Re-subset X matrices and re-fit scaler + classifiers on pruned columns
                    col_idx = [self.feature_cols.index(c) for c in active_cols]
                    X_main_p = X_main_raw[:, col_idx]
                    X_cal_p  = X_cal_raw[:, col_idx]
                    pruned_scaler = StandardScaler()
                    Xm_p = pruned_scaler.fit_transform(X_main_p)
                    Xc_p = pruned_scaler.transform(X_cal_p)
                    pruned_buy  = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
                    pruned_sell = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
                    sw_b_p = sw_buy_main[:len(Xm_p)]
                    sw_s_p = sw_sell_main[:len(Xm_p)]
                    pruned_buy.fit(Xm_p, yb_main, sample_weight=sw_b_p)
                    pruned_sell.fit(Xm_p, ys_main, sample_weight=sw_s_p)
                    # Replace model components with pruned versions
                    self.feature_cols = active_cols
                    self.scaler = pruned_scaler
                    self.clf_buy  = pruned_buy
                    self.clf_sell = pruned_sell
                    # Re-calibrate on pruned calibration set
                    try:
                        self.cal_buy = CalibratedClassifierCV(pruned_buy, method="sigmoid", cv="prefit")
                        self.cal_buy.fit(Xc_p, yb_cal)
                        self.cal_sell = CalibratedClassifierCV(pruned_sell, method="sigmoid", cv="prefit")
                        self.cal_sell.fit(Xc_p, ys_cal)
                    except Exception:
                        self.cal_buy = None; self.cal_sell = None
                    # Update X_main_raw / X_cal_raw references for PSI quantile storage below
                    X_main_raw = X_main_p
                    X_cal = Xc_p
            except Exception as e:
                logger.debug(f"[ML] Feature pruning skipped for {self.symbol}: {e}")

            # Store training quantile boundaries per feature for PSI drift detection.
            # PSI compares current feature distributions against these training baselines.
            try:
                self._train_quantiles = {
                    col: np.nanpercentile(X_main_raw[:, i], np.linspace(0, 100, 11))
                    for i, col in enumerate(self.feature_cols)
                }
            except Exception:
                self._train_quantiles = {}

            try:
                self.cal_buy = CalibratedClassifierCV(self.clf_buy, method="sigmoid", cv="prefit")
                self.cal_buy.fit(X_cal, yb_cal)
                self.cal_sell = CalibratedClassifierCV(self.clf_sell, method="sigmoid", cv="prefit")
                self.cal_sell.fit(X_cal, ys_cal)
            except Exception as e:
                logger.debug(f"[ML] Calibration skipped for {self.symbol}: {e}")
                self.cal_buy = None
                self.cal_sell = None

            # ── Holdout validation: AUC on the embargoed calibration tail ────────
            # Out-of-sample skill measurement per retrain. AUC 0.5 = no skill;
            # this is the evidence base for promoting/rejecting retrained models.
            self.val_auc_buy = None
            self.val_auc_sell = None
            try:
                from sklearn.metrics import roc_auc_score
                clf_b = self.cal_buy  if self.cal_buy  is not None else self.clf_buy
                clf_s = self.cal_sell if self.cal_sell is not None else self.clf_sell
                if len(np.unique(yb_cal)) > 1:
                    self.val_auc_buy = float(roc_auc_score(yb_cal, clf_b.predict_proba(X_cal)[:, 1]))
                if len(np.unique(ys_cal)) > 1:
                    self.val_auc_sell = float(roc_auc_score(ys_cal, clf_s.predict_proba(X_cal)[:, 1]))
                fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
                logger.info(f"[ML] {self.symbol} holdout AUC: "
                            f"buy={fmt(self.val_auc_buy)} sell={fmt(self.val_auc_sell)}")
            except Exception as e:
                logger.debug(f"[ML] Holdout AUC skipped for {self.symbol}: {e}")

            # ── Meta-labeling: train on calibration set (never seen by primary) ──
            # meta_label_buy[i] = 1 when primary predicted BUY and it was correct
            # Features: original X_cal + primary_prob (the "skill" signal)
            try:
                buy_clf_for_meta  = self.cal_buy  if self.cal_buy  is not None else self.clf_buy
                sell_clf_for_meta = self.cal_sell if self.cal_sell is not None else self.clf_sell
                p_buy_cal  = buy_clf_for_meta.predict_proba(X_cal)[:, 1]
                p_sell_cal = sell_clf_for_meta.predict_proba(X_cal)[:, 1]
                meta_buy_label  = ((p_buy_cal  > 0.5) & (yb_cal == 1)).astype(int)
                meta_sell_label = ((p_sell_cal > 0.5) & (ys_cal == 1)).astype(int)
                X_meta_buy  = np.hstack([X_cal, p_buy_cal.reshape(-1, 1)])
                X_meta_sell = np.hstack([X_cal, p_sell_cal.reshape(-1, 1)])
                if meta_buy_label.sum() >= 5 and len(np.unique(meta_buy_label)) > 1:
                    self.clf_meta_buy = GradientBoostingClassifier(
                        n_estimators=80, max_depth=2, learning_rate=0.08, random_state=42)
                    self.clf_meta_buy.fit(X_meta_buy, meta_buy_label)
                if meta_sell_label.sum() >= 5 and len(np.unique(meta_sell_label)) > 1:
                    self.clf_meta_sell = GradientBoostingClassifier(
                        n_estimators=80, max_depth=2, learning_rate=0.08, random_state=42)
                    self.clf_meta_sell.fit(X_meta_sell, meta_sell_label)
                self.trained_meta = True
                logger.info(f"[ML] {self.symbol} 1h meta-classifiers trained "
                            f"(buy pos={meta_buy_label.sum()}, sell pos={meta_sell_label.sum()})")
            except Exception as e:
                logger.debug(f"[ML] Meta-label training skipped for {self.symbol}: {e}")
                self.trained_meta = False

            self.trained = True
            self.trained_at = time.time()
            self.train_samples = len(X)
            self.outcome_samples = len(outcome_data) if outcome_data else 0
            # Snapshot current resolved outcome count so needs_retrain() can
            # detect new outcomes arriving after this train run
            try:
                from database import get_outcome_count
                self.outcome_count_at_train = get_outcome_count(self.symbol)
            except Exception:
                self.outcome_count_at_train = self.outcome_samples
            self._save()

            # Persist to DB for dashboard reporting (skipped when caller handles logging)
            if _log:
                try:
                    from database import log_training
                    trigger = "outcomes" if (outcome_data and len(outcome_data) >= 3) else "time"
                    log_training(self.symbol, len(X), self.outcome_samples, trigger=trigger)
                except Exception:
                    pass

            logger.info(f"[ML] {self.symbol} 1h model trained on {len(X)} samples")
            # Also train the 5m-specific model so it is ready for 5m rule confirmation
            self.train_5m(_log=False)
            return True
        except Exception as e:
            logger.error(f"[ML] Train error for {self.symbol}: {e}")
            return False

    def train_5m(self, _log: bool = True) -> bool:
        """
        Train the 5m-specific classifier pair on 5 days of 5-minute candles.
        ~1,950 bars vs ~420 from 60d/1h — 4-5× more training data at the right
        resolution for confirming 5m rule signals.
        """
        logger.info(f"[ML] Training 5m model for {self.symbol}...")
        try:
            # yfinance supports up to 60 days for 5m interval (~4,680 bars)
            # vs "5d" giving only 390 bars — use full range for robust training
            df = yf.download(self.symbol, period="60d", interval="5m",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) < 200:
                logger.warning(f"[ML] Too little 5m data for {self.symbol} ({len(df)} bars)")
                return False

            spy_df, vix_df = _get_market_ctx()
            sector_df = _get_sector_ctx(self.symbol)
            features = build_features(df, spy_df=spy_df, vix_df=vix_df, sector_df=sector_df)

            # Use the same TP/SL label logic — outcome_bars=24 means 24×5m = 2h horizon
            y_buy_raw, y_sell_raw = build_labels_tpsl(
                df, sl_mult=1.5, tp_mult=2.5, outcome_bars=24
            )
            combined = pd.concat(
                [features, y_buy_raw.rename("y_buy"), y_sell_raw.rename("y_sell")],
                axis=1
            ).dropna()

            if len(combined) < 100:
                logger.warning(f"[ML] Too few 5m label samples for {self.symbol}")
                return False

            self.feature_cols_5m = [c for c in combined.columns
                                    if c not in ("y_buy", "y_sell")]
            X      = combined[self.feature_cols_5m].values
            y_buy  = combined["y_buy"].values
            y_sell = combined["y_sell"].values

            # Scaler: fit only on training partition. Embargo gap (24 bars =
            # label lookahead) prevents outcome-window leakage into calibration.
            _EMBARGO = 24
            n_cal = max(20, len(X) // 5)
            if n_cal + _EMBARGO < len(X) - 50:
                X_main_raw, X_cal_raw = X[:-(n_cal + _EMBARGO)], X[-n_cal:]
                yb_main, yb_cal       = y_buy[:-(n_cal + _EMBARGO)],  y_buy[-n_cal:]
                ys_main, ys_cal       = y_sell[:-(n_cal + _EMBARGO)], y_sell[-n_cal:]
            else:
                X_main_raw, X_cal_raw = X, X
                yb_main, yb_cal       = y_buy, y_buy
                ys_main, ys_cal       = y_sell, y_sell

            X_main = self.scaler_5m.fit_transform(X_main_raw)
            X_cal  = self.scaler_5m.transform(X_cal_raw)

            self.clf_buy_5m.fit(X_main, yb_main)
            self.clf_sell_5m.fit(X_main, ys_main)

            try:
                self.cal_buy_5m = CalibratedClassifierCV(self.clf_buy_5m, method="sigmoid", cv="prefit")
                self.cal_buy_5m.fit(X_cal, yb_cal)
                self.cal_sell_5m = CalibratedClassifierCV(self.clf_sell_5m, method="sigmoid", cv="prefit")
                self.cal_sell_5m.fit(X_cal, ys_cal)
            except Exception as e:
                logger.debug(f"[ML] 5m calibration skipped for {self.symbol}: {e}")
                self.cal_buy_5m  = None
                self.cal_sell_5m = None

            # ── 5m Meta-labeling ─────────────────────────────────────────────
            try:
                buy_clf_m  = self.cal_buy_5m  if self.cal_buy_5m  is not None else self.clf_buy_5m
                sell_clf_m = self.cal_sell_5m if self.cal_sell_5m is not None else self.clf_sell_5m
                p_b5  = buy_clf_m.predict_proba(X_cal)[:, 1]
                p_s5  = sell_clf_m.predict_proba(X_cal)[:, 1]
                mb5 = ((p_b5 > 0.5) & (yb_cal == 1)).astype(int)
                ms5 = ((p_s5 > 0.5) & (ys_cal == 1)).astype(int)
                Xmb5 = np.hstack([X_cal, p_b5.reshape(-1, 1)])
                Xms5 = np.hstack([X_cal, p_s5.reshape(-1, 1)])
                if mb5.sum() >= 5 and len(np.unique(mb5)) > 1:
                    self.clf_meta_buy_5m = GradientBoostingClassifier(
                        n_estimators=80, max_depth=2, learning_rate=0.08, random_state=42)
                    self.clf_meta_buy_5m.fit(Xmb5, mb5)
                if ms5.sum() >= 5 and len(np.unique(ms5)) > 1:
                    self.clf_meta_sell_5m = GradientBoostingClassifier(
                        n_estimators=80, max_depth=2, learning_rate=0.08, random_state=42)
                    self.clf_meta_sell_5m.fit(Xms5, ms5)
                self.trained_meta_5m = True
            except Exception as e:
                logger.debug(f"[ML] 5m meta-label training skipped for {self.symbol}: {e}")
                self.trained_meta_5m = False

            self.trained_5m = True
            self._save()
            logger.info(f"[ML] {self.symbol} 5m model trained on {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"[ML] 5m train error for {self.symbol}: {e}")
            return False

    def predict(self, df: pd.DataFrame, timeframe: str = "1h",
                sentiment_score: float = 0.0,
                sentiment_momentum: float = 0.0) -> dict:
        """
        Returns buy_prob, sell_prob, ai_signal for the latest candle.
        timeframe='5m' → uses the 5m-trained model (correct for confirming 5m rule signals)
        timeframe='1h' → uses the 1h-trained model (for bias/directional context)
        Falls back to the 1h model if 5m model is not yet trained.

        sentiment_score ∈ [-1, 1]: live FinBERT/Gemini score.
          Positive → boosts buy_prob, dampens sell_prob.
          Trained models use 0.0 baseline; inference applies a soft ±20% adjustment.
        sentiment_momentum ∈ [-2, 2]: current score minus score 3h ago.
          Adds directional velocity signal on top of the point-in-time score.
        """
        default = {"buy_prob": None, "sell_prob": None, "ai_signal": None}

        use_5m = (timeframe == "5m") and self.trained_5m and self.feature_cols_5m
        if use_5m:
            trained_ok   = self.trained_5m
            feature_cols = self.feature_cols_5m
            scaler       = self.scaler_5m
            buy_clf      = self.cal_buy_5m  if self.cal_buy_5m  is not None else self.clf_buy_5m
            sell_clf     = self.cal_sell_5m if self.cal_sell_5m is not None else self.clf_sell_5m
        else:
            trained_ok   = self.trained
            feature_cols = self.feature_cols
            scaler       = self.scaler
            buy_clf      = self.cal_buy  if self.cal_buy  is not None else self.clf_buy
            sell_clf     = self.cal_sell if self.cal_sell is not None else self.clf_sell

        if not trained_ok or not feature_cols:
            return default

        meta_buy_clf  = (self.clf_meta_buy_5m  if use_5m else self.clf_meta_buy)
        meta_sell_clf = (self.clf_meta_sell_5m if use_5m else self.clf_meta_sell)

        try:
            spy_df, vix_df = _get_market_ctx()
            sector_df = _get_sector_ctx(self.symbol)
            features = build_features(df, spy_df=spy_df, vix_df=vix_df, sector_df=sector_df)

            missing_cols = [c for c in feature_cols if c not in features.columns]
            if missing_cols:
                logger.debug(f"[ML] {self.symbol}: missing features at predict (filled 0): {missing_cols}")

            # Build a full-width row aligned to feature_cols; fill missing with 0 (neutral).
            last_idx = features.index[-1]
            row_data = {col: (float(features[col].iloc[-1]) if col in features.columns and not pd.isna(features[col].iloc[-1]) else 0.0)
                        for col in feature_cols}
            row = pd.DataFrame([row_data], index=[last_idx])
            if row.isnull().all(axis=None):
                return default

            X = scaler.transform(row[feature_cols].values)
            buy_prob  = float(buy_clf.predict_proba(X)[0][1])
            sell_prob = float(sell_clf.predict_proba(X)[0][1])

            # Soft sentiment adjustment — replaces the hard binary sentiment gate.
            # Each point of |sentiment_score| adjusts probabilities by ±_SENTIMENT_WEIGHT.
            # sentiment_momentum adds a velocity component (dampened by 0.5).
            _SENTIMENT_WEIGHT = 0.20
            _MOMENTUM_WEIGHT  = 0.08
            sentiment_adj = (sentiment_score * _SENTIMENT_WEIGHT +
                             sentiment_momentum * _MOMENTUM_WEIGHT)
            if sentiment_adj != 0.0:
                buy_prob  = float(np.clip(buy_prob  * (1.0 + sentiment_adj), 0.01, 0.99))
                sell_prob = float(np.clip(sell_prob * (1.0 - sentiment_adj), 0.01, 0.99))

            # ── Meta-classifier: "is the primary likely correct?" ────────────
            meta_conf_buy  = None
            meta_conf_sell = None
            try:
                if meta_buy_clf is not None:
                    X_meta_buy  = np.hstack([X, [[buy_prob]]])
                    meta_conf_buy  = float(meta_buy_clf.predict_proba(X_meta_buy)[0][1])
                if meta_sell_clf is not None:
                    X_meta_sell = np.hstack([X, [[sell_prob]]])
                    meta_conf_sell = float(meta_sell_clf.predict_proba(X_meta_sell)[0][1])
            except Exception as e:
                logger.debug(f"[ML] Meta predict error for {self.symbol}: {e}")

            _GAP = 0.10
            ai_signal = None
            if buy_prob > AI_SIGNAL_THRESHOLD and buy_prob > sell_prob + _GAP:
                ai_signal = "BUY"
            elif sell_prob > AI_SIGNAL_THRESHOLD and sell_prob > buy_prob + _GAP:
                ai_signal = "SELL"

            return {
                "buy_prob": buy_prob, "sell_prob": sell_prob, "ai_signal": ai_signal,
                "meta_conf_buy": meta_conf_buy, "meta_conf_sell": meta_conf_sell,
            }
        except Exception as e:
            logger.error(f"[ML] Predict error for {self.symbol}: {e}")
            return default

    def compute_psi(self, df: pd.DataFrame,
                    spy_df: "pd.DataFrame | None" = None,
                    vix_df: "pd.DataFrame | None" = None,
                    psi_threshold: float = 0.20) -> dict:
        """
        Population Stability Index for feature drift detection.
        Compares current df feature distributions against training quantile baselines.
        PSI < 0.10: no drift  |  0.10–0.20: minor  |  > 0.20: significant drift

        Returns {"psi": {feature: psi_value}, "drifted": [features_above_threshold]}.
        """
        result = {"psi": {}, "drifted": []}
        if not getattr(self, "_train_quantiles", {}):
            return result
        try:
            feat_df = build_features(df, spy_df=spy_df, vix_df=vix_df).dropna()
            if feat_df.empty:
                return result
            n_bins = 10
            for col, q_edges in self._train_quantiles.items():
                if col not in feat_df.columns:
                    continue
                live_vals = feat_df[col].dropna().values
                if len(live_vals) < 20:
                    continue
                # Bin live distribution against training quantile edges
                train_edges = np.unique(q_edges)
                train_counts = np.histogram(live_vals, bins=train_edges)[0]
                # Expected: uniform in each training decile (by construction of quantiles)
                expected_pct = np.ones(len(train_counts)) / len(train_counts)
                actual_pct   = train_counts / (train_counts.sum() + 1e-9)
                # PSI computation with small epsilon to avoid log(0)
                psi_val = float(np.sum((actual_pct - expected_pct) *
                                       np.log((actual_pct + 1e-6) / (expected_pct + 1e-6))))
                result["psi"][col] = round(psi_val, 4)
                if psi_val > psi_threshold:
                    result["drifted"].append(col)
        except Exception as e:
            logger.debug(f"[ML] PSI error for {self.symbol}: {e}")
        return result
