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
RETRAIN_OUTCOME_THRESHOLD = int(os.getenv("RETRAIN_OUTCOME_THRESHOLD", "3"))
AI_SIGNAL_THRESHOLD = float(os.getenv("AI_SIGNAL_THRESHOLD", "0.65"))

# ── Market context cache ──────────────────────────────────────────────────────
_mkt_cache: dict = {"ts": 0.0, "spy": None, "vix": None}
_MKT_TTL = 300  # seconds


def _get_market_ctx(period: str = "60d") -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """SPY (1h) + VIX (1d) with 5-min in-memory cache. Both optional on failure."""
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
    except Exception as e:
        logger.debug(f"[ML] SPY fetch failed: {e}")
    try:
        vix = yf.download("^VIX", period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
        if vix.empty:
            vix = None
    except Exception as e:
        logger.debug(f"[ML] VIX fetch failed: {e}")
    _mkt_cache.update({"ts": now, "spy": spy, "vix": vix})
    return spy, vix


def get_predict_market_context() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Public export — bot.py uses this so feature capture shares the same cache."""
    return _get_market_ctx()


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
                   vix_df: pd.DataFrame | None = None) -> pd.DataFrame:
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    ag = gain.ewm(com=13, min_periods=14).mean()
    al = loss.ewm(com=13, min_periods=14).mean()
    feat["rsi"] = 100 - (100 / (1 + ag / al))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - sig) / close  # normalized

    # EMA ratio
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feat["ema_ratio"] = ema9 / ema21 - 1

    # Price momentum
    for p in [3, 5, 10, 20]:
        feat[f"roc_{p}"] = close.pct_change(p)

    # Bollinger Band position
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat["bb_pos"] = (close - sma20) / (2 * std20 + 1e-9)

    # Volume ratio
    vol_ma = volume.rolling(20).mean()
    feat["vol_ratio"] = volume / (vol_ma + 1e-9)

    # ATR (normalized)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    feat["atr_pct"] = tr.rolling(14).mean() / (close + 1e-9)

    # ── Market context (optional) ────────────────────────────────────────────
    # SPY relative strength: is this stock leading or lagging the broad market?
    if spy_df is not None and not spy_df.empty:
        try:
            spy_c = _align(spy_df["Close"].squeeze(), feat.index)
            feat["rel_str_1h"] = close.pct_change(1) - spy_c.pct_change(1)
            feat["rel_str_8h"] = close.pct_change(8) - spy_c.pct_change(8)
            spy_ema50 = spy_c.ewm(span=50, adjust=False).mean()
            feat["spy_regime"] = (spy_c >= spy_ema50).astype(float) * 2 - 1  # +1 bull / -1 bear
            spy_sma20 = spy_c.rolling(20).mean()
            spy_std20 = spy_c.rolling(20).std()
            feat["spy_bb_pos"] = (spy_c - spy_sma20) / (2 * spy_std20 + 1e-9)
        except Exception as e:
            logger.debug(f"[ML] SPY feature error: {e}")

    # VIX regime: is fear elevated relative to recent average?
    if vix_df is not None and not vix_df.empty:
        try:
            vix_c = _align(vix_df["Close"].squeeze(), feat.index)
            vix_ma = vix_c.rolling(20, min_periods=1).mean()
            feat["vix_norm"] = vix_c / (vix_ma + 1e-9)
        except Exception as e:
            logger.debug(f"[ML] VIX feature error: {e}")

    return feat


def build_labels(df: pd.DataFrame, forward: int = 8, threshold: float = 0.008) -> pd.Series:
    """
    Label each bar by forward return over `forward` 1h bars.
    Default 8h / 0.8% matches ATR×2.5 TP horizon better than the old 3h/0.3%.
    """
    close = df["Close"].squeeze()
    ret = close.shift(-forward) / close - 1
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[ret > threshold] = 1
    labels[ret < -threshold] = -1
    return labels


class StockModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.clf_buy = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.clf_sell = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
        self.cal_buy: CalibratedClassifierCV | None = None
        self.cal_sell: CalibratedClassifierCV | None = None
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.trained = False
        self.trained_at = 0.0
        self.train_samples = 0
        self.outcome_samples = 0
        self.outcome_count_at_train = 0  # resolved outcome count when last trained
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
        new_outcomes = current_outcome_count - self.outcome_count_at_train
        if new_outcomes >= RETRAIN_OUTCOME_THRESHOLD:
            return True
        return (time.time() - self.trained_at) > RETRAIN_EVERY

    def train(self, outcome_data: list[dict] | None = None, _log: bool = True) -> bool:
        """
        Train on 60 days of hourly data, then blend in real outcome samples.
        outcome_data: list of {"features": {col: val, ...}, "label": int}
        """
        logger.info(f"[ML] Training {self.symbol}...")
        try:
            df = yf.download(self.symbol, period="60d", interval="1h", progress=False, auto_adjust=True)
            if len(df) < 200:
                logger.warning(f"[ML] Too little data for {self.symbol}")
                return False

            spy_df, vix_df = _get_market_ctx()
            features = build_features(df, spy_df=spy_df, vix_df=vix_df)
            labels = build_labels(df)
            combined = pd.concat([features, labels.rename("label")], axis=1).dropna()

            if len(combined) < 100:
                return False

            self.feature_cols = [c for c in combined.columns if c != "label"]
            X = combined[self.feature_cols].values
            y = combined["label"].values

            # Blend in confirmed real outcomes — repeat 5x so live feedback
            # has enough weight against ~600 synthetic rows
            if outcome_data:
                extra_rows, extra_labels = [], []
                for item in outcome_data:
                    row = [item["features"].get(col, 0.0) for col in self.feature_cols]
                    extra_rows.extend([row] * 5)
                    extra_labels.extend([item["label"]] * 5)
                if extra_rows:
                    X = np.vstack([X, extra_rows])
                    y = np.concatenate([y, extra_labels])
                    logger.info(f"[ML] {self.symbol}: blended {len(outcome_data)} real outcomes (5x weight)")

            X_scaled = self.scaler.fit_transform(X)

            # Platt calibration: hold out 20% for calibration fit so probabilities
            # are well-calibrated (65% confidence ≈ 65% real win rate)
            n_cal = max(20, len(X_scaled) // 5)
            if n_cal < len(X_scaled) - 50:
                X_main, X_cal = X_scaled[:-n_cal], X_scaled[-n_cal:]
                y_main, y_cal = y[:-n_cal], y[-n_cal:]
            else:
                X_main, X_cal = X_scaled, X_scaled
                y_main, y_cal = y, y

            self.clf_buy.fit(X_main, (y_main == 1).astype(int))
            self.clf_sell.fit(X_main, (y_main == -1).astype(int))

            try:
                self.cal_buy = CalibratedClassifierCV(self.clf_buy, method="sigmoid", cv="prefit")
                self.cal_buy.fit(X_cal, (y_cal == 1).astype(int))
                self.cal_sell = CalibratedClassifierCV(self.clf_sell, method="sigmoid", cv="prefit")
                self.cal_sell.fit(X_cal, (y_cal == -1).astype(int))
            except Exception as e:
                logger.debug(f"[ML] Calibration skipped for {self.symbol}: {e}")
                self.cal_buy = None
                self.cal_sell = None

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

            logger.info(f"[ML] {self.symbol} trained on {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"[ML] Train error for {self.symbol}: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> dict:
        """Returns buy_prob, sell_prob, ai_signal for the latest candle."""
        default = {"buy_prob": None, "sell_prob": None, "ai_signal": None}
        if not self.trained or not self.feature_cols:
            return default
        try:
            spy_df, vix_df = _get_market_ctx()
            features = build_features(df, spy_df=spy_df, vix_df=vix_df)
            row = features[self.feature_cols].dropna().iloc[-1:]
            if row.empty:
                return default

            X = self.scaler.transform(row.values)
            buy_clf = self.cal_buy if self.cal_buy is not None else self.clf_buy
            sell_clf = self.cal_sell if self.cal_sell is not None else self.clf_sell
            buy_prob = float(buy_clf.predict_proba(X)[0][1])
            sell_prob = float(sell_clf.predict_proba(X)[0][1])

            # Require a confidence gap: prevents mixed signals when both classifiers
            # are uncertain (e.g. buy=0.67, sell=0.66 — model is not actually decided)
            _GAP = 0.10
            ai_signal = None
            if buy_prob > AI_SIGNAL_THRESHOLD and buy_prob > sell_prob + _GAP:
                ai_signal = "BUY"
            elif sell_prob > AI_SIGNAL_THRESHOLD and sell_prob > buy_prob + _GAP:
                ai_signal = "SELL"

            return {"buy_prob": buy_prob, "sell_prob": sell_prob, "ai_signal": ai_signal}
        except Exception as e:
            logger.error(f"[ML] Predict error for {self.symbol}: {e}")
            return default
