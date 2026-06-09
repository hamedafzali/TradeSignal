from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ET  = ZoneInfo("America/New_York")
CET = ZoneInfo("Europe/Berlin")

# US NYSE/NASDAQ
_US_OPEN  = time(9, 30)
_US_CLOSE = time(16, 0)

# XETRA / Frankfurt (Deutsche Börse) — CET/CEST
_XETRA_OPEN  = time(9, 0)
_XETRA_CLOSE = time(17, 30)

# Crypto tickers trade 24/7 — no market hours restriction
_CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
                   "ADA-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "DOT-USD"}

# Suffixes that indicate a European-listed stock (XETRA hours apply to all)
_XETRA_SUFFIXES = (".DE", ".F", ".XETRA", ".AS", ".PA", ".MI", ".BR", ".LS", ".MC",
                   ".SW", ".CO", ".ST", ".OL", ".HE", ".VI")


def is_crypto(symbol: str) -> bool:
    return symbol.upper() in _CRYPTO_SYMBOLS or symbol.upper().endswith("-USD")


def detect_market(symbol: str) -> str:
    """Auto-detect market from symbol suffix — crypto, xetra (European), or us."""
    s = symbol.upper()
    if is_crypto(s):
        return "crypto"
    for suffix in _XETRA_SUFFIXES:
        if s.endswith(suffix.upper()):
            return "xetra"
    return "us"


def get_symbol_market(symbol: str) -> str:
    """Return the market for a symbol, checking DB override first, then auto-detect."""
    try:
        from database import get_symbol_market as _db_market
        stored = _db_market(symbol)
        if stored:
            return stored
    except Exception:
        pass
    return detect_market(symbol)


# ── Market hours ──────────────────────────────────────────────────────────────

def is_market_open(symbol: str = "") -> bool:
    from datetime import datetime
    market = get_symbol_market(symbol)
    if market == "crypto":
        return True
    if market == "xetra":
        now = datetime.now(CET)
        if now.weekday() >= 5:
            return False
        return _XETRA_OPEN <= now.time() <= _XETRA_CLOSE
    # us
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return _US_OPEN <= now.time() <= _US_CLOSE


def market_session(symbol: str = "") -> str:
    from datetime import datetime
    market = get_symbol_market(symbol)
    if market == "crypto":
        return "open"
    if market == "xetra":
        now = datetime.now(CET)
        if now.weekday() >= 5:
            return "closed"
        t = now.time()
        if t < time(8, 0):
            return "closed"
        if t < _XETRA_OPEN:
            return "pre"
        if t <= _XETRA_CLOSE:
            return "open"
        return "after"
    # us
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return "closed"
    t = now.time()
    if t < time(4, 0):
        return "closed"
    if t < _US_OPEN:
        return "pre"
    if t <= _US_CLOSE:
        return "open"
    return "after"


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _ema200(series: pd.Series) -> float | None:
    if len(series) < 200:
        return None
    return float(series.ewm(span=200, adjust=False).mean().iloc[-1])


def _trend(df_1h: pd.DataFrame) -> str:
    """Trend based on EMA200 on hourly data."""
    close = df_1h["Close"].squeeze()
    if len(close) < 200:
        return "unknown"
    ema200 = close.ewm(span=200, adjust=False).mean()
    price = float(close.iloc[-1])
    ema_val = float(ema200.iloc[-1])
    if price > ema_val * 1.005:
        return "up"
    if price < ema_val * 0.995:
        return "down"
    return "sideways"


def _vol_spike(df: pd.DataFrame) -> bool:
    vol = df["Volume"].squeeze()
    if len(vol) < 20:
        return False
    avg = float(vol.rolling(20).mean().iloc[-1])
    current = float(vol.iloc[-1])
    return current > avg * 1.5


def _vol_sufficient(df: pd.DataFrame) -> bool:
    """Hard gate: current bar volume must be >= 110% of 20-period average."""
    vol = df["Volume"].squeeze()
    if len(vol) < 20:
        return True  # not enough history — don't block
    avg = float(vol.rolling(20).mean().iloc[-1])
    current = float(vol.iloc[-1])
    return current >= avg * 1.1


def _quality_score(rule_hits: int, ai_confidence: float | None,
                   trend: str, action: str, vol_spike: bool,
                   meta_confidence: float | None = None) -> int:
    score = rule_hits  # 0–3
    if ai_confidence and ai_confidence > 0.60:
        score += 1
    if (action == "BUY" and trend == "up") or (action == "SELL" and trend == "down"):
        score += 1
    if vol_spike:
        score += 1
    # Meta-classifier bonus: when secondary model confirms primary correctness
    if meta_confidence is not None and meta_confidence > 0.60:
        score += 1
    return max(1, min(score, 6))


def _stars(score: int) -> str:
    return "⭐" * score + "☆" * max(0, 6 - score)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _kelly_size(win_rate_pct: float, rr: float,
                max_pct: float = 10.0, half: bool = True) -> float:
    """
    Half-Kelly position size as a percentage of portfolio.

    f = (b*p - q) / b   where b=RR, p=win_rate, q=1-p
    Half-Kelly: multiply by 0.5 for safety margin.
    Clamped to [0, max_pct].

    Returns 0.0 when the signal has no positive expectancy.
    """
    if rr <= 0 or win_rate_pct <= 0:
        return 0.0
    p = win_rate_pct / 100.0
    q = 1.0 - p
    b = rr
    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0.0
    size = kelly * (0.5 if half else 1.0) * 100.0
    return round(_clamp(size, 0.0, max_pct), 2)


def _adaptive_tp_sl(price: float, atr_val: float, symbol: str) -> tuple[float, float, str | None]:
    """
    Returns (sl_mult, tp_mult, note). Falls back to defaults when there
    is not enough recent resolved-outcome data.
    """
    base_sl, base_tp = 1.5, 2.5
    if atr_val <= 0 or price <= 0:
        return base_sl, base_tp, None
    try:
        from database import get_symbol_outcome_profile
        profile = get_symbol_outcome_profile(symbol)
    except Exception:
        return base_sl, base_tp, None

    if profile.get("sample_size", 0) < 8:
        return base_sl, base_tp, None

    atr_pct = atr_val / price * 100
    if atr_pct <= 0:
        return base_sl, base_tp, None

    avg_fav = profile.get("avg_favorable_pct")
    avg_adv = profile.get("avg_adverse_pct")
    if avg_fav is None or avg_adv is None or avg_adv <= 0:
        return base_sl, base_tp, None

    target_sl_pct = _clamp(avg_adv * 1.1, 0.35, 4.0)
    target_tp_pct = _clamp(max(avg_fav * 0.85, target_sl_pct * 1.2), 0.5, 6.0)

    timeout_rate = float(profile.get("timeout_rate", 0.0) or 0.0)
    ambiguous_rate = float(profile.get("ambiguous_rate", 0.0) or 0.0)
    if timeout_rate >= 0.25:
        target_tp_pct *= 0.9
    if ambiguous_rate >= 0.20:
        target_sl_pct *= 1.1
        target_tp_pct *= 1.05

    sl_mult = _clamp(target_sl_pct / atr_pct, 1.0, 2.8)
    tp_mult = _clamp(target_tp_pct / atr_pct, 1.4, 4.2)
    note = (
        f"Adaptive TP/SL tuned from {profile['sample_size']} outcomes "
        f"(MFE {avg_fav:.2f}% / MAE {avg_adv:.2f}%)"
    )
    return sl_mult, tp_mult, note


# ── 1h bias for multi-timeframe confirmation ──────────────────────────────────

def get_1h_bias(symbol: str) -> tuple[str | None, pd.DataFrame | None]:
    """
    Returns (bias, df_1h) where bias is 'BUY', 'SELL', or None.
    Uses same RSI/MACD/EMA logic on 1h candles.
    Returns the df so the caller can pass it to _trend().
    """
    try:
        period = "60d" if is_crypto(symbol) else "60d"
        df = yf.download(symbol, period=period, interval="1h",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None, None

        close = df["Close"].squeeze()
        rsi = _rsi(close)
        macd_line, signal_line = _macd(close)
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()

        rsi_now = float(rsi.iloc[-1])
        # State-based: is MACD/EMA currently positioned bullish or bearish?
        # (not requiring a fresh crossover — that fires too rarely)
        macd_bull = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])
        macd_bear = float(macd_line.iloc[-1]) < float(signal_line.iloc[-1])
        ema_bull  = float(ema9.iloc[-1]) > float(ema21.iloc[-1])
        ema_bear  = float(ema9.iloc[-1]) < float(ema21.iloc[-1])

        buy_hits = [rsi_now < 40, macd_bull, ema_bull]
        sell_hits = [rsi_now > 60, macd_bear, ema_bear]

        if sum(buy_hits) >= 2:
            return "BUY", df
        if sum(sell_hits) >= 2:
            return "SELL", df
        return None, df
    except Exception as exc:
        print(f"[signals] 1h bias error for {symbol}: {exc}")
        return None, None


def get_bias_from_df(df: pd.DataFrame) -> str | None:
    """Evaluate multi-timeframe bias from historical candles."""
    if df.empty or len(df) < 50:
        return None

    close = df["Close"].squeeze()
    rsi = _rsi(close)
    macd_line, signal_line = _macd(close)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    rsi_now = float(rsi.iloc[-1])
    macd_bull = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])
    macd_bear = float(macd_line.iloc[-1]) < float(signal_line.iloc[-1])
    ema_bull = float(ema9.iloc[-1]) > float(ema21.iloc[-1])
    ema_bear = float(ema9.iloc[-1]) < float(ema21.iloc[-1])

    buy_hits = [rsi_now < 40, macd_bull, ema_bull]
    sell_hits = [rsi_now > 60, macd_bear, ema_bear]

    if sum(buy_hits) >= 2:
        return "BUY"
    if sum(sell_hits) >= 2:
        return "SELL"
    return None


def signal_from_df(symbol: str, df: pd.DataFrame,
                   df_1h: pd.DataFrame | None = None) -> dict | None:
    """Pure dataframe-based version of the live rule signal logic."""
    if df.empty or len(df) < 50:
        return None

    close = df["Close"].squeeze()
    rsi = _rsi(close)
    macd_line, signal_line = _macd(close)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    atr = _atr(df)

    rsi_now = float(rsi.iloc[-1])
    price = float(close.iloc[-1])
    atr_val = float(atr.iloc[-1])
    if pd.isna(atr_val) or atr_val <= 0:
        return None

    has_vol_spike = _vol_spike(df)
    has_vol = _vol_sufficient(df)
    macd_bull = float(macd_line.iloc[-1]) > float(signal_line.iloc[-1])
    macd_bear = float(macd_line.iloc[-1]) < float(signal_line.iloc[-1])
    ema_bull = float(ema9.iloc[-1]) > float(ema21.iloc[-1])
    ema_bear = float(ema9.iloc[-1]) < float(ema21.iloc[-1])

    # 5m medium-term trend: EMA50 slope over last 3 bars
    # Prevents buying a falling knife when RSI oversold but price still in freefall
    ema50 = close.ewm(span=50, adjust=False).mean()
    # Default False: insufficient data should block the guard, not pass it
    ema50_rising  = float(ema50.iloc[-1]) > float(ema50.iloc[-4]) if len(ema50) >= 4 else False
    ema50_falling = float(ema50.iloc[-1]) < float(ema50.iloc[-4]) if len(ema50) >= 4 else False

    # RSI thresholds: 40/60 on 5m candles — 35/65 was too tight and killed rule signals
    rsi_oversold  = rsi_now < 40
    rsi_overbought = rsi_now > 60
    buy_hits  = [rsi_oversold,  macd_bull, ema_bull]
    sell_hits = [rsi_overbought, macd_bear, ema_bear]
    trend = _trend(df_1h) if df_1h is not None and not df_1h.empty else "unknown"

    sl_mult, tp_mult, adaptive_note = _adaptive_tp_sl(price, atr_val, symbol)
    sl_dist = atr_val * sl_mult
    tp_dist = atr_val * tp_mult
    rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0

    reasons_buy = []
    if rsi_oversold:
        reasons_buy.append(f"RSI oversold ({rsi_now:.1f})")
    if macd_bull:
        reasons_buy.append("MACD above signal (bullish)")
    if ema_bull:
        reasons_buy.append("EMA9 above EMA21 (bullish)")
    if has_vol_spike:
        reasons_buy.append("Volume spike (+50% above avg)")
    if adaptive_note:
        reasons_buy.append(adaptive_note)

    reasons_sell = []
    if rsi_overbought:
        reasons_sell.append(f"RSI overbought ({rsi_now:.1f})")
    if macd_bear:
        reasons_sell.append("MACD below signal (bearish)")
    if ema_bear:
        reasons_sell.append("EMA9 below EMA21 (bearish)")
    if has_vol_spike:
        reasons_sell.append("Volume spike (+50% above avg)")
    if adaptive_note:
        reasons_sell.append(adaptive_note)

    def _make(action: str, reasons: list[str]) -> dict:
        is_buy = action == "BUY"
        tp = round(price + tp_dist if is_buy else price - tp_dist, 2)
        sl = round(price - sl_dist if is_buy else price + sl_dist, 2)
        # Always positive: tp_pct = profit%, sl_pct = loss%
        tp_pct = round(abs(tp - price) / price * 100, 2)
        sl_pct = round(abs(sl - price) / price * 100, 2)
        # Half-Kelly size using symbol walk-forward win rate when available
        try:
            from database import get_wf_results as _wfr
            wf = {r["symbol"]: r for r in _wfr()}
            wr = wf.get(symbol, {}).get("rule_win_rate") or 50.0
        except Exception:
            wr = 50.0
        size_pct = _kelly_size(wr, rr)
        return {
            "symbol": symbol, "action": action, "price": price,
            "tp": tp, "sl": sl, "tp_pct": tp_pct, "sl_pct": sl_pct, "rr": rr,
            "rsi": rsi_now, "trend": trend, "atr": atr_val,
            "vol_spike": has_vol_spike, "reasons": reasons,
            "strength": "RULE", "ai_confidence": None, "quality": 0,
            "sl_mult": round(sl_mult, 2), "tp_mult": round(tp_mult, 2),
            "suggested_size_pct": size_pct,
        }

    # RSI is mandatory — prevents pure trend-chasing signals (MACD+EMA agree but RSI neutral)
    # EMA50 slope guard: don't buy when 5m medium-term trend is still falling (falling knife)
    if rsi_oversold and sum(buy_hits) >= 2 and has_vol and ema50_rising:
        return _make("BUY", reasons_buy)
    if rsi_overbought and sum(sell_hits) >= 2 and has_vol and ema50_falling:
        return _make("SELL", reasons_sell)
    return None


# ── Main signal function ──────────────────────────────────────────────────────

def get_signal(symbol: str, df_1h: pd.DataFrame | None = None) -> tuple[dict | None, pd.DataFrame | None]:
    """
    Returns (signal, df_5m) or (None, df_5m).
    For crypto, uses 1h candles instead of 5m (more reliable data).
    """
    try:
        if is_crypto(symbol):
            interval = "1h"
            period = "30d"
            min_bars = 50
        else:
            interval = "5m"
            period = "5d"
            min_bars = 50

        df = yf.download(symbol, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < min_bars:
            return None, None

        return signal_from_df(symbol, df, df_1h=df_1h), df

    except Exception as exc:
        print(f"[signals] Error fetching {symbol}: {exc}")
        return None, None


def combine_signals(rule_sig: dict | None, ml_result: dict,
                    symbol: str, price: float, rsi: float,
                    df_5m: pd.DataFrame | None = None,
                    df_1h: pd.DataFrame | None = None,
                    bias_1h: str | None = None) -> dict | None:
    ai_signal = ml_result.get("ai_signal")
    buy_prob = ml_result.get("buy_prob")
    sell_prob = ml_result.get("sell_prob")
    meta_conf_buy  = ml_result.get("meta_conf_buy")
    meta_conf_sell = ml_result.get("meta_conf_sell")
    rule_action = rule_sig["action"] if rule_sig else None

    sig = None

    if rule_action and ai_signal and rule_action == ai_signal:
        sig = dict(rule_sig)
        sig["strength"] = "STRONG"
        sig["ai_confidence"] = buy_prob if rule_action == "BUY" else sell_prob
        # Attach meta-confidence so downstream can use it for quality scoring
        sig["meta_confidence"] = (meta_conf_buy if rule_action == "BUY" else meta_conf_sell)
    elif rule_action:
        sig = dict(rule_sig)
        sig["strength"] = "RULE"
        sig["ai_confidence"] = buy_prob if rule_action == "BUY" else sell_prob
        sig["meta_confidence"] = (meta_conf_buy if rule_action == "BUY" else meta_conf_sell)
    elif ai_signal:
        prob = buy_prob if ai_signal == "BUY" else sell_prob
        if prob is not None and prob >= 0.70:
            # RSI gate: AI-only BUY must not be overbought; AI-only SELL must not be oversold
            if ai_signal == "BUY" and rsi > 60:
                print(f"[signals] AI BUY blocked on {symbol}: RSI {rsi:.1f} overbought")
                return None
            if ai_signal == "SELL" and rsi < 40:
                print(f"[signals] AI SELL blocked on {symbol}: RSI {rsi:.1f} oversold")
                return None
            # AI-only signals require 1h bias agreement — no bias means no confirmation
            if bias_1h is None:
                print(f"[signals] AI {ai_signal} blocked on {symbol}: no 1h bias confirmation")
                return None
            if bias_1h != ai_signal:
                # already handled below but guard here too
                return None
            atr_val = 0.0
            trend = "unknown"
            if df_5m is not None and not df_5m.empty:
                atr_series = _atr(df_5m)
                atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            if df_1h is not None and not df_1h.empty:
                trend = _trend(df_1h)
            is_buy = ai_signal == "BUY"
            sl_mult, tp_mult, adaptive_note = _adaptive_tp_sl(price, atr_val, symbol)
            tp_dist = atr_val * tp_mult
            sl_dist = atr_val * sl_mult
            tp = round(price + tp_dist if is_buy else price - tp_dist, 2)
            sl = round(price - sl_dist if is_buy else price + sl_dist, 2)
            # Always positive: tp_pct = profit%, sl_pct = loss%
            tp_pct = round(abs(tp - price) / price * 100, 2)
            sl_pct = round(abs(sl - price) / price * 100, 2)
            rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0
            try:
                from database import get_wf_results as _wfr
                wf = {r["symbol"]: r for r in _wfr()}
                ai_wr = wf.get(symbol, {}).get("ai_win_rate") or 50.0
            except Exception:
                ai_wr = 50.0
            sig = {
                "symbol": symbol, "action": ai_signal, "price": price,
                "tp": tp, "sl": sl, "tp_pct": tp_pct, "sl_pct": sl_pct, "rr": rr,
                "rsi": rsi, "trend": trend, "atr": atr_val,
                "vol_spike": False,
                "reasons": [f"AI model confidence {prob * 100:.0f}%"] + ([adaptive_note] if adaptive_note else []),
                "strength": "AI", "ai_confidence": prob, "quality": 0,
                "sl_mult": round(sl_mult, 2), "tp_mult": round(tp_mult, 2),
                "suggested_size_pct": _kelly_size(ai_wr, rr),
                "meta_confidence": (meta_conf_buy if ai_signal == "BUY" else meta_conf_sell),
            }

    if sig is None:
        return None

    # EMA200 hard filter: block counter-trend signals
    if df_1h is not None and not df_1h.empty:
        ema200_trend = _trend(df_1h)
        if ema200_trend == "down" and sig["action"] == "BUY":
            print(f"[signals] EMA200 blocked BUY on {symbol} (price below EMA200)")
            return None
        if ema200_trend == "up" and sig["action"] == "SELL":
            print(f"[signals] EMA200 blocked SELL on {symbol} (price above EMA200)")
            return None

    # Multi-timeframe filter: if 1h bias exists and disagrees, suppress
    if bias_1h is not None and bias_1h != sig["action"]:
        print(f"[signals] MTF suppressed {sig['action']} on {symbol} "
              f"(1h bias={bias_1h})")
        return None

    # Tag as MTF-confirmed when 1h bias agrees
    if bias_1h == sig["action"]:
        sig["mtf_confirmed"] = True
        if "MTF confirmed" not in " ".join(sig.get("reasons", [])):
            sig.setdefault("reasons", []).append("Multi-timeframe confirmed (1h+5m)")
    else:
        sig["mtf_confirmed"] = False

    # Compute final quality score
    rule_hits = sum([
        sig["rsi"] < 40 if sig["action"] == "BUY" else sig["rsi"] > 60,
        any("MACD" in r for r in sig["reasons"]),
        any("EMA" in r for r in sig["reasons"]),
    ])
    sig["quality"] = _quality_score(
        rule_hits,
        sig.get("ai_confidence"),
        sig.get("trend", "unknown"),
        sig["action"],
        sig.get("vol_spike", False),
        meta_confidence=sig.get("meta_confidence"),
    )
    sig["stars"] = _stars(sig["quality"])

    return sig
