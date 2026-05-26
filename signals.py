from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

ET = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

# Crypto tickers trade 24/7 — no market hours restriction
_CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
                   "ADA-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "DOT-USD"}


def is_crypto(symbol: str) -> bool:
    return symbol.upper() in _CRYPTO_SYMBOLS or symbol.upper().endswith("-USD")


# ── Market hours ──────────────────────────────────────────────────────────────

def is_market_open(symbol: str = "") -> bool:
    if is_crypto(symbol):
        return True
    from datetime import datetime
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    return _MARKET_OPEN <= now.time() <= _MARKET_CLOSE


def market_session(symbol: str = "") -> str:
    if is_crypto(symbol):
        return "open"
    from datetime import datetime
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return "closed"
    t = now.time()
    if t < time(4, 0):
        return "closed"
    if t < _MARKET_OPEN:
        return "pre"
    if t <= _MARKET_CLOSE:
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


def _quality_score(rule_hits: int, ai_confidence: float | None,
                   trend: str, action: str, vol_spike: bool) -> int:
    score = rule_hits  # 0–3
    if ai_confidence and ai_confidence > 0.60:
        score += 1
    if (action == "BUY" and trend == "up") or (action == "SELL" and trend == "down"):
        score += 1
    if vol_spike:
        score += 1
    return max(1, min(score, 5))


def _stars(score: int) -> str:
    return "⭐" * score + "☆" * (5 - score)


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
        macd_bull = (float(macd_line.iloc[-2]) < float(signal_line.iloc[-2])
                     and float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]))
        macd_bear = (float(macd_line.iloc[-2]) > float(signal_line.iloc[-2])
                     and float(macd_line.iloc[-1]) < float(signal_line.iloc[-1]))
        ema_bull = (float(ema9.iloc[-2]) < float(ema21.iloc[-2])
                    and float(ema9.iloc[-1]) > float(ema21.iloc[-1]))
        ema_bear = (float(ema9.iloc[-2]) > float(ema21.iloc[-2])
                    and float(ema9.iloc[-1]) < float(ema21.iloc[-1]))

        buy_hits = [rsi_now < 45, macd_bull, ema_bull]
        sell_hits = [rsi_now > 55, macd_bear, ema_bear]

        if sum(buy_hits) >= 2:
            return "BUY", df
        if sum(sell_hits) >= 2:
            return "SELL", df
        return None, df
    except Exception as exc:
        print(f"[signals] 1h bias error for {symbol}: {exc}")
        return None, None


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

        close = df["Close"].squeeze()
        rsi = _rsi(close)
        macd_line, signal_line = _macd(close)
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        atr = _atr(df)

        rsi_now = float(rsi.iloc[-1])
        price = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1])

        has_vol_spike = _vol_spike(df)

        macd_bull = (float(macd_line.iloc[-2]) < float(signal_line.iloc[-2])
                     and float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]))
        macd_bear = (float(macd_line.iloc[-2]) > float(signal_line.iloc[-2])
                     and float(macd_line.iloc[-1]) < float(signal_line.iloc[-1]))
        ema_bull = (float(ema9.iloc[-2]) < float(ema21.iloc[-2])
                    and float(ema9.iloc[-1]) > float(ema21.iloc[-1]))
        ema_bear = (float(ema9.iloc[-2]) > float(ema21.iloc[-2])
                    and float(ema9.iloc[-1]) < float(ema21.iloc[-1]))

        buy_hits = [rsi_now < 45, macd_bull, ema_bull]
        sell_hits = [rsi_now > 55, macd_bear, ema_bear]

        trend = _trend(df_1h) if df_1h is not None and not df_1h.empty else "unknown"

        sl_dist = atr_val * 1.5
        tp_dist = atr_val * 2.5
        rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0

        reasons_buy = []
        if rsi_now < 45:
            reasons_buy.append(f"RSI oversold ({rsi_now:.1f})")
        if macd_bull:
            reasons_buy.append("MACD bullish crossover")
        if ema_bull:
            reasons_buy.append("EMA9 crossed above EMA21")
        if has_vol_spike:
            reasons_buy.append("Volume spike (+50% above avg)")

        reasons_sell = []
        if rsi_now > 55:
            reasons_sell.append(f"RSI overbought ({rsi_now:.1f})")
        if macd_bear:
            reasons_sell.append("MACD bearish crossover")
        if ema_bear:
            reasons_sell.append("EMA9 crossed below EMA21")
        if has_vol_spike:
            reasons_sell.append("Volume spike (+50% above avg)")

        def _make(action: str, hits: int, reasons: list) -> dict:
            is_buy = action == "BUY"
            tp = round(price + tp_dist if is_buy else price - tp_dist, 2)
            sl = round(price - sl_dist if is_buy else price + sl_dist, 2)
            tp_pct = round((tp - price) / price * 100, 2)
            sl_pct = round((sl - price) / price * 100, 2)
            return {
                "symbol": symbol, "action": action, "price": price,
                "tp": tp, "sl": sl, "tp_pct": tp_pct, "sl_pct": sl_pct, "rr": rr,
                "rsi": rsi_now, "trend": trend, "atr": atr_val,
                "vol_spike": has_vol_spike, "reasons": reasons,
                "strength": "RULE", "ai_confidence": None, "quality": 0,
            }

        if sum(buy_hits) >= 2:
            return _make("BUY", sum(buy_hits), reasons_buy), df
        if sum(sell_hits) >= 2:
            return _make("SELL", sum(sell_hits), reasons_sell), df

        return None, df

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
    rule_action = rule_sig["action"] if rule_sig else None

    sig = None

    if rule_action and ai_signal and rule_action == ai_signal:
        sig = dict(rule_sig)
        sig["strength"] = "STRONG"
        sig["ai_confidence"] = buy_prob if rule_action == "BUY" else sell_prob
    elif rule_action:
        sig = dict(rule_sig)
        sig["strength"] = "RULE"
        sig["ai_confidence"] = buy_prob if rule_action == "BUY" else sell_prob
    elif ai_signal:
        prob = buy_prob if ai_signal == "BUY" else sell_prob
        if prob is not None and prob >= 0.70:
            atr_val = 0.0
            trend = "unknown"
            if df_5m is not None and not df_5m.empty:
                atr_series = _atr(df_5m)
                atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            if df_1h is not None and not df_1h.empty:
                trend = _trend(df_1h)
            is_buy = ai_signal == "BUY"
            tp_dist = atr_val * 2.5
            sl_dist = atr_val * 1.5
            tp = round(price + tp_dist if is_buy else price - tp_dist, 2)
            sl = round(price - sl_dist if is_buy else price + sl_dist, 2)
            tp_pct = round((tp - price) / price * 100, 2)
            sl_pct = round((sl - price) / price * 100, 2)
            rr = round(tp_dist / sl_dist, 1) if sl_dist > 0 else 0
            sig = {
                "symbol": symbol, "action": ai_signal, "price": price,
                "tp": tp, "sl": sl, "tp_pct": tp_pct, "sl_pct": sl_pct, "rr": rr,
                "rsi": rsi, "trend": trend, "atr": atr_val,
                "vol_spike": False,
                "reasons": [f"AI model confidence {prob * 100:.0f}%"],
                "strength": "AI", "ai_confidence": prob, "quality": 0,
            }

    if sig is None:
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
        sig["rsi"] < 45 if sig["action"] == "BUY" else sig["rsi"] > 55,
        any("MACD" in r for r in sig["reasons"]),
        any("EMA" in r for r in sig["reasons"]),
    ])
    sig["quality"] = _quality_score(
        rule_hits,
        sig.get("ai_confidence"),
        sig.get("trend", "unknown"),
        sig["action"],
        sig.get("vol_spike", False),
    )
    sig["stars"] = _stars(sig["quality"])

    return sig
