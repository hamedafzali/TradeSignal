"""
Backtesting engine.
Simulates rule-based signals on historical OHLCV data and reports
win rate, average P&L, max drawdown, and per-trade results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import yfinance as yf

from signals import _rsi, _macd, _atr, _vol_spike, is_crypto


@dataclass
class Trade:
    symbol: str
    action: str          # 'BUY' | 'SELL'
    entry_date: str
    entry_price: float
    tp: float
    sl: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'TP' | 'SL' | 'TIMEOUT'
    pnl_pct: Optional[float] = None


def _fetch(symbol: str, years: int = 2) -> pd.DataFrame:
    period = f"{years * 365}d"
    interval = "1d"
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def _simulate(symbol: str, df: pd.DataFrame,
              sl_mult: float = 1.5, tp_mult: float = 2.5,
              timeout_bars: int = 20) -> list[Trade]:
    """
    Walk-forward simulation on daily bars.
    Signal: 2-of-3 indicators (RSI, MACD, EMA9/21 crossover).
    Exit: next-day TP/SL check on High/Low; timeout after `timeout_bars` days.
    """
    if len(df) < 60:
        return []

    close = df["Close"].squeeze()
    rsi = _rsi(close)
    macd_line, signal_line = _macd(close)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    atr = _atr(df)

    trades: list[Trade] = []
    in_trade = False
    current: Trade | None = None

    for i in range(30, len(df) - 1):
        date_str = str(df.index[i])[:10]
        price = float(close.iloc[i])
        atr_val = float(atr.iloc[i])
        if pd.isna(atr_val) or atr_val == 0:
            continue

        # ── Manage open trade ────────────────────────────────────────────────
        if in_trade and current is not None:
            day_high = float(df["High"].iloc[i])
            day_low = float(df["Low"].iloc[i])
            bars_held = trades[-1] if trades else None

            if current.action == "BUY":
                if day_high >= current.tp:
                    current.exit_date = date_str
                    current.exit_price = current.tp
                    current.exit_reason = "TP"
                    current.pnl_pct = (current.tp - current.entry_price) / current.entry_price * 100
                    in_trade = False
                elif day_low <= current.sl:
                    current.exit_date = date_str
                    current.exit_price = current.sl
                    current.exit_reason = "SL"
                    current.pnl_pct = (current.sl - current.entry_price) / current.entry_price * 100
                    in_trade = False
            else:  # SELL
                if day_low <= current.tp:
                    current.exit_date = date_str
                    current.exit_price = current.tp
                    current.exit_reason = "TP"
                    current.pnl_pct = (current.entry_price - current.tp) / current.entry_price * 100
                    in_trade = False
                elif day_high >= current.sl:
                    current.exit_date = date_str
                    current.exit_price = current.sl
                    current.exit_reason = "SL"
                    current.pnl_pct = (current.entry_price - current.sl) / current.entry_price * 100
                    in_trade = False

            # Timeout: close at close price after N bars
            if in_trade and current is not None:
                entry_idx = df.index.get_loc(
                    pd.Timestamp(current.entry_date) if not isinstance(
                        df.index[0], str) else current.entry_date
                ) if current.entry_date in [str(x)[:10] for x in df.index] else None
                if entry_idx is not None and (i - entry_idx) >= timeout_bars:
                    current.exit_date = date_str
                    current.exit_price = price
                    current.exit_reason = "TIMEOUT"
                    if current.action == "BUY":
                        current.pnl_pct = (price - current.entry_price) / current.entry_price * 100
                    else:
                        current.pnl_pct = (current.entry_price - price) / current.entry_price * 100
                    in_trade = False
            continue  # don't open new trade while managing existing one

        # ── Look for signal ──────────────────────────────────────────────────
        rsi_now = float(rsi.iloc[i])
        macd_bull = (float(macd_line.iloc[i - 1]) < float(signal_line.iloc[i - 1])
                     and float(macd_line.iloc[i]) > float(signal_line.iloc[i]))
        macd_bear = (float(macd_line.iloc[i - 1]) > float(signal_line.iloc[i - 1])
                     and float(macd_line.iloc[i]) < float(signal_line.iloc[i]))
        ema_bull = (float(ema9.iloc[i - 1]) < float(ema21.iloc[i - 1])
                    and float(ema9.iloc[i]) > float(ema21.iloc[i]))
        ema_bear = (float(ema9.iloc[i - 1]) > float(ema21.iloc[i - 1])
                    and float(ema9.iloc[i]) < float(ema21.iloc[i]))

        buy_hits = sum([rsi_now < 40, macd_bull, ema_bull])
        sell_hits = sum([rsi_now > 60, macd_bear, ema_bear])

        sl_dist = atr_val * sl_mult
        tp_dist = atr_val * tp_mult

        action = None
        if buy_hits >= 2:
            action = "BUY"
        elif sell_hits >= 2:
            action = "SELL"

        if action:
            is_buy = action == "BUY"
            tp = round(price + tp_dist if is_buy else price - tp_dist, 4)
            sl = round(price - sl_dist if is_buy else price + sl_dist, 4)
            current = Trade(
                symbol=symbol, action=action,
                entry_date=date_str, entry_price=price,
                tp=tp, sl=sl,
            )
            trades.append(current)
            in_trade = True

    # Close any still-open trade at last bar
    if in_trade and current is not None and current.exit_date is None:
        last_price = float(close.iloc[-1])
        current.exit_date = str(df.index[-1])[:10]
        current.exit_price = last_price
        current.exit_reason = "TIMEOUT"
        if current.action == "BUY":
            current.pnl_pct = (last_price - current.entry_price) / current.entry_price * 100
        else:
            current.pnl_pct = (current.entry_price - last_price) / current.entry_price * 100

    return trades


def _max_drawdown(trades: list[Trade]) -> float:
    """Peak-to-trough drawdown on cumulative P&L curve."""
    if not trades:
        return 0.0
    pnls = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    if not pnls:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 2)


def run_backtest(symbol: str, years: int = 2,
                 sl_mult: float = 1.5, tp_mult: float = 2.5) -> dict:
    """
    Run a full backtest. Returns a summary dict suitable for display.
    """
    df = _fetch(symbol, years=years)
    if df.empty:
        return {"error": f"No data for {symbol}"}

    trades = _simulate(symbol, df, sl_mult=sl_mult, tp_mult=tp_mult)
    completed = [t for t in trades if t.pnl_pct is not None]

    if not completed:
        return {"symbol": symbol, "total_trades": 0, "error": "No trades generated"}

    wins = [t for t in completed if t.pnl_pct and t.pnl_pct > 0]
    losses = [t for t in completed if t.pnl_pct and t.pnl_pct <= 0]
    tp_hits = [t for t in completed if t.exit_reason == "TP"]
    sl_hits = [t for t in completed if t.exit_reason == "SL"]

    win_rate = round(len(wins) / len(completed) * 100, 1)
    avg_pnl = round(sum(t.pnl_pct for t in completed) / len(completed), 2)
    avg_win = round(sum(t.pnl_pct for t in wins) / len(wins), 2) if wins else 0.0
    avg_loss = round(sum(t.pnl_pct for t in losses) / len(losses), 2) if losses else 0.0
    total_pnl = round(sum(t.pnl_pct for t in completed), 2)
    max_dd = _max_drawdown(completed)

    return {
        "symbol": symbol,
        "years": years,
        "total_trades": len(completed),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "total_pnl_pct": total_pnl,
        "max_drawdown_pct": max_dd,
        "tp_hits": len(tp_hits),
        "sl_hits": len(sl_hits),
        "timeouts": len([t for t in completed if t.exit_reason == "TIMEOUT"]),
        "trades": [
            {
                "entry_date": t.entry_date,
                "action": t.action,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl_pct": round(t.pnl_pct, 2) if t.pnl_pct is not None else None,
            }
            for t in completed[-20:]  # last 20 trades for display
        ],
    }


def format_backtest_message(result: dict) -> str:
    if "error" in result and result.get("total_trades", 0) == 0:
        return f"❌ Backtest failed for {result.get('symbol', '?')}: {result['error']}"

    sym = result["symbol"]
    yrs = result.get("years", 2)
    n = result["total_trades"]
    wr = result["win_rate"]
    avg = result["avg_pnl_pct"]
    total = result["total_pnl_pct"]
    dd = result["max_drawdown_pct"]
    tp = result["tp_hits"]
    sl = result["sl_hits"]
    to = result["timeouts"]
    avg_w = result["avg_win_pct"]
    avg_l = result["avg_loss_pct"]

    win_emoji = "🟢" if wr >= 55 else ("🟡" if wr >= 45 else "🔴")
    pnl_emoji = "📈" if total > 0 else "📉"

    lines = [
        f"📊 *Backtest: {sym}* ({yrs}y)",
        "",
        f"Trades: {n}  |  Win rate: {win_emoji} {wr}%",
        f"Avg P&L: {avg:+.2f}%  |  Total: {pnl_emoji} {total:+.2f}%",
        f"Avg win: +{avg_w:.2f}%  |  Avg loss: {avg_l:.2f}%",
        f"Max drawdown: -{dd:.2f}%",
        "",
        f"TP hits: {tp} | SL hits: {sl} | Timeout: {to}",
    ]
    return "\n".join(lines)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a backtest on a symbol")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL or BTC-USD")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--sl", type=float, default=1.5, dest="sl_mult")
    parser.add_argument("--tp", type=float, default=2.5, dest="tp_mult")
    args = parser.parse_args()

    result = run_backtest(args.symbol, years=args.years,
                          sl_mult=args.sl_mult, tp_mult=args.tp_mult)
    print(format_backtest_message(result))
    print()
    if "trades" in result:
        print("Last trades:")
        for t in result["trades"]:
            pnl = f"{t['pnl_pct']:+.2f}%" if t["pnl_pct"] is not None else "?"
            print(f"  {t['entry_date']}  {t['action']:4s}  "
                  f"entry={t['entry']:.2f}  exit={t['exit']:.2f}  "
                  f"({t['exit_reason']:7s})  {pnl}")
