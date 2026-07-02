"""
Swing-strategy backtest gate.

Replays the exact swing rule (signals.swing_signal_from_df) over ~2 years of
daily bars per symbol, simulating TP/SL/timeout outcomes bar by bar. The gate
decides per symbol whether the swing scan may fire live signals for it:

    PASS  =  n_signals >= MIN_SIGNALS  and  expectancy >= MIN_EXPECTANCY_R

Run inside the bot container:
    python3 backtest_swing.py            # report only
    python3 backtest_swing.py --apply    # report + write swing_enabled flags
"""
import sys

import numpy as np

from signals import (SWING_HORIZON_DAYS, SWING_SL_MULT, SWING_TP_MULT,
                     fetch_ohlc, swing_signal_from_df)

# The rule is selective (~3 signals/symbol/year), so per-symbol samples are
# tiny. The statistically meaningful gate is the AGGREGATE across all symbols;
# per-symbol we only exclude names with clear negative evidence.
MIN_AGG_TRADES = 100      # aggregate sample floor
MIN_AGG_EXPECTANCY = 0.10  # aggregate avg R per trade
EXCLUDE_N = 5              # symbol excluded when n >= this and expectancy < 0


def backtest_symbol(symbol: str, tp_mult: float = SWING_TP_MULT,
                    sl_mult: float = SWING_SL_MULT,
                    df=None) -> dict | None:
    if df is None:
        df = fetch_ohlc(symbol, period="2y", interval="1d")
    if df is None or len(df) < 260:
        return None

    high = df["High"].squeeze().astype(float).to_numpy()
    low = df["Low"].squeeze().astype(float).to_numpy()
    close = df["Close"].squeeze().astype(float).to_numpy()

    n = len(df)
    trades = []
    i = 210
    while i < n - 1:
        sig = swing_signal_from_df(symbol, df.iloc[: i + 1],
                                   tp_mult=tp_mult, sl_mult=sl_mult)
        if sig is None:
            i += 1
            continue
        entry, tp, sl = sig["price"], sig["tp"], sig["sl"]
        sl_dist = entry - sl
        if sl_dist <= 0:
            i += 1
            continue

        outcome, r_mult, exit_bar = None, 0.0, None
        last_j = min(i + SWING_HORIZON_DAYS, n - 1)
        for j in range(i + 1, last_j + 1):
            hit_tp = high[j] >= tp
            hit_sl = low[j] <= sl
            if hit_tp and hit_sl:
                outcome, r_mult, exit_bar = "neutral", 0.0, j  # ambiguous bar
                break
            if hit_tp:
                outcome, r_mult, exit_bar = "win", (tp - entry) / sl_dist, j
                break
            if hit_sl:
                outcome, r_mult, exit_bar = "loss", -1.0, j
                break
        if outcome is None:
            if last_j < i + SWING_HORIZON_DAYS:
                break  # signal too close to the end of data — drop it
            outcome, r_mult, exit_bar = "timeout", (close[last_j] - entry) / sl_dist, last_j

        trades.append({"outcome": outcome, "r": r_mult,
                       "hold_days": exit_bar - i})
        i = exit_bar + 1  # one open position per symbol — resume after exit

    if not trades:
        return {"symbol": symbol, "n": 0, "passed": False}

    wins = sum(1 for t in trades if t["outcome"] == "win")
    losses = sum(1 for t in trades if t["outcome"] == "loss")
    expectancy = float(np.mean([t["r"] for t in trades]))
    decisive = wins + losses
    return {
        "symbol": symbol,
        "n": len(trades),
        "wins": wins,
        "losses": losses,
        "neutral": sum(1 for t in trades if t["outcome"] == "neutral"),
        "timeout": sum(1 for t in trades if t["outcome"] == "timeout"),
        "win_rate": round(100 * wins / decisive, 1) if decisive else None,
        "expectancy_r": round(expectancy, 3),
        "avg_hold_days": round(float(np.mean([t["hold_days"] for t in trades])), 1),
    }


def main() -> None:
    apply = "--apply" in sys.argv
    grid = "--grid" in sys.argv
    from database import get_active_symbols, set_swing_enabled

    symbols = get_active_symbols()

    if grid:
        # Robustness check across TP geometry — the strategy should not live
        # or die on one parameter value. Data downloaded once per symbol.
        frames = {s: fetch_ohlc(s, period="2y", interval="1d") for s in symbols}
        for tp in (2.0, 2.5):
            rs = [backtest_symbol(s, tp_mult=tp, df=frames[s]) for s in symbols]
            rs = [r for r in rs if r and r["n"] > 0]
            n_all = sum(r["n"] for r in rs)
            e = sum(r["expectancy_r"] * r["n"] for r in rs) / n_all if n_all else 0
            w = sum(r["wins"] for r in rs); l = sum(r["losses"] for r in rs)
            wr = 100 * w / (w + l) if (w + l) else 0
            print(f"TP {tp}xATR: {n_all} trades · WR {wr:.1f}% · E {e:+.3f}R")
        return

    print(f"Backtesting swing rule on {len(symbols)} symbols "
          f"(2y daily, TP {SWING_TP_MULT}xATR / SL {SWING_SL_MULT}xATR, "
          f"{SWING_HORIZON_DAYS}d horizon)\n")

    results = []
    for sym in symbols:
        try:
            r = backtest_symbol(sym)
        except Exception as exc:
            print(f"{sym:10s} ERROR {exc}")
            continue
        if r is None:
            print(f"{sym:10s} no data")
            continue
        results.append(r)
        if r["n"] == 0:
            print(f"{sym:10s} 0 signals")
            continue
        print(f"{sym:10s} n={r['n']:>3} WR={str(r['win_rate']):>5}% "
              f"E={r['expectancy_r']:+.3f}R hold={r['avg_hold_days']:>4}d")

    traded = [r for r in results if r["n"] > 0]
    all_n = sum(r["n"] for r in traded)
    agg_e = (sum(r["expectancy_r"] * r["n"] for r in traded) / all_n) if all_n else 0.0
    agg_wins = sum(r["wins"] for r in traded)
    agg_losses = sum(r["losses"] for r in traded)
    agg_wr = 100 * agg_wins / (agg_wins + agg_losses) if (agg_wins + agg_losses) else 0
    aggregate_ok = all_n >= MIN_AGG_TRADES and agg_e >= MIN_AGG_EXPECTANCY

    print(f"\nAGGREGATE: {all_n} trades across {len(traded)} symbols · "
          f"WR {agg_wr:.1f}% · expectancy {agg_e:+.3f}R")
    print(f"GATE ({'PASS' if aggregate_ok else 'FAIL'}): need n>={MIN_AGG_TRADES} "
          f"and E>={MIN_AGG_EXPECTANCY:+.2f}R in aggregate")

    # Per-symbol: enabled when the aggregate passes, excluding symbols with
    # clear negative evidence of their own
    enabled = []
    for r in results:
        bad = r["n"] >= EXCLUDE_N and r["expectancy_r"] < 0
        r["passed"] = aggregate_ok and not bad
        if r["passed"]:
            enabled.append(r["symbol"])
        elif aggregate_ok and bad:
            print(f"  excluded {r['symbol']}: n={r['n']}, E={r['expectancy_r']:+.3f}R")
    print(f"ENABLED SYMBOLS: {len(enabled)}/{len(results)}")

    if apply:
        for r in results:
            set_swing_enabled(r["symbol"], r["passed"])
        print(f"APPLIED: swing_enabled written for {len(enabled)} symbols")
    else:
        print("(dry run — pass --apply to write swing_enabled flags)")


if __name__ == "__main__":
    main()
