"""
Multi-asset trend following (tactical asset allocation).

  Faber, "A Quantitative Approach to Tactical Asset Allocation" (2007/2013);
  Moskowitz, Ooi & Pedersen, "Time Series Momentum" (2012);
  Hurst, Ooi & Pedersen, "A Century of Evidence on Trend-Following" (2017):
  holding an asset class only while it is above its ~10-month average has,
  across a century and many markets, kept most of the return of buy-and-hold
  with far smaller drawdowns.

Rule (pre-registered 2026-11-07, before any backtest was run here) — Faber's
GTAA-5: five asset classes, 20% each — US stocks (SPY), developed ex-US
stocks (EFA), 7-10y Treasuries (IEF), US REITs (VNQ), commodities (DBC).
At each month end an asset is held only if its close is above its 10-month
simple moving average; otherwise that 20% sits in short Treasuries (SHY).

The verdict uses the same gate as every other strategy: Jensen's alpha vs
the equal-weight buy-and-hold of the same five assets, out of sample, with
plateau and deflated-Sharpe checks.

For EU investors these map to UCITS ETFs (e.g. S&P 500, MSCI World ex-USA,
US Treasury 7-10y, US property, broad commodities, US Treasury 1-3y); the
signal is the asset-class decision, the ETF is the investor's choice.

Research:  python3 trend_alloc.py [--no-download]
Paper:     python3 trend_alloc.py --paper
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

import momentum as eng

STRATEGY = "trend_alloc"
RESULTS_PATH = os.path.join(eng._DIR, "data", "research_trend_alloc.json")
RISKY = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
CASH = "SHY"
UNIVERSES = {"global": RISKY + [CASH]}
BENCH = {"global": RISKY}

DEFAULT_CFG = {"sma": 10, "top": None}
GRID = {"sma": [6, 8, 10, 12], "top": [None, 3]}
# Monthly-portfolio variants already tried (xsmom 54, momentum 96, lowvol 32)
PRIOR_TRIALS = 54 + 96 + 32


def target_weights(mc: pd.DataFrame, daily_ret: pd.DataFrame, index_m: pd.Series,
                   m: pd.Timestamp, cfg: dict) -> tuple[pd.Series, dict]:
    w = pd.Series(0.0, index=mc.columns)
    i = mc.index.get_loc(m)
    info = {"trend_ok": True, "picks": [], "above": {}}
    if i < max(cfg["sma"], 12):
        return w, info
    hist = mc.iloc[: i + 1]
    risky = [s for s in RISKY if s in mc.columns]
    above = {}
    for s in risky:
        px = hist[s].dropna()
        if len(px) >= cfg["sma"]:
            above[s] = bool(px.iloc[-1] > px.iloc[-cfg["sma"]:].mean())
    if len(above) < len(risky):
        return w, info
    info["above"] = above
    held = [s for s in risky if above[s]]
    if cfg.get("top"):
        # Faber's aggressive variant: among assets in an uptrend, the top N by
        # average 3/6/12-month return, 1/N each
        mom = {s: sum(hist[s].iloc[-1] / hist[s].iloc[-1 - L] - 1 for L in (3, 6, 12)) / 3
               for s in held}
        held = sorted(held, key=lambda s: -mom[s])[: cfg["top"]]
        slot = 1.0 / cfg["top"]
    else:
        slot = 1.0 / len(risky)
    for s in held:
        w[s] = slot
    if CASH in w.index and pd.notna(mc.loc[m, CASH]):
        w[CASH] = max(0.0, 1.0 - w.sum())
    info["picks"] = held
    info["trend_ok"] = bool(held)
    return w, info


def load(download: bool) -> pd.DataFrame:
    closes = eng.load_closes(UNIVERSES["global"], download=download)
    full = closes.dropna()
    return closes.loc[full.index[0]:]        # start once every asset trades


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="global", choices=["global", "all"])
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    if args.paper:
        for line in eng.paper_rebalance(["global"], download=not args.no_download,
                                        cfg=DEFAULT_CFG, strategy=STRATEGY,
                                        universes=UNIVERSES, weight_fn=target_weights,
                                        bench_cols=BENCH):
            print(line)
        return
    print("Loading asset-class ETFs…", file=sys.stderr)
    r = eng.study(load(not args.no_download), "global", weight_fn=target_weights,
                  grid=GRID, default=DEFAULT_CFG, prior_trials=PRIOR_TRIALS,
                  bench_cols=RISKY)
    r["strategy"] = STRATEGY
    r["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(eng.report(r).replace("Momentum", "Trend allocation"))
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"global": r}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
