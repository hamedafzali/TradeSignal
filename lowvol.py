"""
Low-volatility (defensive) equities.

  Haugen & Heins (1975); Blitz & van Vliet, "The Volatility Effect" (2007);
  Baker, Bradley & Wurgler (2011); Frazzini & Pedersen, "Betting Against
  Beta" (2014): the least volatile stocks have earned about the same return
  as the market with markedly less risk, i.e. positive risk-adjusted alpha,
  in the US, Europe and globally, over many decades.

Rule (pre-registered 2026-11-07, before any backtest was run here):
  each month-end rank the universe by trailing 252-day volatility of daily
  returns, hold the lowest-volatility quintile (k=20 of ~95 US / ~70 EU large
  caps) equal-weighted, rebalance monthly. Long-only, fully invested.

The verdict is on Jensen's alpha vs the equal-weight universe (the same
gate as momentum): a defensive portfolio with beta < 1 and the same return
shows positive alpha; one that only lowers returns does not.

Research:  python3 lowvol.py [--region us|eu|all] [--no-download]
Paper:     python3 lowvol.py --paper --region us
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import momentum as eng

STRATEGY = "lowvol"
RESULTS_PATH = os.path.join(eng._DIR, "data", "research_lowvol.json")
UNIVERSES = eng.UNIVERSES

DEFAULT_CFG = {"win": 252, "k": 20, "trend": False}
GRID = {"win": [126, 252], "k": [10, 15, 20, 30], "trend": [False, True]}
# Monthly-portfolio variants already tried on these universes (xsmom 54 +
# momentum 2 x 48): counted for the deflated Sharpe ratio.
PRIOR_TRIALS = 54 + 96


def target_weights(mc: pd.DataFrame, daily_ret: pd.DataFrame, index_m: pd.Series,
                   m: pd.Timestamp, cfg: dict) -> tuple[pd.Series, dict]:
    """Equal weight in the k least volatile names, using data up to m only."""
    w = pd.Series(0.0, index=mc.columns)
    info = {"trend_ok": True, "picks": []}
    hist = daily_ret.loc[:m].iloc[-cfg["win"]:]
    if len(hist) < cfg["win"]:
        return w, info
    ok = hist.notna().sum() >= int(0.9 * cfg["win"])
    ok &= mc.loc[m].notna()
    vol = hist.loc[:, ok].std()
    if len(vol) < 2 * cfg["k"]:
        return w, info
    if cfg.get("trend"):
        lvl = index_m.loc[:m]
        if len(lvl) >= eng.TREND_MONTHS and lvl.iloc[-1] < lvl.iloc[-eng.TREND_MONTHS:].mean():
            info["trend_ok"] = False
            return w, info
    picks = vol.nsmallest(cfg["k"]).index
    w[picks] = 1.0 / cfg["k"]
    info["picks"] = list(picks)
    info["avg_vol"] = round(float(vol[picks].mean() * np.sqrt(252)), 3)
    return w, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="all", choices=["us", "eu", "all"])
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    regions = ["us", "eu"] if args.region == "all" else [args.region]
    if args.paper:
        for line in eng.paper_rebalance(regions, download=not args.no_download,
                                        cfg=DEFAULT_CFG, strategy=STRATEGY,
                                        universes=UNIVERSES, weight_fn=target_weights):
            print(line)
        return
    results = {}
    if os.path.exists(RESULTS_PATH):
        try:
            results = json.load(open(RESULTS_PATH))
        except Exception:
            results = {}
    for reg in regions:
        print(f"Loading {reg} universe…", file=sys.stderr)
        closes = eng.load_closes(UNIVERSES[reg], download=not args.no_download)
        r = eng.study(closes, reg, weight_fn=target_weights, grid=GRID,
                      default=DEFAULT_CFG, prior_trials=PRIOR_TRIALS,
                      survivorship=True)   # today's index members
        r["strategy"] = STRATEGY
        r["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        results[reg] = r
        print(eng.report(r).replace("Momentum", "Low-volatility"))
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()
