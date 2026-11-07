"""
Sector and country ETF momentum: a survivorship-free test of momentum.

  Moskowitz & Grinblatt, "Do Industries Explain Momentum?" (1999);
  Asness, Moskowitz & Pedersen, "Value and Momentum Everywhere" (2013):
  industries and country equity indices that outperformed over the past year
  keep outperforming for months, in every market and asset class studied.

Why this exists: momentum.py tests momentum on today's S&P 100 / EU large caps,
which back-fills survivors into the past (survivorship bias), so it can never
validate. ETFs that exist for the whole test are free of that bias.

Universe (pre-registered 2026-11-07, rule-based, before any backtest here):
every Select Sector SPDR from the 1998 launch (9) plus every iShares
single-country ETF that existed before 2001 (19) — 28 funds.

Rule = momentum.py's textbook rule with k sized to the universe: each month
end hold the top quarter (k=7) by 12-1 month return, winners only, 1/k each,
scaled to 15% target volatility, in cash when the equal-weight universe is
below its 10-month average. Benchmark: equal weight of all 28.

Research:  python3 etf_mom.py [--no-download]
Paper:     python3 etf_mom.py --paper
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import momentum as eng

STRATEGY = "etf_mom"
RESULTS_PATH = os.path.join(eng._DIR, "data", "research_etf_mom.json")
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
COUNTRIES = ["EWA", "EWC", "EWD", "EWG", "EWH", "EWI", "EWJ", "EWK", "EWL", "EWN",
             "EWO", "EWP", "EWQ", "EWS", "EWU", "EWW", "EWY", "EWT", "EWZ"]
UNIVERSES = {"global": SECTORS + COUNTRIES}

DEFAULT_CFG = {"look": 12, "skip": 1, "k": 7, "vol_target": 0.15, "trend": True}
GRID = {"look": [6, 12], "skip": [0, 1], "k": [5, 7, 10],
        "vol_target": [None, 0.15], "trend": [False, True]}
# Monthly-portfolio variants already tried (xsmom 54, momentum 96, lowvol 32,
# trend_alloc 8)
PRIOR_TRIALS = 54 + 96 + 32 + 8


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
                                        universes=UNIVERSES):
            print(line)
        return
    print("Loading sector and country ETFs…", file=sys.stderr)
    closes = eng.load_closes(UNIVERSES["global"], download=not args.no_download)
    r = eng.study(closes, "global", grid=GRID, default=DEFAULT_CFG,
                  prior_trials=PRIOR_TRIALS)
    r["strategy"] = STRATEGY
    r["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    print(eng.report(r).replace("Momentum", "ETF momentum"))
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"global": r}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
