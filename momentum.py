"""
Cross-sectional momentum, implemented the way the evidence says it works.

  Jegadeesh & Titman (1993); Asness, Frazzini, Israel & Moskowitz, "Fact,
  Fiction and Momentum Investing" (2014): rank by past 6-12 month return,
  skip the most recent month, hold the winners, rebalance monthly.
  Barroso & Santa-Clara (2015): scale exposure by realized volatility
  — roughly doubles Sharpe and removes most of the crash risk.
  Hurst, Ooi & Pedersen (2017) / Faber (2007): a market trend filter
  (index below its 10-month average -> cash) cuts bear-market drawdowns.

Long-only, no leverage (exposure capped at 100%; the rest is cash), one
region at a time (no US-vs-EU currency/region bet hidden in the ranking).

`target_weights` is shared by the research harness below and by the bot's
paper tracker, so what is tested is exactly what is tracked.

Research:  python3 momentum.py [--region us|eu|all] [--no-download]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from statistics import NormalDist

import numpy as np
import pandas as pd

from edge_stats import DEFAULT_COST_PCT, deflated_sharpe, market_of

_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(_DIR, "data", "research_momentum.json")

# Broad large-cap universes: index constituents, NOT a hand-picked watchlist
# (a watchlist of names that already went up hands momentum its winners).
# Still today's constituents, so some survivorship bias remains.
UNIVERSES = {
    "us": ("AAPL,ABBV,ABT,ACN,ADBE,AIG,AMD,AMGN,AMT,AMZN,AVGO,AXP,BA,BAC,BK,BKNG,BLK,"
           "BMY,BRK-B,C,CAT,CHTR,CL,CMCSA,COF,COP,COST,CRM,CSCO,CVS,CVX,DE,DHR,DIS,DUK,"
           "EMR,F,FDX,GD,GE,GILD,GM,GOOG,GS,HD,HON,IBM,INTC,INTU,ISRG,JNJ,JPM,KO,LIN,LLY,"
           "LMT,LOW,MA,MCD,MDLZ,MDT,MET,META,MMM,MO,MRK,MS,MSFT,NEE,NFLX,NKE,NVDA,ORCL,"
           "PEP,PFE,PG,PM,QCOM,RTX,SBUX,SCHW,SO,SPG,T,TGT,TMO,TXN,UNH,UNP,UPS,USB,V,VZ,"
           "WFC,WMT,XOM").split(","),
    "eu": ("ADS.DE,AD.AS,AI.PA,AIR.PA,ALV.DE,ABI.BR,ASML.AS,CS.PA,BAS.DE,BAYN.DE,BBVA.MC,"
           "SAN.MC,BMW.DE,BNP.PA,DB1.DE,DHL.DE,DTE.DE,ENEL.MI,ENI.MI,EL.PA,RMS.PA,IBE.MC,"
           "ITX.MC,IFX.DE,INGA.AS,ISP.MI,KER.PA,OR.PA,MC.PA,MBG.DE,MUV2.DE,NOKIA.HE,ORA.PA,"
           "SAF.PA,SAN.PA,SAP.DE,SU.PA,SIE.DE,TTE.PA,UCG.MI,DG.PA,VOW3.DE,BN.PA,BEI.DE,"
           "CBK.DE,CON.DE,DBK.DE,EOAN.DE,FRE.DE,HEI.DE,HEN3.DE,HNR1.DE,MRK.DE,MTX.DE,"
           "RHM.DE,RWE.DE,SRT3.DE,SY1.DE,ZAL.DE,NESN.SW,NOVN.SW,ROG.SW,UBSG.SW,ZURN.SW,"
           "ABBN.SW,NOVO-B.CO,PHIA.AS,HEIA.AS,CAP.PA,RI.PA,VIV.PA,STMPA.PA").split(","),
}

# Textbook defaults, fixed BEFORE looking at results.
DEFAULT_CFG = {"look": 12, "skip": 1, "k": 15, "vol_target": 0.15, "trend": True}
GRID = {"look": [6, 12], "skip": [0, 1], "k": [10, 15, 20],
        "vol_target": [None, 0.15], "trend": [False, True]}
VOL_WINDOW = 126       # ~6 months of daily returns (Barroso & Santa-Clara)
TREND_MONTHS = 10      # Faber 10-month moving average
# Variants of cross-sectional momentum already tried in research_daily.py on
# other universes: counted for the deflated Sharpe ratio.
PRIOR_TRIALS = 18 * 3


# ── strategy ──────────────────────────────────────────────────────────────────

def month_ends(daily_close: pd.DataFrame) -> pd.DataFrame:
    return daily_close.resample("ME").last()


def market_index(daily_ret: pd.DataFrame) -> pd.Series:
    """Equal-weight index of the universe (daily rebalanced) — the benchmark
    level the trend filter looks at."""
    return (1 + daily_ret.mean(axis=1, skipna=True).fillna(0)).cumprod()


def target_weights(mc: pd.DataFrame, daily_ret: pd.DataFrame, index_m: pd.Series,
                   m: pd.Timestamp, cfg: dict) -> tuple[pd.Series, dict]:
    """
    Target weights at month-end `m` using only data up to and including m.
    Returns (weights summing to <= 1, info). Cash = 1 - sum(weights).
    """
    look, skip, k = cfg["look"], cfg["skip"], cfg["k"]
    w = pd.Series(0.0, index=mc.columns)
    i = mc.index.get_loc(m)
    info = {"trend_ok": True, "scale": 1.0, "vol": None, "picks": []}
    if i < look + skip:
        return w, info
    score = mc.iloc[i - skip] / mc.iloc[i - skip - look] - 1
    score = score[mc.iloc[i].notna() & score.notna()]
    if len(score) < 2 * k:
        return w, info

    if cfg.get("trend"):
        lvl = index_m.loc[:m]
        if len(lvl) >= TREND_MONTHS and lvl.iloc[-1] < lvl.iloc[-TREND_MONTHS:].mean():
            info["trend_ok"] = False
            return w, info

    picks = score[score > 0].nlargest(k).index      # absolute filter: winners only
    if len(picks) == 0:
        return w, info
    w[picks] = 1.0 / k                              # unfilled slots stay cash
    info["picks"] = list(picks)

    if cfg.get("vol_target"):
        hist = daily_ret.loc[:m, picks].iloc[-VOL_WINDOW:]
        port = hist.fillna(0).mean(axis=1) * (len(picks) / k)
        vol = float(port.std() * np.sqrt(252)) if len(port) > 20 else None
        if vol and vol > 0:
            scale = min(1.0, cfg["vol_target"] / vol)  # no leverage
            w *= scale
            info["scale"], info["vol"] = round(scale, 3), round(vol, 3)
    return w, info


def backtest(daily_close: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Monthly portfolio vs equal-weight universe, net of turnover costs."""
    daily_ret = daily_close.pct_change(fill_method=None)
    mc = month_ends(daily_close)
    index_m = month_ends(market_index(daily_ret).to_frame("i"))["i"]
    fwd = mc.pct_change(fill_method=None).shift(-1)
    cost_side = pd.Series({s: DEFAULT_COST_PCT[market_of(s)] / 200 for s in mc.columns})
    prev = pd.Series(0.0, index=mc.columns)
    rows = []
    for m in mc.index[:-1]:
        fr = fwd.loc[m]
        w, info = target_weights(mc, daily_ret, index_m, m, cfg)
        avail = fr.notna() & mc.loc[m].notna()
        if avail.sum() < 2 * cfg["k"]:
            prev[:] = 0.0
            continue
        w = w.where(avail, 0.0)
        cost = float(((w - prev).abs() * cost_side).sum())
        port = float((w * fr.fillna(0)).sum()) - cost
        rows.append({"t": m + pd.offsets.MonthEnd(1), "port": port,
                     "bench": float(fr[avail].mean()), "exposure": float(w.sum()),
                     "turnover": float((w - prev).abs().sum())})
        grown = w * (1 + fr.fillna(0))
        tot = float(grown.sum() + (1 - w.sum()))   # cash earns 0
        prev = grown / tot if tot > 0 else grown
    return pd.DataFrame(rows).set_index("t") if rows else pd.DataFrame()


# ── statistics ────────────────────────────────────────────────────────────────

def alpha_beta(port: pd.Series, bench: pd.Series) -> dict:
    """OLS port = a + b*bench (monthly). Alpha is the return not explained by
    simply holding the universe at the strategy's own average exposure."""
    n = len(port)
    if n < 24:
        return {"n": n}
    x, y = bench.to_numpy(), port.to_numpy()
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    a = ym - b * xm
    resid = y - a - b * x
    s2 = float((resid ** 2).sum() / (n - 2))
    se_a = np.sqrt(s2 * (1 / n + xm ** 2 / sxx))
    return {"n": n, "alpha_mo": round(100 * a, 3), "alpha_lo": round(100 * (a - 1.96 * se_a), 3),
            "alpha_hi": round(100 * (a + 1.96 * se_a), 3), "beta": round(b, 2),
            "alpha_t": round(a / se_a, 2) if se_a > 0 else None}


def perf(r: pd.Series) -> dict:
    if len(r) < 12:
        return {"months": len(r)}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    sd = r.std()
    return {"months": len(r), "cagr": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
            "vol": round(100 * sd * np.sqrt(12), 1),
            "sharpe": round(float(r.mean() / sd * np.sqrt(12)), 2) if sd > 0 else None,
            "max_dd": round(100 * float((eq / eq.cummax() - 1).min()), 1),
            "worst_month": round(100 * float(r.min()), 1)}


def _key(c: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in c.items())


def _neighbours(cfg: dict) -> list[dict]:
    out = []
    for k, vals in GRID.items():
        i = vals.index(cfg[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**cfg, k: vals[j]})
    return out


def verdict(oos_ab: dict, plateau: float, dsr: float | None) -> str:
    if oos_ab.get("n", 0) < 36:
        return "INSUFFICIENT DATA"
    if oos_ab["alpha_mo"] <= 0:
        return "NO EDGE"
    if oos_ab["alpha_lo"] > 0 and plateau >= 0.6 and (dsr or 0) >= 0.95:
        return "DATA VALIDATED"
    return "TRADE-OFF"


# ── research ──────────────────────────────────────────────────────────────────

def load_closes(symbols: list[str], download: bool) -> pd.DataFrame:
    from research_daily import drop_forming_bar, load_daily
    cols = {}
    for s in symbols:
        df = load_daily(s, download=download)
        if df is not None and len(df) > 300:
            cols[s] = drop_forming_bar(df)["Close"]
        else:
            print(f"  skip {s}", file=sys.stderr)
    return pd.DataFrame(cols).sort_index()


def study(daily_close: pd.DataFrame, region: str) -> dict:
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    runs = {_key(c): backtest(daily_close, c) for c in cfgs}
    first = min(r.index[0] for r in runs.values() if not r.empty)
    last = max(r.index[-1] for r in runs.values() if not r.empty)
    cut = first + (last - first) * 2 / 3

    res = {}
    for k, r in runs.items():
        a, b = r[r.index < cut], r[r.index >= cut]
        res[k] = {"is": alpha_beta(a["port"], a["bench"]),
                  "oos": alpha_beta(b["port"], b["bench"])}

    def pick(cfg):
        r = runs[_key(cfg)]
        b = r[r.index >= cut]
        return {"cfg": cfg, "is": res[_key(cfg)]["is"], "oos": res[_key(cfg)]["oos"],
                "oos_perf": perf(b["port"]), "oos_bench": perf(b["bench"]),
                "oos_exposure": round(float(b["exposure"].mean()), 2),
                "turnover_yr": round(float(r["turnover"].mean() * 12), 2)}

    best_cfg = max(cfgs, key=lambda c: res[_key(c)]["is"].get("alpha_mo", -9))
    out = {}
    for label, cfg in (("default", DEFAULT_CFG), ("best_is", best_cfg)):
        p = pick(cfg)
        nb = _neighbours(cfg)
        plateau = sum(1 for c in nb if res[_key(c)]["oos"].get("alpha_mo", -1) > 0) / len(nb)
        # Deflated Sharpe on the OOS months, penalised for every variant tried
        b = runs[_key(cfg)]
        b = b[b.index >= cut]["port"]
        srs = [float(r[r.index >= cut]["port"].mean() / r[r.index >= cut]["port"].std())
               for r in runs.values() if len(r[r.index >= cut]) > 12]
        dsr = deflated_sharpe(b.to_numpy(), srs, n_trials=len(cfgs) + PRIOR_TRIALS)
        p.update({"plateau": round(plateau, 2), "dsr": None if dsr is None else round(dsr, 3)})
        p["verdict"] = verdict(p["oos"], plateau, dsr)
        out[label] = p
    pos = sum(1 for v in res.values() if v["oos"].get("alpha_mo", -1) > 0)
    return {"region": region, "symbols": daily_close.shape[1],
            "period": [str(first.date()), str(last.date())], "oos_from": str(cut.date()),
            "grid_oos_positive": f"{pos}/{len(cfgs)}", **out}


def _fmt(p: dict) -> str:
    o, pf, bm = p["oos"], p["oos_perf"], p["oos_bench"]
    return "\n".join([
        f"    alpha OOS {o['alpha_mo']:+.2f}%/mo [{o['alpha_lo']:+.2f} … {o['alpha_hi']:+.2f}] "
        f"t={o['alpha_t']} beta={o['beta']} · IS alpha {p['is'].get('alpha_mo', 0):+.2f}%/mo",
        f"    OOS strategy CAGR {pf['cagr']}% vol {pf['vol']}% Sharpe {pf['sharpe']} "
        f"maxDD {pf['max_dd']}% worst month {pf['worst_month']}%",
        f"    OOS universe CAGR {bm['cagr']}% vol {bm['vol']}% Sharpe {bm['sharpe']} "
        f"maxDD {bm['max_dd']}% worst month {bm['worst_month']}%",
        f"    avg exposure {p['oos_exposure']} · turnover {p['turnover_yr']}x/yr · "
        f"plateau {p['plateau']} · deflated Sharpe {p['dsr']}"])


def report(r: dict) -> str:
    return "\n".join([
        f"=== Momentum [{r['region']}] {r['symbols']} symbols, {r['period'][0]} → "
        f"{r['period'][1]}, OOS from {r['oos_from']} · grid OOS-alpha>0 {r['grid_oos_positive']}",
        f"  textbook default {_key(r['default']['cfg'])}: {r['default']['verdict']}",
        _fmt(r["default"]),
        f"  best on IS {_key(r['best_is']['cfg'])}: {r['best_is']['verdict']}",
        _fmt(r["best_is"])])


# ── paper tracking ────────────────────────────────────────────────────────────

def last_completed_month(today: datetime | None = None) -> str:
    """'YYYY-MM' of the last calendar month that has fully closed."""
    t = today or datetime.utcnow()
    y, m = (t.year, t.month - 1) if t.month > 1 else (t.year - 1, 12)
    return f"{y:04d}-{m:02d}"


def paper_rebalance(regions: list[str], download: bool = True,
                    cfg: dict | None = None) -> list[str]:
    """Record the target portfolio at the last completed month end for each
    region (idempotent). The paper record is the return of each recorded
    portfolio up to the next one — see strategy_gate.momentum_paper_returns."""
    from database import get_paper_rebalances, save_paper_rebalance
    cfg = cfg or DEFAULT_CFG
    want = last_completed_month()
    done = []
    for reg in regions:
        if any(r["asof"][:7] == want for r in get_paper_rebalances("momentum", reg)):
            continue
        closes = load_closes(UNIVERSES[reg], download=download)
        mc = month_ends(closes)
        ms = [m for m in mc.index if m.strftime("%Y-%m") == want]
        if not ms:
            print(f"  {reg}: no data for {want}", file=sys.stderr)
            continue
        m = ms[0]
        daily_ret = closes.pct_change(fill_method=None)
        index_m = month_ends(market_index(daily_ret).to_frame("i"))["i"]
        w, info = target_weights(mc, daily_ret, index_m, m, cfg)
        w = w[w > 0]
        prices = {s: round(float(p), 6) for s, p in mc.loc[m].dropna().items()}
        info = {**info, "cfg": cfg, "exposure": round(float(w.sum()), 3)}
        if save_paper_rebalance("momentum", reg, m.strftime("%Y-%m-%d"),
                                {s: round(float(v), 6) for s, v in w.items()}, prices, info):
            done.append(f"{reg} {m.date()}: {len(w)} holdings, exposure {w.sum():.0%}"
                        + ("" if info["trend_ok"] else " (trend filter: cash)"))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="all", choices=["us", "eu", "all"])
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--paper", action="store_true",
                    help="record this month's paper portfolio instead of researching")
    args = ap.parse_args()
    regions = ["us", "eu"] if args.region == "all" else [args.region]
    if args.paper:
        for line in paper_rebalance(regions, download=not args.no_download):
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
        closes = load_closes(UNIVERSES[reg], download=not args.no_download)
        r = study(closes, reg)
        r["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        results[reg] = r
        print(report(r))
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()
