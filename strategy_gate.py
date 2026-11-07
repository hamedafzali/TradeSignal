"""
Strategy gate: decides which strategies may reach users.

Rules were fixed in advance (2026-11-07), before seeing future results, and
are applied mechanically every week after the research runs:

  research -> paper   automatically, when held-out research is DATA VALIDATED
                      (net-of-cost CI above 0, robust neighbourhood, and for
                      monthly portfolio strategies a deflated Sharpe >= 0.95).
  paper    -> live    NEVER automatic. The admin is told a strategy is
                      eligible once research is DATA VALIDATED and it has
                      >= 90 days of paper record with positive net expectancy
                      over a minimum sample. A human flips it with /strategy.
  live     -> paper   automatically, when any of:
                        - research now says NO EDGE
                        - the live record is confidently negative
                          (net 95% CI upper bound < 0 over a minimum sample)
                        - live drawdown > 1.5x the worst out-of-sample
                          drawdown seen in research.

Paper strategies keep generating and logging signals (published=0) so the
record keeps growing — nothing is thrown away, nothing reaches users.
Rules are never edited in the middle of a drawdown.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from edge_stats import clustered_mean_ci, max_drawdown_r

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_DIR, "data")

DD_MULT = 1.5
PAPER_DAYS_FOR_LIVE = 90
# minimum record size: (to call it confidently negative, to call it eligible)
MIN_N = {"intraday": (30, 30), "swing": (30, 30),
         "momentum": (6, 3), "lowvol": (6, 3), "trend_alloc": (6, 3)}
UNIT = {"intraday": "R", "swing": "R",
        "momentum": "%/mo", "lowvol": "%/mo", "trend_alloc": "%/mo"}
# Monthly portfolio strategies -> their research results file (keyed by region)
PORTFOLIO_RESULTS = {"momentum": "research_momentum.json",
                     "lowvol": "research_lowvol.json",
                     "trend_alloc": "research_trend_alloc.json"}


def _load(name: str) -> dict | None:
    try:
        with open(os.path.join(DATA, name)) as f:
            return json.load(f)
    except Exception:
        return None


# ── research evidence ─────────────────────────────────────────────────────────

def research_evidence(name: str) -> dict | None:
    """Latest held-out research for a strategy:
    {verdict, max_dd (same units as its live record, >= 0, or None), detail, ...}
    None if there is no research on file (then no research-based action)."""
    if name == "intraday":
        res = _load("research_results.json") or {}
        tf = "5m-alpaca" if "5m-alpaca" in res else "5m"
        r = res.get(tf)
        if not r:
            return None
        oos = (r.get("live") or {}).get("oos") or {}
        return {"verdict": r.get("verdict"), "max_dd": oos.get("max_dd_r"),
                "detail": f"{tf}: live rule OOS net {oos.get('avg_r')}R (n={oos.get('n')}), "
                          f"grid OOS-positive {r.get('grid_oos_positive')}",
                "generated_at": r.get("generated_at")}
    if name == "swing":
        d = _load("research_daily.json") or {}
        s = (d.get("strategies") or {}).get("swing")
        if not s:
            return None
        lo = s.get("live_oos") or {}
        return {"verdict": s.get("verdict"), "max_dd": lo.get("max_dd_r"),
                "detail": f"live rule OOS excess {lo.get('excess')}%/trade "
                          f"[{lo.get('exc_lo')} … {lo.get('exc_hi')}] (n={lo.get('n')}), "
                          f"grid OOS-positive {s.get('grid_oos_positive')}",
                "generated_at": d.get("generated_at")}
    if name in PORTFOLIO_RESULTS:
        d = _load(PORTFOLIO_RESULTS[name]) or {}
        regs = {k: v["default"] for k, v in d.items() if isinstance(v, dict) and "default" in v}
        if not regs:
            return None
        verdicts = {k: v.get("verdict") for k, v in regs.items()}
        passing = sorted(k for k, v in verdicts.items() if v == "DATA VALIDATED")
        if passing:
            verdict = "DATA VALIDATED"
        elif all(v == "NO EDGE" for v in verdicts.values()):
            verdict = "NO EDGE"
        else:
            verdict = "TRADE-OFF"
        dds = [abs(regs[k].get("oos_perf", {}).get("max_dd") or 0) for k in (passing or regs)]
        detail = "; ".join(
            f"{k}: {v.get('verdict')} alpha {v['oos'].get('alpha_mo')}%/mo "
            f"[{v['oos'].get('alpha_lo')} … {v['oos'].get('alpha_hi')}] DSR {v.get('dsr')}"
            for k, v in sorted(regs.items()) if v.get("oos"))
        return {"verdict": verdict, "regions": passing,
                "max_dd": max(dds) if dds and max(dds) > 0 else None, "detail": detail,
                "generated_at": max(str(d[k].get("generated_at", "")) for k in regs)}
    return None


# ── own record ────────────────────────────────────────────────────────────────

def signal_record(mode: str, since: str | None, published: bool) -> dict:
    """Net-of-cost R record of a signal strategy since its last status change."""
    from database import get_expectancy_stats
    st = get_expectancy_stats(mode=mode, since=since, published=published)
    o = st.get("overall") or {}
    net = o.get("net") or {}
    curve = [c["net"] for c in st.get("curve") or []]
    dd = 0.0
    peak = 0.0
    for c in curve:
        peak = max(peak, c)
        dd = max(dd, peak - c)
    return {"n": o.get("n", 0), "avg": net.get("avg_r"), "ci_hi": net.get("ci_hi"),
            "dd": round(dd, 2)}


def portfolio_paper_returns(rebalances: list[dict]) -> list[dict]:
    """Monthly paper returns from consecutive rebalances: each month holds the
    previous target weights from one rebalance's prices to the next one's,
    net of turnover cost; excess is vs the equal-weight universe."""
    from edge_stats import DEFAULT_COST_PCT, market_of
    out, prev_w = [], {}
    for a, b in zip(rebalances, rebalances[1:]):
        w, pa, pb = a["weights"], a["prices"], b["prices"]
        rets = {s: pb[s] / pa[s] - 1 for s in pa if s in pb and pa[s] and pb[s]}
        if not rets:
            continue
        cost = sum(abs(w.get(s, 0) - prev_w.get(s, 0)) * DEFAULT_COST_PCT[market_of(s)] / 200
                   for s in set(w) | set(prev_w))
        port = sum(wt * rets.get(s, 0.0) for s, wt in w.items()) - cost
        bsyms = (a.get("info") or {}).get("bench_symbols")
        brets = [rets[s] for s in bsyms if s in rets] if bsyms else list(rets.values())
        bench = sum(brets) / len(brets) if brets else 0.0
        out.append({"t": b["asof"], "port": port, "bench": bench, "excess": port - bench})
        prev_w = {s: wt * (1 + rets.get(s, 0.0)) for s, wt in w.items()}
        tot = sum(prev_w.values()) + (1 - sum(w.values()))
        prev_w = {s: v / tot for s, v in prev_w.items()} if tot > 0 else prev_w
    return out


momentum_paper_returns = portfolio_paper_returns   # backwards-compatible name


def portfolio_record(name: str, since: str | None) -> dict:
    """Paper/live record of a monthly portfolio strategy, pooled over regions,
    in % per month (excess vs its equal-weight universe)."""
    from database import get_paper_rebalances
    months = []
    regions = {}
    for rb in get_paper_rebalances(name):
        regions.setdefault(rb["region"], []).append(rb)
    for reg, rbs in regions.items():
        months += portfolio_paper_returns(rbs)
    if since:
        months = [m for m in months if m["t"] >= since[:10]]
    months.sort(key=lambda m: m["t"])
    if not months:
        return {"n": 0, "avg": None, "ci_hi": None, "dd": 0.0}
    exc = [100 * m["excess"] for m in months]
    ci = clustered_mean_ci(exc, [f"{m['t']}" for m in months]) if len(exc) >= 2 else {}
    # drawdown of the strategy itself (%), compounding ignored — small months
    return {"n": len(exc), "avg": round(sum(exc) / len(exc), 3),
            "ci_hi": None if ci.get("hi") is None else round(ci["hi"], 3),
            "dd": max_drawdown_r([100 * m["port"] for m in months])}


# ── the decision ──────────────────────────────────────────────────────────────

def decide(name: str, status: str, research: dict | None, record: dict,
           days_in_status: float) -> dict:
    """Pure decision. Returns {"new_status": str|None, "reason": str,
    "eligible": bool}. new_status None = no change."""
    unit = UNIT.get(name, "R")
    n_neg, n_pos = MIN_N.get(name, (30, 30))
    verdict = (research or {}).get("verdict")
    n, avg, ci_hi, dd = record.get("n", 0), record.get("avg"), record.get("ci_hi"), record.get("dd", 0.0)
    res_dd = (research or {}).get("max_dd")

    if status == "live":
        if verdict == "NO EDGE":
            return {"new_status": "paper", "eligible": False,
                    "reason": f"auto-demoted: held-out research now NO EDGE ({research.get('detail')})"}
        if n >= n_neg and ci_hi is not None and ci_hi < 0:
            return {"new_status": "paper", "eligible": False,
                    "reason": f"auto-demoted: live record confidently negative — "
                              f"{avg:+.3f}{unit}, 95% CI upper {ci_hi:+.3f} (n={n})"}
        if res_dd and dd > DD_MULT * res_dd:
            return {"new_status": "paper", "eligible": False,
                    "reason": f"auto-demoted: live drawdown {dd:.2f}{unit} > "
                              f"{DD_MULT}x worst out-of-sample drawdown {res_dd:.2f}{unit}"}
        return {"new_status": None, "eligible": False, "reason": "live, within limits"}

    if status == "research":
        if verdict == "DATA VALIDATED":
            return {"new_status": "paper", "eligible": False,
                    "reason": f"auto-promoted to paper: held-out research DATA VALIDATED "
                              f"({research.get('detail')})"}
        return {"new_status": None, "eligible": False,
                "reason": f"research verdict {verdict or 'n/a'}"}

    if status == "paper":
        if (verdict == "DATA VALIDATED" and days_in_status >= PAPER_DAYS_FOR_LIVE
                and n >= n_pos and avg is not None and avg > 0
                and not (res_dd and dd > DD_MULT * res_dd)):
            return {"new_status": None, "eligible": True,
                    "reason": f"ELIGIBLE for live: research DATA VALIDATED, "
                              f"{days_in_status:.0f} days paper, record {avg:+.3f}{unit} (n={n})"}
        return {"new_status": None, "eligible": False,
                "reason": f"paper: research {verdict or 'n/a'}, record n={n}"
                          + (f" avg {avg:+.3f}{unit}" if avg is not None else "")
                          + f", {days_in_status:.0f} days"}

    return {"new_status": None, "eligible": False, "reason": status}


def _days_since(iso: str | None) -> float:
    if not iso:
        return 0.0
    try:
        return (datetime.utcnow() - datetime.fromisoformat(iso[:19])).total_seconds() / 86400
    except ValueError:
        return 0.0


def evaluate_all(apply: bool = True) -> list[dict]:
    """Run the gate over every registered strategy. Applies status changes
    (actor 'gate') and stores the evidence. Returns one row per strategy:
    {name, status, new_status, eligible, reason, research, record}."""
    from database import list_strategies, set_strategy_evidence, set_strategy_status
    out = []
    for s in list_strategies():
        name, status = s["name"], s["status"]
        if status == "retired":
            continue
        research = research_evidence(name)
        since = s.get("updated_at")
        if name in PORTFOLIO_RESULTS:
            record = portfolio_record(name, since)
        else:
            record = signal_record(name, since, published=(status == "live"))
        d = decide(name, status, research, record, _days_since(since))
        if apply:
            set_strategy_evidence(name, {"research": research, "record": record,
                                         "checked_at": datetime.utcnow().isoformat(timespec="seconds"),
                                         "gate": d["reason"]})
            if d["new_status"]:
                set_strategy_status(name, d["new_status"], d["reason"], actor="gate")
        out.append({"name": name, "status": status, **d, "research": research, "record": record})
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate the strategy gate")
    ap.add_argument("--apply", action="store_true", help="apply status changes")
    args = ap.parse_args()
    for r in evaluate_all(apply=args.apply):
        arrow = f" -> {r['new_status']}" if r["new_status"] else ""
        print(f"{r['name']:9s} {r['status']}{arrow} · {r['reason']}")
