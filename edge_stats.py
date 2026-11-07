"""
Shared statistics for judging whether a strategy has an edge.

Used by both the live scoreboard (database.get_expectancy_stats) and the
offline research harness (research.py) so both report the same numbers.

Everything is in R-multiples: profit divided by the initial risk (entry→stop
distance). Costs are converted into R per trade: cost_R = cost_pct / sl_pct.
With a 0.33% stop, a 0.10% round-trip cost is already 0.3R per trade, so
ignoring costs makes tight-stop intraday strategies look much better than
they are.
"""
from __future__ import annotations

import math
from collections import defaultdict

# Round-trip cost (commission + spread + slippage) as % of price, per market.
# Conservative retail defaults; overridable via app_settings cost_pct_<market>.
DEFAULT_COST_PCT = {"us": 0.05, "eu": 0.10, "crypto": 0.20}

_EU_SUFFIXES = (".DE", ".F", ".XETRA", ".AS", ".PA", ".MI", ".BR", ".LS", ".MC",
                ".SW", ".CO", ".ST", ".OL", ".HE", ".VI", ".L")


def market_of(symbol: str) -> str:
    s = (symbol or "").upper()
    if s.endswith("-USD"):
        return "crypto"
    if s.endswith(_EU_SUFFIXES):
        return "eu"
    return "us"


def clustered_mean_ci(values: list[float], clusters: list[str],
                      z: float = 1.96) -> dict:
    """
    Mean with a cluster-robust standard error (clusters = trading days).

    Signals on the same day are correlated (same market move hits many
    symbols), so a naive SE overstates confidence. The clustered SE treats
    each day as one independent observation of the summed residuals.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "se": None, "lo": None, "hi": None, "days": 0}
    mean = sum(values) / n
    by_day: dict[str, float] = defaultdict(float)
    for v, c in zip(values, clusters):
        by_day[c] += v - mean
    g = len(by_day)
    if n < 2 or g < 2:
        return {"n": n, "mean": mean, "se": None, "lo": None, "hi": None, "days": g}
    ss = sum(s * s for s in by_day.values())
    se = math.sqrt(ss * g / (g - 1)) / n
    return {"n": n, "mean": mean, "se": se,
            "lo": mean - z * se, "hi": mean + z * se, "days": g}


def summarize_r(rs: list[float], days: list[str], wins: int | None = None,
                avg_rr: float | None = None) -> dict:
    """Expectancy summary for a list of R results with their trade days."""
    ci = clustered_mean_ci(rs, days)
    n = ci["n"]
    out = {
        "n": n,
        "days": ci["days"],
        "avg_r": round(ci["mean"], 3),
        "ci_lo": None if ci["lo"] is None else round(ci["lo"], 3),
        "ci_hi": None if ci["hi"] is None else round(ci["hi"], 3),
        "total_r": round(sum(rs), 2),
    }
    if wins is not None and n:
        out["win_rate"] = round(100 * wins / n, 1)
    if avg_rr:
        out["breakeven_wr"] = round(100 / (1 + avg_rr), 1)
    # Trades needed for the CI to exclude zero at the current mean/SE
    # (SE scales ~1/sqrt(n)). Only meaningful for a positive mean.
    if ci["se"] and ci["mean"] > 0:
        need = n * (1.96 * ci["se"] / ci["mean"]) ** 2
        if need <= 20 * n:  # beyond that the estimate is meaningless
            out["n_needed"] = int(math.ceil(need))
    out["verdict"] = verdict(out)
    return out


def verdict(s: dict, min_n: int = 30) -> str:
    """EDGE / UNPROVEN / NO EDGE for a single sample (no OOS split)."""
    if s["n"] < min_n or s["ci_lo"] is None:
        return "INSUFFICIENT DATA"
    if s["avg_r"] <= 0:
        return "NO EDGE"
    if s["ci_lo"] > 0:
        return "EDGE"
    return "UNPROVEN"


def deflated_sharpe(returns, trial_srs: list[float], n_trials: int) -> float | None:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): probability that
    the true Sharpe of `returns` is above what the best of `n_trials` random
    strategies would show by luck. >= 0.95 is the usual bar.

    returns    per-period returns of the selected strategy
    trial_srs  per-period Sharpe ratios of the variants tried (for their spread)
    n_trials   how many variants were tried in total (incl. earlier studies)
    """
    from statistics import NormalDist
    r = [float(x) for x in returns if x == x]
    t = len(r)
    if t < 12 or len(trial_srs) < 2 or n_trials < 2:
        return None
    mean = sum(r) / t
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (t - 1))
    if sd == 0:
        return None
    sr = mean / sd
    skew = sum(((x - mean) / sd) ** 3 for x in r) / t
    kurt = sum(((x - mean) / sd) ** 4 for x in r) / t
    m = sum(trial_srs) / len(trial_srs)
    v = sum((s - m) ** 2 for s in trial_srs) / (len(trial_srs) - 1)
    nd, g = NormalDist(), 0.5772156649
    sr0 = math.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / n_trials)
                          + g * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return None
    return nd.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom))


def max_drawdown_r(rs: list[float]) -> float:
    """Worst peak-to-trough fall of a cumulative R (or %) curve, >= 0."""
    peak = cum = dd = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return round(dd, 2)
