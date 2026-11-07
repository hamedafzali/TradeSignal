"""
Daily-bar strategy research — which long-only daily rules actually beat
simply holding the stock, net of costs, on data they were not tuned on?

Research only: nothing here changes what the bot sends.

Strategies (long-only, matching the product — Trade Republic users can't short):
  swing     the bot's live swing rule (pullback in stacked uptrend), ATR bracket
  rsi2      Connors-style short-term reversal: RSI(2) oversold above SMA200,
            exit on close > SMA(n) or after max hold
  breakout  N-day closing-high breakout above SMA200 (Donchian / 52-week-high
            family), exit on M-day closing low or 2xATR initial stop
  xsmom     cross-sectional momentum (Jegadeesh-Titman): each month hold the
            top-K symbols by L-month return skipping the last S months,
            absolute filter (only positive momentum; else cash)

Execution: signal on the close of day t, entry at the OPEN of t+1 (no
look-ahead). Stop and target both inside one daily bar -> assume the stop
(conservative). Gaps through a level fill at the open. One position per symbol.

The benchmark matters more than the raw return. A long-only rule in a bull
market "makes money" just by being long. Each trade is therefore scored as
EXCESS return = net return - (symbol's average daily return in that period
x days held). A rule earns its place only if the excess is positive out of
sample.

Method (same discipline as research.py):
  - global time split: first 2/3 of the timeline in-sample (IS), last 1/3 OOS
  - best config chosen on IS excess only; reported on OOS
  - plateau: share of one-step neighbours that are also OOS-positive
  - CI clustered by entry week (same-week trades are correlated)
  DATA VALIDATED  OOS excess > 0, CI lower bound > 0, OOS net > 0, plateau >= 0.6
  TRADE-OFF       OOS excess > 0 but not significant or no plateau
  NO EDGE         OOS excess <= 0

Known biases: the universe is today's watchlist (survivors) — absolute returns
are flattered; the excess-vs-drift measure is computed on the same survivors,
which removes most of that. xsmom ranks local-currency returns across US/EU.

Run:  python3 research_daily.py [--symbols ...] [--no-download]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from edge_stats import DEFAULT_COST_PCT, clustered_mean_ci, market_of, max_drawdown_r

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_DIR, "data", "research_cache")
RESULTS_PATH = os.path.join(_DIR, "data", "research_daily.json")
SINCE = "2005-01-01"
MIN_IS_TRADES = 100

STRATEGIES = {
    "swing": {
        "live": {"sl": 1.5, "tp": 2.5, "hold": 10},
        "grid": {"sl": [1.0, 1.5, 2.0], "tp": [1.5, 2.5, 3.5], "hold": [5, 10, 20]},
    },
    "rsi2": {
        "live": None,
        "grid": {"thresh": [5, 10, 15, 20], "exit_sma": [5, 10], "hold": [5, 10, 20]},
    },
    "breakout": {
        "live": None,
        "grid": {"entry": [20, 55, 100, 252], "exit": [10, 20, 50]},
        "valid": lambda c: c["exit"] < c["entry"],
    },
    "xsmom": {
        "live": None,
        "grid": {"look": [3, 6, 12], "skip": [0, 1], "k": [5, 10, 15]},
    },
}


# ── data ──────────────────────────────────────────────────────────────────────

def load_daily(symbol: str, download: bool = True) -> pd.DataFrame | None:
    """Split/dividend-adjusted daily bars since 2005, cached (refreshed whole:
    adjustment factors change retroactively with every dividend)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol.replace('/', '_')}_1d.pkl")
    df = None
    if download:
        from signals import fetch_ohlc
        df = fetch_ohlc(symbol, start=SINCE, interval="1d")
        if df is not None and not df.empty:
            df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            df.to_pickle(path)
    if (df is None or df.empty) and os.path.exists(path):
        df = pd.read_pickle(path)
    if df is None or df.empty:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    ok = (df["Close"] > 0) & (df["Open"] > 0) & df["High"].notna() & df["Low"].notna()
    return df[ok]


def drop_forming_bar(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) and df.index[-1].date() >= datetime.utcnow().date():
        return df.iloc[:-1]
    return df


# ── indicators / signals ──────────────────────────────────────────────────────

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    from signals import _atr, _macd, _rsi
    c = df["Close"]
    x = pd.DataFrame(index=df.index)
    x["open"], x["high"], x["low"], x["close"] = df["Open"], df["High"], df["Low"], c
    x["atr"] = _atr(df)
    x["sma200"] = c.rolling(200).mean()
    x["sma50"] = c.rolling(50).mean()
    x["ema20"] = c.ewm(span=20, adjust=False).mean()
    x["rsi"] = _rsi(c)
    x["rsi2"] = _rsi(c, 2)
    m, s = _macd(c)
    x["hist"] = m - s
    x["ret"] = c.pct_change()
    return x


def swing_signals(x: pd.DataFrame) -> np.ndarray:
    """Vectorised copy of signals.swing_signal_from_df (same thresholds)."""
    c, low = x["close"], x["low"]
    uptrend = (c > x["sma50"]) & (x["sma50"] > x["sma200"]) & (x["ema20"] > x["ema20"].shift(5))
    dipped = ((low <= x["ema20"]).astype(int).rolling(5).max() > 0) | \
             ((x["rsi"] < 45).astype(int).rolling(5).max() > 0)
    recovery = (x["rsi"] > x["rsi"].shift(1)) & (c > c.shift(1)) & (x["hist"] > x["hist"].shift(1))
    sig = uptrend & dipped & recovery & (x["rsi"] < 65) & (x["atr"] > 0)
    return sig.fillna(False).to_numpy()


# ── trade simulation (entry at next open) ─────────────────────────────────────

def _trade(sym, x, i_sig, i_exit, px_exit, how):
    e = i_sig + 1
    entry = x["open"].iat[e]
    return {"symbol": sym, "t": x.index[e], "exit_t": x.index[i_exit],
            "ret": 100 * (px_exit / entry - 1), "days": i_exit - e + 1, "how": how}


def sim_bracket(sym, x, sig, sl_mult, tp_mult, hold):
    """ATR bracket from the entry open; timeout at the close of day `hold`."""
    o, h, l, c, atr = (x[k].to_numpy() for k in ("open", "high", "low", "close", "atr"))
    n, out, i = len(x), [], 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        last = e + hold - 1
        if last >= n:
            break                      # can't resolve yet
        entry = o[e]
        sl, tp = entry - sl_mult * atr[i], entry + tp_mult * atr[i]
        res = None
        for j in range(e, last + 1):
            if j > e and o[j] <= sl:
                res = (j, o[j], "gap_sl"); break
            if j > e and o[j] >= tp:
                res = (j, o[j], "gap_tp"); break
            if l[j] <= sl:
                res = (j, sl, "sl"); break   # incl. both-hit: assume stop first
            if h[j] >= tp:
                res = (j, tp, "tp"); break
        if res is None:
            res = (last, c[last], "timeout")
        tr = _trade(sym, x, i, *res)
        tr["sl_pct"] = 100 * sl_mult * atr[i] / entry
        out.append(tr)
        i = res[0]                     # one position per symbol: next signal after exit
    return out


def sim_rsi2(sym, x, thresh, exit_sma, hold):
    c = x["close"].to_numpy()
    sma = x["close"].rolling(exit_sma).mean().to_numpy()
    sig = ((x["close"] > x["sma200"]) & (x["rsi2"] < thresh)).to_numpy()
    n, out, i = len(x), [], 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        last = e + hold - 1
        if last >= n:
            break
        j = next((j for j in range(e, last + 1) if c[j] > sma[j]), last)
        out.append(_trade(sym, x, i, j, c[j], "sma" if j < last or c[j] > sma[j] else "timeout"))
        i = j
    return out


def sim_breakout(sym, x, entry_n, exit_n, stop_mult=2.0, max_hold=250):
    c, l, o, atr = (x[k].to_numpy() for k in ("close", "low", "open", "atr"))
    hi = x["close"].rolling(entry_n).max().shift(1).to_numpy()
    lo = x["close"].rolling(exit_n).min().shift(1).to_numpy()
    sig = (x["close"].to_numpy() > hi) & (x["close"] > x["sma200"]).to_numpy() & (atr > 0)
    n, out, i = len(x), [], 0
    while i < n - 1:
        if not sig[i]:
            i += 1
            continue
        e = i + 1
        stop = o[e] - stop_mult * atr[i]
        res = None
        for j in range(e, min(e + max_hold, n)):
            if j > e and o[j] <= stop:
                res = (j, o[j], "gap_sl"); break
            if l[j] <= stop:
                res = (j, stop, "sl"); break
            if c[j] < lo[j]:
                res = (j, c[j], "channel"); break
        if res is None:
            if e + max_hold > n:
                break                  # still open at the end of data
            res = (e + max_hold - 1, c[e + max_hold - 1], "timeout")
        out.append(_trade(sym, x, i, *res))
        i = res[0]
    return out


# ── cross-sectional momentum (monthly portfolio) ──────────────────────────────

def monthly_closes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: x["close"].resample("ME").last() for s, x in data.items()})


def sim_xsmom(mc: pd.DataFrame, look: int, skip: int, k: int) -> pd.DataFrame:
    """Monthly returns of the top-k momentum portfolio vs the equal-weight
    universe. Signal at month-end close m, held through month m+1."""
    fwd = mc.pct_change(fill_method=None).shift(-1)
    score = mc.shift(skip) / mc.shift(skip + look) - 1
    cost_side = pd.Series({s: DEFAULT_COST_PCT[market_of(s)] / 200 for s in mc.columns})
    rows, prev_w = [], pd.Series(0.0, index=mc.columns)
    for m in mc.index[:-1]:
        sc, fr = score.loc[m], fwd.loc[m]
        avail = sc.notna() & fr.notna()
        if avail.sum() < 2 * k:
            prev_w[:] = 0.0
            continue
        top = sc[avail & (sc > 0)].nlargest(k).index
        w = pd.Series(0.0, index=mc.columns)
        w[top] = 1.0 / k                 # unfilled slots stay in cash
        turnover_cost = float(((w - prev_w).abs() * cost_side).sum())
        port = float((w[top] * fr[top]).sum()) - turnover_cost
        bench = float(fr[avail].mean())
        rows.append({"t": m + pd.offsets.MonthEnd(1), "port": port, "bench": bench,
                     "excess": port - bench, "n_held": len(top)})
        # weights drift with returns before next rebalance
        grown = w * (1 + fr.fillna(0))
        prev_w = grown / grown.sum() * w.sum() if grown.sum() > 0 else grown
    return pd.DataFrame(rows).set_index("t") if rows else pd.DataFrame()


def xsmom_robustness(mc: pd.DataFrame, cfg: dict, start, cut) -> dict:
    """Is the momentum edge a property of the method, or of a few hindsight
    winners in today's watchlist? Re-run OOS on sub-universes."""
    def oos(cols, k=None):
        c = {**cfg, "k": k or cfg["k"]}
        m = sim_xsmom(mc[cols], **c)
        m = m[m.index >= cut] if not m.empty else m
        return score_months(m)

    cols = list(mc.columns)
    total = (mc.ffill().iloc[-1] / mc.bfill().iloc[0] - 1).sort_values(ascending=False)
    out = {
        "us_only": oos([c for c in cols if market_of(c) == "us"], k=max(2, cfg["k"] // 2)),
        "eu_only": oos([c for c in cols if market_of(c) == "eu"]),
        "drop_top5_winners": oos([c for c in cols if c not in set(total.index[:5])]),
        "drop_top10_winners": oos([c for c in cols if c not in set(total.index[:10])]),
        "top10_winners": list(total.index[:10]),
    }
    loo = {s: oos([c for c in cols if c != s]).get("excess") for s in cols}
    worst = min(loo, key=lambda s: loo[s] if loo[s] is not None else 1e9)
    out["leave_one_out_min"] = {"dropped": worst, "excess": loo[worst]}
    return out


def portfolio_stats(r: pd.Series) -> dict:
    if len(r) < 12:
        return {"months": len(r)}
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    return {"months": len(r), "cagr": round(100 * (eq.iloc[-1] ** (1 / yrs) - 1), 1),
            "vol": round(100 * r.std() * np.sqrt(12), 1),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(12)), 2) if r.std() > 0 else None,
            "max_dd": round(100 * float((eq / eq.cummax() - 1).min()), 1)}


# ── scoring ───────────────────────────────────────────────────────────────────

def score_trades(trades: list[dict], drift: dict, cost_scale: float = 1.0,
                 cost_override: float | None = None) -> dict:
    """Net and excess-vs-holding return per trade (in %), CI clustered by week."""
    if not trades:
        return {"n": 0}
    net, exc, wk, wins, days, r = [], [], [], 0, [], []
    for t in trades:
        cost = cost_override if cost_override is not None else \
            DEFAULT_COST_PCT[market_of(t["symbol"])] * cost_scale
        v = t["ret"] - cost
        net.append(v)
        exc.append(v - drift.get(t["symbol"], 0.0) * t["days"])
        wk.append(t["t"].strftime("%G-%V"))
        wins += v > 0
        days.append(t["days"])
        if "sl_pct" in t and t["sl_pct"] > 0:
            r.append(v / t["sl_pct"])
    ci_e = clustered_mean_ci(exc, wk)
    ci_n = clustered_mean_ci(net, wk)
    out = {"n": len(net), "weeks": ci_e["days"],
           "net": round(ci_n["mean"], 3), "net_lo": _r(ci_n["lo"]),
           "excess": round(ci_e["mean"], 3), "exc_lo": _r(ci_e["lo"]), "exc_hi": _r(ci_e["hi"]),
           "win_rate": round(100 * wins / len(net), 1),
           "avg_days": round(float(np.mean(days)), 1)}
    if r:
        out["avg_r"] = round(float(np.mean(r)), 3)
        rt = sorted((t["t"], v / t["sl_pct"]) for t, v in zip(trades, net)
                    if t.get("sl_pct", 0) > 0)
        out["max_dd_r"] = max_drawdown_r([x for _, x in rt])
    return out


def score_months(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    ci = clustered_mean_ci(list(df["excess"] * 100), [str(i) for i in range(len(df))])
    return {"n": len(df), "excess": round(ci["mean"], 3), "exc_lo": _r(ci["lo"]),
            "exc_hi": _r(ci["hi"]), "net": round(100 * float(df["port"].mean()), 3),
            "port": portfolio_stats(df["port"]), "bench": portfolio_stats(df["bench"]),
            "avg_held": round(float(df["n_held"].mean()), 1)}


def _r(v):
    return None if v is None else round(v, 3)


def verdict(oos: dict, plateau: float | None) -> str:
    if oos.get("n", 0) < 30 or oos.get("exc_lo") is None:
        return "INSUFFICIENT DATA"
    if oos["excess"] <= 0:
        return "NO EDGE"
    if oos["exc_lo"] > 0 and oos["net"] > 0 and (plateau or 0) >= 0.6:
        return "DATA VALIDATED"
    return "TRADE-OFF"


def _key(c: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in c.items())


def _configs(spec: dict) -> list[dict]:
    g = spec["grid"]
    cfgs = [dict(zip(g, v)) for v in itertools.product(*g.values())]
    return [c for c in cfgs if spec.get("valid", lambda _: True)(c)]


def _neighbours(cfg: dict, grid: dict) -> list[dict]:
    out = []
    for k, vals in grid.items():
        i = vals.index(cfg[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**cfg, k: vals[j]})
    return out


# ── study ─────────────────────────────────────────────────────────────────────

def run(symbols: list[str], download: bool = True,
        only: list[str] | None = None) -> dict:
    data = {}
    for s in symbols:
        df = load_daily(s, download=download)
        if download:
            time.sleep(0.3)
        if df is None or len(df) < 400:
            print(f"  skip {s}: no/short data", file=sys.stderr)
            continue
        data[s] = indicators(drop_forming_bar(df))
    if not data:
        raise SystemExit("no data")

    start = min(x.index[200] for x in data.values())   # after SMA200 warm-up
    end = max(x.index[-1] for x in data.values())
    cut = start + (end - start) * 2 / 3

    def drift(lo, hi):   # per-symbol mean daily return (%) inside [lo, hi)
        return {s: 100 * float(x["ret"][(x.index >= lo) & (x.index < hi)].mean())
                for s, x in data.items()}
    drift_is, drift_oos = drift(start, cut), drift(cut, end + pd.Timedelta(days=1))
    swing_sig = {s: swing_signals(x) for s, x in data.items()}

    def trades_for(name, cfg):
        out = []
        for s, x in data.items():
            if name == "swing":
                out += sim_bracket(s, x, swing_sig[s], cfg["sl"], cfg["tp"], cfg["hold"])
            elif name == "rsi2":
                out += sim_rsi2(s, x, cfg["thresh"], cfg["exit_sma"], cfg["hold"])
            elif name == "breakout":
                out += sim_breakout(s, x, cfg["entry"], cfg["exit"])
        return [t for t in out if t["t"] >= start]

    def split(trades):
        # a trade belongs to IS only if it also EXITED before the cut
        return ([t for t in trades if t["exit_t"] < cut],
                [t for t in trades if t["t"] >= cut])

    results = {}
    mc = monthly_closes(data)
    for name, spec in STRATEGIES.items():
        if only and name not in only:
            continue
        print(f"  {name}…", file=sys.stderr)
        grid = {}
        for cfg in _configs(spec):
            if name == "xsmom":
                m = sim_xsmom(mc, **cfg)
                m = m[m.index >= start] if not m.empty else m
                is_, oos = (score_months(m[m.index < cut]), score_months(m[m.index >= cut])) \
                    if not m.empty else ({"n": 0}, {"n": 0})
                grid[_key(cfg)] = {"cfg": cfg, "is": is_, "oos": oos, "_m": m}
            else:
                tr = trades_for(name, cfg)
                a, b = split(tr)
                grid[_key(cfg)] = {"cfg": cfg, "is": score_trades(a, drift_is),
                                   "oos": score_trades(b, drift_oos), "_t": tr}
        min_n = 36 if name == "xsmom" else MIN_IS_TRADES
        eligible = [g for g in grid.values() if g["is"].get("n", 0) >= min_n]
        if not eligible:
            results[name] = {"verdict": "INSUFFICIENT DATA"}
            continue
        best = max(eligible, key=lambda g: g["is"]["excess"])
        nb = [grid[_key(c)] for c in _neighbours(best["cfg"], spec["grid"]) if _key(c) in grid]
        plateau = (sum(1 for g in nb if g["oos"].get("excess", -1) > 0) / len(nb)) if nb else None
        pos = sum(1 for g in grid.values() if g["oos"].get("excess", -1) > 0)
        r = {"best_cfg": best["cfg"], "best_is": best["is"], "best_oos": best["oos"],
             "plateau_share": None if plateau is None else round(plateau, 2),
             "grid_oos_positive": f"{pos}/{len(grid)}",
             "verdict": verdict(best["oos"], plateau)}

        if name == "xsmom":
            m = best["_m"]
            mo = m[m.index >= cut]
            r["oos_by_year"] = {str(y): round(100 * float(g["excess"].sum()), 2)
                                for y, g in mo.groupby(mo.index.year)}
            r["oos_median_excess"] = round(100 * float(mo["excess"].median()), 3)
            r["oos_hit_rate"] = round(100 * float((mo["excess"] > 0).mean()), 1)
            r["robustness"] = xsmom_robustness(mc, best["cfg"], start, cut)
        else:
            _, b = split(best["_t"])
            r["oos_cost"] = {f"{c:.2f}%": score_trades(b, drift_oos, cost_override=c)["excess"]
                             for c in (0.0, 0.25, 0.50)}
            r["oos_by_market"] = {mk: score_trades([t for t in b if market_of(t["symbol"]) == mk],
                                                   drift_oos)
                                  for mk in ("us", "eu")}
            yrs = {}
            for t in b:
                yrs.setdefault(str(t["t"].year), []).append(t)
            r["oos_by_year"] = {y: score_trades(v, drift_oos)["excess"] for y, v in sorted(yrs.items())}
            if spec["live"]:
                lg = grid[_key(spec["live"])]
                r["live_cfg"], r["live_is"], r["live_oos"] = spec["live"], lg["is"], lg["oos"]
                la, lb = split(lg["_t"])
                r["live_oos_by_market"] = {mk: score_trades(
                    [t for t in lb if market_of(t["symbol"]) == mk], drift_oos) for mk in ("us", "eu")}
        results[name] = r

    return {"generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "symbols": len(data), "period": [str(start.date()), str(end.date())],
            "oos_from": str(cut.date()), "strategies": results}


# ── report ────────────────────────────────────────────────────────────────────

def _fmt(s: dict) -> str:
    if not s or not s.get("n"):
        return "n=0"
    ci = f"[{s['exc_lo']:+.2f} … {s['exc_hi']:+.2f}]" if s.get("exc_lo") is not None else "[n/a]"
    unit = "mo" if "port" in s else "tr"
    extra = f" · {s['win_rate']}% win · {s['avg_days']}d" if "win_rate" in s else ""
    rr = f" · {s['avg_r']:+.2f}R" if "avg_r" in s else ""
    return (f"excess {s['excess']:+.2f}%/{unit} {ci} · net {s['net']:+.2f}%{rr}"
            f" · n={s['n']}{extra}")


def report(res: dict) -> str:
    L = [f"=== Daily-bar research: {res['symbols']} symbols, {res['period'][0]} → "
         f"{res['period'][1]}, OOS from {res['oos_from']} ===",
         "excess = net return minus what simply holding the stock earned over the same days"]
    for name, r in res["strategies"].items():
        L.append(f"\n── {name}: {r['verdict']}")
        if "best_cfg" not in r:
            continue
        if r.get("live_cfg"):
            L.append(f"  live config {_key(r['live_cfg'])}")
            L.append(f"    IS : {_fmt(r['live_is'])}")
            L.append(f"    OOS: {_fmt(r['live_oos'])}")
            for mk, s in r["live_oos_by_market"].items():
                L.append(f"      {mk}: {_fmt(s)}")
        L.append(f"  best on IS {_key(r['best_cfg'])}")
        L.append(f"    IS : {_fmt(r['best_is'])}")
        L.append(f"    OOS: {_fmt(r['best_oos'])}")
        L.append(f"    plateau {r['plateau_share']} · grid OOS-positive {r['grid_oos_positive']}")
        if "oos_by_market" in r:
            for mk, s in r["oos_by_market"].items():
                L.append(f"      {mk}: {_fmt(s)}")
            L.append("    OOS excess at round-trip cost " +
                     " · ".join(f"{k}: {v:+.2f}%" for k, v in r["oos_cost"].items()))
        if r["best_oos"].get("port"):
            p, b = r["best_oos"]["port"], r["best_oos"]["bench"]
            L.append(f"    OOS portfolio CAGR {p.get('cagr')}% Sharpe {p.get('sharpe')} "
                     f"maxDD {p.get('max_dd')}%  vs equal-weight CAGR {b.get('cagr')}% "
                     f"Sharpe {b.get('sharpe')} maxDD {b.get('max_dd')}%")
        L.append("    OOS excess by year " +
                 " ".join(f"{y}:{v:+.2f}" for y, v in r["oos_by_year"].items()))
        if "robustness" in r:
            rb = r["robustness"]
            L.append(f"    OOS monthly excess median {r['oos_median_excess']:+.2f}% · "
                     f"months beating benchmark {r['oos_hit_rate']}%")
            L.append("    robustness (OOS, same config):")
            for k in ("us_only", "eu_only", "drop_top5_winners", "drop_top10_winners"):
                L.append(f"      {k:<19} {_fmt(rb[k])}")
            lo = rb["leave_one_out_min"]
            L.append(f"      worst leave-one-out: without {lo['dropped']} → excess {lo['excess']:+.2f}%/mo")
            L.append(f"      biggest winners (full period): {', '.join(rb['top10_winners'])}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", help="comma-separated; default = active symbols")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--strategies", help="comma-separated subset of " + ",".join(STRATEGIES))
    ap.add_argument("--out", default=RESULTS_PATH, help="results JSON path")
    args = ap.parse_args()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from database import get_active_symbols
        symbols = [s for s in get_active_symbols() if market_of(s) != "crypto"]
    print(f"Loading {len(symbols)} symbols (1d since {SINCE})…", file=sys.stderr)
    only = args.strategies.split(",") if args.strategies else None
    res = run(symbols, download=not args.no_download, only=only)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(report(res))


if __name__ == "__main__":
    main()
