"""
Research harness — does the live rule have an edge, net of costs, out of sample?

Replays the live intraday rule (signals.signal_from_df + combine_signals
filters) on historical bars, simulates outcomes the same way the bot resolves
them (TP/SL walk, same-bar ambiguity -> exit at close, 24h wall-clock timeout),
charges round-trip costs, and then answers three questions honestly:

  1. Live config: what is the net expectancy (R/trade) with a 95% CI?
  2. Held-out test: pick the best parameter set on the first ~2/3 of the
     timeline ONLY, then measure it on the last ~1/3 it never saw.
  3. Plateau: do neighbouring parameter sets also work out of sample, or is
     the "best" one a lucky spike?

Verdict (on the held-out period):
  DATA VALIDATED  OOS net mean > 0, CI lower bound > 0, and a plateau
  TRADE-OFF       OOS net mean > 0 but not significant or no plateau
  NO EDGE         OOS net mean <= 0
  If it says NO EDGE, believe it.

Data: yfinance serves only 60 days of 5m bars, so every run merges what it
downloads into data/research_cache/ — the 5m history grows with each run.
The --tf 1h mode tests the same rule on 1h bars (730 days, 4h context) for a
longer sample (a different timeframe, so a robustness check, not the live
system).

Not modelled (each only removes trades, never adds edge): regime filter,
earnings guard, max-3-same-direction-per-scan, ML gating, adaptive TP/SL.
Live scans also see the in-progress bar; this replay uses completed bars.

Run inside the bot container:
    python3 research.py                    # all active symbols, 5m
    python3 research.py --tf 1h            # 2-year 1h robustness check
    python3 research.py --symbols SAP.DE,AAPL --no-download
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from edge_stats import DEFAULT_COST_PCT, market_of, summarize_r

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_DIR, "data", "research_cache")
RESULTS_PATH = os.path.join(_DIR, "data", "research_results.json")

TIMEOUT_HOURS = 24
DEDUP_HOURS = 6

LIVE = {"rsi": 40, "sl": 1.5, "tp": 2.5, "filters": "trend+mtf", "delay": 0}
GRID = {
    "rsi": [30, 35, 40, 45],            # BUY below x, SELL above 100-x
    "sl": [1.0, 1.5, 2.0, 2.5],         # stop  = sl x ATR(14)
    "tp": [1.5, 2.5, 3.5],              # target = tp x ATR(14)
    "filters": ["trend+mtf", "trend", "mtf", "none"],
}
TF_SPEC = {
    # base interval, yfinance period, context interval, context source
    "5m": {"interval": "5m", "period": "60d", "bar": pd.Timedelta(minutes=5),
           "ctx": "1h", "ctx_bar": pd.Timedelta(hours=1)},
    "1h": {"interval": "1h", "period": "730d", "bar": pd.Timedelta(hours=1),
           "ctx": "4h", "ctx_bar": pd.Timedelta(hours=4)},
}


# ── data ──────────────────────────────────────────────────────────────────────

def _download(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    from signals import fetch_ohlc
    df = fetch_ohlc(symbol, period=period, interval=interval)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.tz_convert("UTC")


def load_bars(symbol: str, interval: str, period: str,
              download: bool = True) -> pd.DataFrame | None:
    """Cached bars, merged with a fresh download so history accumulates."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol.replace('/', '_')}_{interval}.pkl")
    cached = pd.read_pickle(path) if os.path.exists(path) else None
    fresh = _download(symbol, interval, period) if download else None
    if fresh is not None and len(fresh) and cached is not None and len(cached):
        # The newest cached bar may have been in progress — fresh data wins.
        df = pd.concat([cached[cached.index < fresh.index[0]], fresh])
    else:
        df = fresh if fresh is not None else cached
    if df is None or df.empty:
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["Close"] > 0) & df["High"].notna() & df["Low"].notna()]
    if fresh is not None:
        df.to_pickle(path)
    return df



# ── Alpaca (free IEX feed): years of US 5m history ───────────────────────────
# Free account keys via env ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY.
# Free plan: IEX exchange only (prices track the consolidated tape closely;
# volume is IEX's share, so the relative volume gate is a proxy), 200 req/min,
# the most recent 15 minutes are not queryable.
ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/{sym}/bars"
ALPACA_SINCE = "2016-01-01"


def alpaca_keys() -> tuple[str, str] | None:
    k, s = os.getenv("ALPACA_API_KEY_ID", ""), os.getenv("ALPACA_API_SECRET_KEY", "")
    return (k, s) if k and s else None


def _alpaca_get(url: str, keys: tuple[str, str], retries: int = 5) -> dict:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": keys[0], "APCA-API-SECRET-KEY": keys[1],
        "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Alpaca request failed after {retries} attempts")


def parse_alpaca_bars(bars: list[dict]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 09:30–16:00 ET weekday bars (yfinance intraday is regular-hours only)."""
    if df.empty:
        return df
    et = df.index.tz_convert("America/New_York")
    mins = et.hour * 60 + et.minute
    keep = (et.weekday < 5) & (mins >= 9 * 60 + 30) & (mins < 16 * 60)
    return df[keep]


def fetch_alpaca(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                 keys: tuple[str, str], timeframe: str = "5Min") -> pd.DataFrame:
    import urllib.parse
    sym = symbol.replace("-", ".")
    out, token = [], None
    while True:
        q = {"timeframe": timeframe, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 10000,
             "feed": "iex", "adjustment": "all", "sort": "asc"}
        if token:
            q["page_token"] = token
        data = _alpaca_get(ALPACA_BARS_URL.format(sym=urllib.parse.quote(sym)) + "?"
                           + urllib.parse.urlencode(q), keys)
        out.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        time.sleep(0.35)                  # stay under 200 req/min
        if not token:
            break
    return regular_session(parse_alpaca_bars(out))


def load_alpaca_5m(symbol: str, download: bool = True,
                   since: str = ALPACA_SINCE) -> pd.DataFrame | None:
    """Incrementally cached Alpaca 5m bars (regular session, split-adjusted).

    adjustment=all re-adjusts history after splits/dividends, so a cache built
    before a corporate action is stale: refetch fully once a month."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{symbol.replace('/', '_')}_5m_alpaca.pkl")
    cached = pd.read_pickle(path) if os.path.exists(path) else None
    keys = alpaca_keys()
    if not download or keys is None:
        return cached
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)
    stale = (cached is None or cached.empty or
             (time.time() - os.path.getmtime(path)) > 30 * 86400)
    start = pd.Timestamp(since, tz="UTC") if stale else cached.index[-1] - pd.Timedelta(days=1)
    fresh = fetch_alpaca(symbol, start, end, keys)
    if stale or cached is None:
        df = fresh
    else:
        df = pd.concat([cached[cached.index < fresh.index[0]], fresh]) if len(fresh) else cached
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df):
        df.to_pickle(path)
    return df


def hourly_from_5m(df: pd.DataFrame) -> pd.DataFrame:
    """1h bars aligned like yfinance US hourly bars (09:30, 10:30, …)."""
    et = df.tz_convert("America/New_York")
    h = et.resample("1h", offset="30min").agg({"Open": "first", "High": "max", "Low": "min",
                                                "Close": "last", "Volume": "sum"}).dropna()
    return h.tz_convert("UTC")

# ── indicators (identical formulas to signals.py) ─────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.where(d > 0, 0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(com=period - 1, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def _macd_bull_bear(close: pd.Series):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    sig = line.ewm(span=9, adjust=False).mean()
    return line > sig, line < sig


def _context(ctx: pd.DataFrame, ctx_bar: pd.Timedelta) -> pd.DataFrame:
    """EMA200 trend and 2-of-3 bias per context bar, keyed by completion time."""
    close = ctx["Close"]
    ema200 = close.ewm(span=200, adjust=False).mean()
    n = np.arange(1, len(close) + 1)
    trend = np.where(n < 200, "unknown",
                     np.where(close > ema200 * 1.005, "up",
                              np.where(close < ema200 * 0.995, "down", "sideways")))
    rsi = _rsi(close)
    mb, ms = _macd_bull_bear(close)
    e9 = close.ewm(span=9, adjust=False).mean()
    e21 = close.ewm(span=21, adjust=False).mean()
    buy = (rsi < 40).astype(int) + mb.astype(int) + (e9 > e21).astype(int)
    sell = (rsi > 60).astype(int) + ms.astype(int) + (e9 < e21).astype(int)
    bias = np.where(buy >= 2, "BUY", np.where(sell >= 2, "SELL", ""))
    bias = np.where(n < 50, "", bias)
    # A context bar is usable once it has completed: start + its length, but
    # never later than the next bar's start (handles session-end short bars).
    starts = pd.Series(ctx.index)
    ends = starts + ctx_bar
    nxt = starts.shift(-1).fillna(ends.iloc[-1])
    done = ends.where(ends <= nxt, nxt).to_numpy()
    return pd.DataFrame({"done": done, "trend": trend, "bias": bias})


def prepare(symbol: str, base: pd.DataFrame, ctx: pd.DataFrame, spec: dict) -> dict:
    """Per-bar indicator flags plus the context known when each bar closed."""
    close, high, low, vol = base["Close"], base["High"], base["Low"], base["Volume"]
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    rsi = _rsi(close)
    mb, ms = _macd_bull_bear(close)
    e9 = close.ewm(span=9, adjust=False).mean()
    e21 = close.ewm(span=21, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    vavg = vol.rolling(20).mean()
    is_eu = market_of(symbol) == "eu"
    has_vol = np.ones(len(base), bool) if is_eu else \
        ((vol >= vavg * 1.1) | vavg.isna()).to_numpy()

    bar_done = base.index + spec["bar"]
    c = _context(ctx, spec["ctx_bar"])
    m = pd.merge_asof(pd.DataFrame({"t": bar_done}), c.sort_values("done"),
                      left_on="t", right_on="done", direction="backward")
    return {
        "symbol": symbol,
        "t": bar_done,                         # signal time = bar close
        "close": close.to_numpy(), "high": high.to_numpy(), "low": low.to_numpy(),
        "atr": atr.to_numpy(), "rsi": rsi.to_numpy(),
        "macd_bull": mb.to_numpy(), "macd_bear": ms.to_numpy(),
        "ema_bull": (e9 > e21).to_numpy(), "ema_bear": (e9 < e21).to_numpy(),
        "e50_up": (e50 > e50.shift(3)).to_numpy(),
        "e50_dn": (e50 < e50.shift(3)).to_numpy(),
        "has_vol": has_vol,
        "trend": m["trend"].fillna("unknown").to_numpy(),
        "bias": m["bias"].fillna("").to_numpy(),
        "valid": (np.arange(len(base)) >= 49) & atr.gt(0).to_numpy(),
    }


# ── signals and trade simulation ──────────────────────────────────────────────

def detect(p: dict, rsi_lo: float, filters: str) -> list[tuple[int, int]]:
    """Bar indices + direction (+1 BUY / -1 SELL) where the live rule fires."""
    rsi = p["rsi"]
    over_s, over_b = rsi < rsi_lo, rsi > 100 - rsi_lo
    buy = (over_s & (over_s.astype(int) + p["macd_bull"] + p["ema_bull"] >= 2)
           & p["has_vol"] & p["e50_up"] & p["valid"])
    sell = (over_b & (over_b.astype(int) + p["macd_bear"] + p["ema_bear"] >= 2)
            & p["has_vol"] & p["e50_dn"] & p["valid"] & ~buy)
    if "trend" in filters:
        buy &= p["trend"] != "down"
        sell &= p["trend"] != "up"
    if "mtf" in filters:
        buy &= (p["bias"] == "") | (p["bias"] == "BUY")
        sell &= (p["bias"] == "") | (p["bias"] == "SELL")

    out, last = [], {}
    t = p["t"]
    dedup = pd.Timedelta(hours=DEDUP_HOURS)
    for i in np.flatnonzero(buy | sell):
        d = 1 if buy[i] else -1
        prev = last.get(d)
        if prev is not None and t[i] - prev < dedup:
            continue                      # live: same action within 6h is skipped
        last[d] = t[i]
        out.append((int(i), d))
    return out


def simulate(p: dict, sigs: list[tuple[int, int]], sl_mult: float,
             tp_mult: float, delay: int = 0) -> list[dict]:
    """Walk each trade forward like bot._resolve_signal_outcome."""
    t, close, high, low, atr = p["t"], p["close"], p["high"], p["low"], p["atr"]
    n = len(close)
    tmo = pd.Timedelta(hours=TIMEOUT_HOURS)
    trades = []
    for i, d in sigs:
        e = i + delay                     # delay>0: filled bars later (stale quotes)
        if e >= n - 1:
            continue
        entry = close[e]
        sl_dist = atr[i] * sl_mult
        if not sl_dist > 0:
            continue
        tp = entry + d * atr[i] * tp_mult
        sl = entry - d * sl_dist
        # bars that closed before the 24h deadline; data must extend past it
        end = int(t.searchsorted(t[e] + tmo, side="left"))
        if end >= n:
            continue                      # not enough future data to resolve
        hi, lo = high[e + 1:end], low[e + 1:end]
        if d > 0:
            hit_tp, hit_sl = hi >= tp, lo <= sl
        else:
            hit_tp, hit_sl = lo <= tp, hi >= sl
        k_tp = int(np.argmax(hit_tp)) if hit_tp.any() else None
        k_sl = int(np.argmax(hit_sl)) if hit_sl.any() else None
        if k_tp is None and k_sl is None:
            out, px = "timeout", close[end - 1]
        elif k_sl is None or (k_tp is not None and k_tp < k_sl):
            out, px = "tp", tp
        elif k_tp is None or k_sl < k_tp:
            out, px = "sl", sl
        else:
            out, px = "both", close[e + 1 + k_tp]   # ambiguous bar -> close
        trades.append({
            "symbol": p["symbol"], "t": t[i], "dir": d, "outcome": out,
            "r": d * (px - entry) / sl_dist,
            "sl_pct": sl_dist / entry * 100,
        })
    return trades


# ── evaluation ────────────────────────────────────────────────────────────────

def score(trades: list[dict], cost_scale: float = 1.0,
          cost_override: float | None = None) -> dict:
    if not trades:
        return {"n": 0, "avg_r": 0.0, "ci_lo": None, "ci_hi": None, "verdict": "INSUFFICIENT DATA"}
    rs, days = [], []
    for tr in trades:
        c = cost_override if cost_override is not None else \
            DEFAULT_COST_PCT[market_of(tr["symbol"])] * cost_scale
        rs.append(tr["r"] - c / tr["sl_pct"])
        days.append(tr["t"].strftime("%Y-%m-%d"))
    wins = sum(1 for tr in trades if tr["outcome"] == "tp")
    s = summarize_r(rs, days, wins)
    s["gross_r"] = round(float(np.mean([tr["r"] for tr in trades])), 3)
    return s


def _key(c: dict) -> str:
    return f"rsi{c['rsi']}/{100 - c['rsi']} sl{c['sl']} tp{c['tp']} {c['filters']}"


def _neighbours(cfg: dict) -> list[dict]:
    out = []
    for k in ("rsi", "sl", "tp"):
        vals = GRID[k]
        j = vals.index(cfg[k])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(vals):
                out.append({**cfg, k: vals[jj]})
    return out


def run(symbols: list[str], tf: str = "5m", download: bool = True,
        oos_frac: float = 1 / 3, cost_scale: float = 1.0, min_is: int = 30,
        source: str = "yf") -> dict:
    spec = TF_SPEC[tf]
    if source == "alpaca":
        if tf != "5m":
            raise SystemExit("--source alpaca supports --tf 5m only")
        symbols = [s for s in symbols if market_of(s) == "us"]
    prepared = []
    for s in symbols:
        if source == "alpaca":
            try:
                base = load_alpaca_5m(s, download)
            except Exception as e:
                print(f"  skip {s}: alpaca error {e}")
                continue
        else:
            base = load_bars(s, spec["interval"], spec["period"], download)
        if base is None or len(base) < 300:
            print(f"  skip {s}: no data")
            continue
        if source == "alpaca":
            ctx = hourly_from_5m(base)
        elif tf == "5m":
            ctx = load_bars(s, "1h", "730d", download)
        else:
            ctx = base.resample("4h").agg({"Open": "first", "High": "max", "Low": "min",
                                           "Close": "last", "Volume": "sum"}).dropna()
        if ctx is None or ctx.empty:
            print(f"  skip {s}: no context data")
            continue
        prepared.append(prepare(s, base, ctx, spec))
        if download and source == "yf":
            time.sleep(0.3)               # be gentle with yfinance
    if not prepared:
        raise SystemExit("no data")

    t0 = min(p["t"][0] for p in prepared)
    t1 = max(p["t"][-1] for p in prepared)
    cut = t0 + (t1 - t0) * (1 - oos_frac)

    def split(trades):
        return ([x for x in trades if x["t"] < cut], [x for x in trades if x["t"] >= cut])

    # Every grid config, trades over all symbols
    results = {}
    sig_cache = {}
    for rsi, flt in itertools.product(GRID["rsi"], GRID["filters"]):
        sig_cache[(rsi, flt)] = [(p, detect(p, rsi, flt)) for p in prepared]
    for rsi, sl, tp, flt in itertools.product(GRID["rsi"], GRID["sl"], GRID["tp"], GRID["filters"]):
        cfg = {"rsi": rsi, "sl": sl, "tp": tp, "filters": flt}
        trades = [tr for p, sg in sig_cache[(rsi, flt)] for tr in simulate(p, sg, sl, tp)]
        is_t, oos_t = split(trades)
        results[_key(cfg)] = {"cfg": cfg, "all": score(trades, cost_scale),
                              "is": score(is_t, cost_scale), "oos": score(oos_t, cost_scale),
                              "_trades": trades}

    live = results[_key(LIVE)]
    live_tr = live["_trades"]

    # Selection on in-sample ONLY
    eligible = [r for r in results.values() if r["is"]["n"] >= min_is]
    best = max(eligible, key=lambda r: r["is"]["avg_r"]) if eligible else live
    nb = [results[_key(c)] for c in _neighbours(best["cfg"])]
    plateau = (sum(1 for r in nb if r["oos"]["n"] and r["oos"]["avg_r"] > 0) / len(nb)) if nb else 0.0
    grid_oos_pos = sum(1 for r in results.values() if r["oos"]["n"] >= 10 and r["oos"]["avg_r"] > 0)
    grid_oos_n = sum(1 for r in results.values() if r["oos"]["n"] >= 10)

    o = best["oos"]
    if o["n"] < 30 or o["ci_lo"] is None:
        final = "INSUFFICIENT DATA"
    elif o["avg_r"] <= 0:
        final = "NO EDGE"
    elif o["ci_lo"] > 0 and plateau >= 0.6:
        final = "DATA VALIDATED"
    else:
        final = "TRADE-OFF"

    # Sensitivities on the live config
    cost_sens = {f"{c:.2f}%": score(live_tr, cost_override=c) for c in (0.0, 0.05, 0.10, 0.20)}
    delay_sens = {}
    sigs = sig_cache[(LIVE["rsi"], LIVE["filters"])]
    for dly in (0, 1, 3):
        tr = [x for p, sg in sigs for x in simulate(p, sg, LIVE["sl"], LIVE["tp"], dly)]
        delay_sens[f"{dly} bar"] = score(tr, cost_scale)
    by_market = {}
    for m in ("us", "eu", "crypto"):
        tr = [x for x in live_tr if market_of(x["symbol"]) == m]
        if tr:
            by_market[m] = score(tr, cost_scale)
    by_dir = {name: score([x for x in live_tr if x["dir"] == d], cost_scale)
              for name, d in (("BUY", 1), ("SELL", -1))}

    def _pub(r):
        return {k: v for k, v in r.items() if k != "_trades"}

    top_is = sorted(eligible, key=lambda r: r["is"]["avg_r"], reverse=True)[:10]
    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "tf": tf if source == "yf" else f"{tf}-{source}", "source": source,
        "symbols": len(prepared),
        "period": [str(t0)[:16], str(t1)[:16]], "oos_from": str(cut)[:16],
        "cost_pct": {m: round(c * cost_scale, 3) for m, c in DEFAULT_COST_PCT.items()},
        "live": _pub(live), "live_by_market": by_market, "live_by_dir": by_dir,
        "cost_sensitivity": cost_sens, "delay_sensitivity": delay_sens,
        "best_is": _pub(best), "best_neighbours": [_pub(r) for r in nb],
        "plateau_share": round(plateau, 2),
        "grid_oos_positive": f"{grid_oos_pos}/{grid_oos_n}",
        "top_is": [_pub(r) for r in top_is],
        "verdict": final,
    }


# ── report ────────────────────────────────────────────────────────────────────

def _fmt(s: dict) -> str:
    if not s.get("n"):
        return "n=0"
    ci = "" if s.get("ci_lo") is None else f" [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"
    wr = f" WR {s['win_rate']:.0f}%" if s.get("win_rate") is not None else ""
    return (f"n={s['n']:<5} gross {s.get('gross_r', 0):+.3f}R  net {s['avg_r']:+.3f}R{ci}"
            f"{wr}  → {s['verdict']}")


def report(res: dict) -> str:
    L = []
    L.append(f"=== Research: live rule on {res['tf']} bars, {res['symbols']} symbols ===")
    L.append(f"period {res['period'][0]} → {res['period'][1]}   held-out from {res['oos_from']}")
    L.append(f"costs (round trip %): {res['cost_pct']}")
    L.append("")
    L.append(f"LIVE CONFIG ({_key(LIVE)})")
    L.append(f"  all  {_fmt(res['live']['all'])}")
    L.append(f"  IS   {_fmt(res['live']['is'])}")
    L.append(f"  OOS  {_fmt(res['live']['oos'])}")
    for m, s in res["live_by_market"].items():
        L.append(f"  {m:<4} {_fmt(s)}")
    for m, s in res["live_by_dir"].items():
        L.append(f"  {m:<4} {_fmt(s)}")
    L.append("  cost sensitivity (all markets at a flat cost):")
    for k, s in res["cost_sensitivity"].items():
        L.append(f"    {k:<6} {_fmt(s)}")
    L.append("  entry-delay sensitivity (fill N bars after the signal bar):")
    for k, s in res["delay_sensitivity"].items():
        L.append(f"    {k:<6} {_fmt(s)}")
    L.append("")
    b = res["best_is"]
    L.append(f"BEST ON IN-SAMPLE: {_key(b['cfg'])}")
    L.append(f"  IS   {_fmt(b['is'])}")
    L.append(f"  OOS  {_fmt(b['oos'])}   ← the honest number")
    L.append(f"  neighbours positive OOS: {res['plateau_share']:.0%}")
    for r in res["best_neighbours"]:
        L.append(f"    {_key(r['cfg']):<34} OOS {_fmt(r['oos'])}")
    L.append(f"  whole grid positive OOS: {res['grid_oos_positive']}")
    L.append("")
    L.append("TOP 10 BY IN-SAMPLE (what an optimiser would pick) → their OOS:")
    for r in res["top_is"]:
        L.append(f"  {_key(r['cfg']):<34} IS {r['is']['avg_r']:+.3f}  OOS {r['oos']['avg_r']:+.3f} (n={r['oos']['n']})")
    L.append("")
    L.append(f"VERDICT: {res['verdict']}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tf", choices=list(TF_SPEC), default="5m")
    ap.add_argument("--symbols", help="comma-separated; default = active symbols")
    ap.add_argument("--no-download", action="store_true", help="use cache only")
    ap.add_argument("--cost-scale", type=float, default=1.0,
                    help="multiply default per-market costs (0 = gross)")
    ap.add_argument("--oos", type=float, default=1 / 3, help="held-out fraction")
    ap.add_argument("--source", choices=["yf", "alpaca"], default="yf",
                    help="alpaca = years of US 5m bars (needs ALPACA_API_* env)")
    args = ap.parse_args()
    if args.source == "alpaca" and alpaca_keys() is None and not args.no_download:
        raise SystemExit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set")

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        from database import get_active_symbols
        symbols = get_active_symbols()
    print(f"Loading {len(symbols)} symbols ({args.tf})…", file=sys.stderr)
    res = run(symbols, tf=args.tf, download=not args.no_download,
              oos_frac=args.oos, cost_scale=args.cost_scale, source=args.source)
    print(report(res))
    try:
        all_res = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else {}
    except Exception:
        all_res = {}
    all_res[res["tf"]] = res
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
