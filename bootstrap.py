"""
Bootstrap ML training from historical backtest outcomes.

Runs the live rule engine on 2 years of hourly data per symbol,
captures the ML feature vector at each signal bar, labels it from
the TP/SL/timeout outcome, and injects everything into model.train().

Run once (or periodically) to give the ML layer a real starting point
instead of training purely on synthetic price-pattern labels.

Usage:
    python bootstrap.py                    # all active symbols
    python bootstrap.py BTC-USD AAPL       # specific symbols
    python bootstrap.py --years 1          # shorter history
"""

from __future__ import annotations

import argparse
import logging
import sys
import os

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _fetch(symbol: str, years: int = 2, interval: str = "1h") -> pd.DataFrame:
    period = f"{min(years * 365, 730)}d"
    df = yf.download(symbol, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def _label_outcome(action: str, entry: float, tp: float, sl: float,
                   future_high: float, future_low: float) -> str:
    """Determine outcome from the next N bars' high/low range."""
    if action == "BUY":
        if future_high >= tp:
            return "correct"
        if future_low <= sl:
            return "incorrect"
    else:
        if future_low <= tp:
            return "correct"
        if future_high >= sl:
            return "incorrect"
    return "neutral"


def bootstrap_symbol(symbol: str, years: int = 2,
                     sl_mult: float = 1.5, tp_mult: float = 2.5,
                     outcome_bars: int = 24) -> list[dict]:
    """
    Walk through historical data, fire signals using the live rule engine,
    look ahead `outcome_bars` bars for TP/SL resolution, and return a list
    of {"features": {...}, "label": int, "outcome": str, "date": str}.
    """
    from signals import signal_from_df, get_bias_from_df, _atr
    from ml_signals import build_features

    logger.info(f"[bootstrap] {symbol}: fetching {years}y of 1h data...")
    df = _fetch(symbol, years=years)
    if len(df) < 300:
        logger.warning(f"[bootstrap] {symbol}: not enough data ({len(df)} bars)")
        return []

    # Market context: fetch once, pre-align to symbol index to avoid look-ahead per bar
    logger.info(f"[bootstrap] {symbol}: fetching market context (SPY/VIX)...")
    spy_full = _fetch("SPY", years=years, interval="1h")
    vix_full = _fetch("^VIX", years=years, interval="1d")

    def _pre_align(mkt_df: pd.DataFrame, base_index) -> pd.DataFrame | None:
        if mkt_df.empty:
            return None
        try:
            s = mkt_df.copy()
            if getattr(s.index, "tz", None) != getattr(base_index, "tz", None):
                if getattr(s.index, "tz", None) is None:
                    s.index = s.index.tz_localize("UTC")
                if getattr(base_index, "tz", None) is not None:
                    s.index = s.index.tz_convert(base_index.tz)
                else:
                    s.index = s.index.tz_localize(None)
            return s.reindex(base_index, method="ffill")
        except Exception as e:
            logger.debug(f"[bootstrap] market align failed: {e}")
            return None

    spy_aligned = _pre_align(spy_full, df.index)
    vix_aligned = _pre_align(vix_full, df.index)

    samples: list[dict] = []
    last_signal_bar = -999  # prevent same-direction repeat within 6 bars

    for i in range(200, len(df) - outcome_bars):
        window = df.iloc[:i + 1]

        # Bias from 4h resample of same data
        bias_df = window.resample("4h").agg({
            "Open": "first", "High": "max",
            "Low": "min", "Close": "last", "Volume": "sum",
        }).dropna()
        bias = get_bias_from_df(bias_df)

        rule_sig = signal_from_df(symbol, window)
        if rule_sig is None:
            continue

        # Simple dedup: skip if same direction fired in the last 6 bars
        if i - last_signal_bar < 6:
            continue
        last_signal_bar = i

        # Compute TP/SL
        entry = float(window["Close"].iloc[-1])
        atr_series = _atr(window)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        if atr_val <= 0:
            continue

        is_buy = rule_sig["action"] == "BUY"
        tp = entry + atr_val * tp_mult if is_buy else entry - atr_val * tp_mult
        sl = entry - atr_val * sl_mult if is_buy else entry + atr_val * sl_mult

        # Look ahead to determine outcome
        future = df.iloc[i + 1: i + 1 + outcome_bars]
        future_high = float(future["High"].max())
        future_low  = float(future["Low"].min())
        outcome = _label_outcome(rule_sig["action"], entry, tp, sl,
                                 future_high, future_low)

        # Capture ML features at signal bar — include market context up to bar i
        try:
            spy_w = spy_aligned.iloc[:i + 1] if spy_aligned is not None else None
            vix_w = vix_aligned.iloc[:i + 1] if vix_aligned is not None else None
            feat_df = build_features(window, spy_df=spy_w, vix_df=vix_w)
            feat_row = feat_df.dropna().iloc[-1]
            features = feat_row.to_dict()
        except Exception as e:
            logger.debug(f"[bootstrap] {symbol} bar {i} feature error: {e}")
            continue

        # Label: correct=+1 for BUY, -1 for SELL; incorrect=flipped; neutral=0
        if outcome == "correct":
            label = 1 if is_buy else -1
        elif outcome == "incorrect":
            label = -1 if is_buy else 1
        else:
            label = 0

        samples.append({
            "features": features,
            "label": label,
            "outcome": outcome,
            "date": str(df.index[i])[:10],
            "action": rule_sig["action"],
            "entry": round(entry, 4),
        })

    logger.info(
        f"[bootstrap] {symbol}: {len(samples)} signal samples — "
        f"correct={sum(1 for s in samples if s['outcome']=='correct')}  "
        f"incorrect={sum(1 for s in samples if s['outcome']=='incorrect')}  "
        f"neutral={sum(1 for s in samples if s['outcome']=='neutral')}"
    )
    return samples


def run_bootstrap(symbols: list[str], years: int = 2,
                  min_samples: int = 10,
                  job_id: int | None = None) -> dict[str, dict]:
    from ml_signals import StockModel
    from database import (log_training, get_outcome_count,
                          start_training_job, update_training_job,
                          finish_training_job)
    import json

    if job_id is None:
        job_id = start_training_job("bootstrap", total_symbols=len(symbols))
        logger.info(f"[bootstrap] Started job_id={job_id} for {len(symbols)} symbols")

    results = {}
    for idx, symbol in enumerate(symbols):
        logger.info(f"\n{'='*50}")
        logger.info(f"[bootstrap] Processing {symbol} ({idx+1}/{len(symbols)})...")
        update_training_job(job_id, done_symbols=idx, current_symbol=symbol)

        samples = bootstrap_symbol(symbol, years=years)
        if len(samples) < min_samples:
            logger.warning(f"[bootstrap] {symbol}: only {len(samples)} samples, skipping train")
            results[symbol] = {"samples": len(samples), "trained": False, "reason": "too few samples"}
            update_training_job(job_id, done_symbols=idx + 1, current_symbol=None)
            continue

        correct = sum(1 for s in samples if s["outcome"] == "correct")
        incorrect = sum(1 for s in samples if s["outcome"] == "incorrect")
        neutral = sum(1 for s in samples if s["outcome"] == "neutral")
        non_neutral = correct + incorrect
        win_rate = round(correct / max(1, non_neutral) * 100, 1)

        training_samples = [s for s in samples if s["outcome"] != "neutral"]
        logger.info(f"[bootstrap] {symbol}: training on {len(training_samples)} non-neutral samples")

        model = StockModel(symbol)
        ok = model.train(outcome_data=training_samples, _log=False)

        if ok:
            log_training(
                symbol,
                train_samples=len(training_samples),
                outcome_samples=len(training_samples),
                trigger="bootstrap",
                win_rate=win_rate,
                correct_count=correct,
                incorrect_count=incorrect,
                neutral_count=neutral,
                job_id=job_id,
            )

        results[symbol] = {
            "total_samples": len(samples),
            "training_samples": len(training_samples),
            "correct": correct,
            "incorrect": incorrect,
            "neutral": neutral,
            "win_rate": win_rate,
            "trained": ok,
        }

        if ok:
            logger.info(
                f"[bootstrap] {symbol}: ✓ model trained  "
                f"win_rate={win_rate}%  ({correct} correct / {incorrect} incorrect)"
            )
        else:
            logger.error(f"[bootstrap] {symbol}: ✗ training failed")

        update_training_job(job_id, done_symbols=idx + 1, current_symbol=None)

    trained_count = sum(1 for r in results.values() if r.get("trained"))
    avg_wr = round(
        sum(r["win_rate"] for r in results.values() if r.get("trained")) / max(1, trained_count), 1
    ) if trained_count else 0.0
    summary = json.dumps({"trained": trained_count, "total": len(symbols), "avg_win_rate": avg_wr})
    finish_training_job(job_id, status="done", result_summary=summary,
                        note=f"{trained_count}/{len(symbols)} trained, avg win rate {avg_wr}%")
    logger.info(f"[bootstrap] Job {job_id} complete — {trained_count}/{len(symbols)} trained")
    return results


def print_summary(results: dict[str, dict]) -> None:
    print("\n" + "="*60)
    print("BOOTSTRAP SUMMARY")
    print("="*60)
    print(f"{'Symbol':<12} {'Signals':>8} {'Win%':>6} {'Correct':>8} {'Wrong':>8} {'Trained':>8}")
    print("-"*60)
    for sym, r in results.items():
        if not r.get("trained"):
            print(f"{sym:<12} {'—':>8}  {'—':>6}  {'—':>8}  {'—':>8}  {'SKIP':>8}  {r.get('reason','')}")
            continue
        print(
            f"{sym:<12} {r['total_samples']:>8} {r['win_rate']:>5.1f}%"
            f" {r['correct']:>8} {r['incorrect']:>8} {'✓':>8}"
        )
    trained = sum(1 for r in results.values() if r.get("trained"))
    print(f"\n{trained}/{len(results)} symbols trained successfully.")
    win_rates = [r["win_rate"] for r in results.values() if r.get("trained")]
    if win_rates:
        avg_wr = sum(win_rates) / len(win_rates)
        verdict = "🟢 Strategy has edge" if avg_wr >= 52 else ("🟡 Marginal" if avg_wr >= 48 else "🔴 Below random")
        print(f"Average historical win rate: {avg_wr:.1f}%  {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap ML from historical backtest outcomes")
    parser.add_argument("symbols", nargs="*", help="Symbols to bootstrap (default: all active)")
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--min-samples", type=int, default=10, dest="min_samples")
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        try:
            from database import get_active_symbols, init_db
            init_db()
            symbols = get_active_symbols()
            if not symbols:
                logger.error("No active symbols in DB. Pass symbols as arguments.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Could not load symbols from DB: {e}")
            sys.exit(1)

    logger.info(f"Bootstrapping {len(symbols)} symbols: {symbols}")
    results = run_bootstrap(symbols, years=args.years, min_samples=args.min_samples)
    print_summary(results)
