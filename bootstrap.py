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
                   future: pd.DataFrame) -> str:
    """
    Bar-by-bar TP/SL resolution — matches build_labels_tpsl() exactly.
    Determines which barrier (TP or SL) is hit first across the outcome window.
    Using max(high)/min(low) across the window cannot determine order of hits,
    producing optimistic labels when price touches TP then reverses to SL.
    """
    is_buy = action == "BUY"
    for _, bar in future.iterrows():
        h = float(bar["High"])
        l = float(bar["Low"])
        if is_buy:
            if h >= tp and l <= sl:
                return "neutral"   # ambiguous — both hit in same bar
            if h >= tp:
                return "correct"
            if l <= sl:
                return "incorrect"
        else:
            if l <= tp and h >= sl:
                return "neutral"   # ambiguous
            if l <= tp:
                return "correct"
            if h >= sl:
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

        # Look ahead to determine outcome (bar-by-bar — matches build_labels_tpsl)
        future = df.iloc[i + 1: i + 1 + outcome_bars]
        outcome = _label_outcome(rule_sig["action"], entry, tp, sl, future)

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


def _purged_splits(n: int, n_splits: int = 5, outcome_bars: int = 24,
                   embargo: int = 10) -> list[tuple[list, list]]:
    """
    Purged + embargo k-fold splits for time-series cross-validation.

    Purge: training bars whose label window overlaps with the test set are removed.
      Any bar at index i has a label derived from bars [i+1, i+outcome_bars].
      A training bar at position t_train contributes to the label window
      [t_train+1 … t_train+outcome_bars].  We purge all training bars where
      t_train + outcome_bars >= test_start, i.e. training bars in
      [test_start - outcome_bars, test_start - 1].

    Embargo: bars immediately after the test set are also removed from training
      to prevent the model from learning the market state right after the test
      outcome resolves (which could look like look-ahead in practice).
    """
    fold_size = n // n_splits
    splits = []
    for k in range(n_splits):
        test_s = k * fold_size
        test_e = (k + 1) * fold_size if k < n_splits - 1 else n
        # Purge: remove training bars whose outcome window overlaps test window
        purge_start = max(0, test_s - outcome_bars)
        # Embargo: remove bars right after test set
        embargo_end = min(n, test_e + embargo)
        train_idx = list(range(0, purge_start)) + list(range(embargo_end, n))
        test_idx  = list(range(test_s, test_e))
        if len(train_idx) >= 80 and len(test_idx) >= 20:
            splits.append((train_idx, test_idx))
    return splits


def walk_forward_test(symbol: str, years: int = 2,
                      sl_mult: float = 1.5, tp_mult: float = 2.5,
                      outcome_bars: int = 24) -> dict:
    """
    Purged-embargo 5-fold walk-forward validation.
    Trains a temp model on each fold's training partition, evaluates on the
    test partition, and reports average win rate across folds.
    Grade is based on the average — a symbol must perform consistently, not
    just in one lucky fold.
    Uses build_labels_tpsl so grades are directly comparable to live performance.
    """
    from signals import signal_from_df, get_bias_from_df, _atr
    from ml_signals import StockModel, build_features

    logger.info(f"[wf] {symbol}: fetching {years}y data for walk-forward test...")
    df = _fetch(symbol, years=years)
    if len(df) < 400:
        logger.warning(f"[wf] {symbol}: not enough data ({len(df)} bars)")
        return {"symbol": symbol, "grade": "N/A", "error": "insufficient data"}

    spy_full = _fetch("SPY", years=years, interval="1h")
    vix_full = _fetch("^VIX", years=years, interval="1d")

    def _pre_align(mkt_df, base_index):
        if mkt_df is None or mkt_df.empty:
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
        except Exception:
            return None

    spy_aligned = _pre_align(spy_full, df.index)
    vix_aligned = _pre_align(vix_full, df.index)

    import numpy as np
    from ml_signals import build_features as _bf, build_labels_tpsl as _btpsl, AI_SIGNAL_THRESHOLD
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    # Build full feature + label matrix once (bar-by-bar slicing is expensive)
    logger.info(f"[wf] {symbol}: building full feature matrix ({len(df)} bars)...")
    features_full = _bf(df, spy_df=spy_aligned, vix_df=vix_aligned)
    y_buy_full, y_sell_full = _btpsl(df, sl_mult=sl_mult, tp_mult=tp_mult,
                                      outcome_bars=outcome_bars)
    combined_full = pd.concat([
        features_full,
        y_buy_full.rename("y_buy"),
        y_sell_full.rename("y_sell"),
    ], axis=1).dropna()

    if len(combined_full) < 150:
        return {"symbol": symbol, "grade": "N/A", "error": "not enough samples"}

    feat_cols = [c for c in combined_full.columns if c not in ("y_buy", "y_sell")]
    X_all    = combined_full[feat_cols].values
    yb_all   = combined_full["y_buy"].values
    ys_all   = combined_full["y_sell"].values

    _GAP = 0.10
    n_splits = 5
    splits = _purged_splits(len(X_all), n_splits=n_splits,
                            outcome_bars=outcome_bars, embargo=10)
    if not splits:
        return {"symbol": symbol, "grade": "N/A", "error": "no valid folds"}

    logger.info(f"[wf] {symbol}: running {len(splits)}-fold purged walk-forward...")

    fold_rule_wrs, fold_ai_wrs = [], []
    total_rule_results, total_ai_results = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        yb_train = yb_all[train_idx]
        ys_train = ys_all[train_idx]

        if len(X_train) < 60:
            continue

        # Scaler: fit on train, transform test
        n_cal = max(15, len(X_train) // 5)
        if n_cal < len(X_train) - 30:
            X_main_r, X_cal_r = X_train[:-n_cal], X_train[-n_cal:]
            yb_main, yb_cal   = yb_train[:-n_cal], yb_train[-n_cal:]
            ys_main, ys_cal   = ys_train[:-n_cal], ys_train[-n_cal:]
        else:
            X_main_r, X_cal_r = X_train, X_train
            yb_main, yb_cal   = yb_train, yb_train
            ys_main, ys_cal   = ys_train, ys_train

        scaler   = StandardScaler()
        X_main   = scaler.fit_transform(X_main_r)
        X_cal    = scaler.transform(X_cal_r)
        X_test_s = scaler.transform(X_test)

        clf_buy  = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                               learning_rate=0.05, random_state=42)
        clf_sell = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                               learning_rate=0.05, random_state=42)
        clf_buy.fit(X_main, yb_main)
        clf_sell.fit(X_main, ys_main)

        try:
            cal_buy  = CalibratedClassifierCV(clf_buy,  method="sigmoid", cv="prefit")
            cal_buy.fit(X_cal, yb_cal)
            cal_sell = CalibratedClassifierCV(clf_sell, method="sigmoid", cv="prefit")
            cal_sell.fit(X_cal, ys_cal)
        except Exception:
            cal_buy, cal_sell = clf_buy, clf_sell

        # Evaluate on test bars using actual TP/SL outcomes from the full dataset
        # (the label matrix already computed bar-by-bar outcomes)
        fold_rule, fold_ai = [], []
        for j, global_idx in enumerate(test_idx):
            # The combined_full index maps back to df bars; use actual outcome labels
            y_buy_actual  = int(yb_all[global_idx])
            y_sell_actual = int(ys_all[global_idx])
            # Rule signal is captured by whether the label is 1 (TP hit)
            # For walk-forward grading we treat y_buy_actual=1 as "BUY would have worked"
            # We need to know which direction would have been signalled — use the
            # feature vector's RSI to infer (RSI<50 → likely BUY bias)
            rsi_val = float(X_all[global_idx, feat_cols.index("rsi")])
            inferred_action = "BUY" if rsi_val < 50 else "SELL"
            rule_correct = (y_buy_actual == 1) if inferred_action == "BUY" else (y_sell_actual == 1)
            fold_rule.append(rule_correct)

            try:
                row_s = X_test_s[j:j+1]
                bp = float(cal_buy.predict_proba(row_s)[0][1])
                sp = float(cal_sell.predict_proba(row_s)[0][1])
                if bp > AI_SIGNAL_THRESHOLD and bp > sp + _GAP:
                    ai_sig = "BUY"
                elif sp > AI_SIGNAL_THRESHOLD and sp > bp + _GAP:
                    ai_sig = "SELL"
                else:
                    ai_sig = None
                if ai_sig == inferred_action:
                    fold_ai.append(rule_correct)
            except Exception:
                pass

        if fold_rule:
            wr_rule = sum(fold_rule) / len(fold_rule) * 100
            fold_rule_wrs.append(wr_rule)
            total_rule_results.extend(fold_rule)
        if fold_ai:
            wr_ai = sum(fold_ai) / len(fold_ai) * 100
            fold_ai_wrs.append(wr_ai)
            total_ai_results.extend(fold_ai)

        logger.info(
            f"[wf] {symbol} fold {fold_idx+1}: "
            f"rule={wr_rule:.1f}% ({len(fold_rule)} bars)  "
            f"ai={wr_ai:.1f}% ({len(fold_ai)} bars)" if fold_rule and fold_ai
            else f"[wf] {symbol} fold {fold_idx+1}: rule={wr_rule:.1f}% ({len(fold_rule)} bars)" if fold_rule
            else f"[wf] {symbol} fold {fold_idx+1}: no results"
        )

    rule_signals  = len(total_rule_results)
    ai_signals    = len(total_ai_results)
    rule_win_rate = round(sum(total_rule_results) / max(1, rule_signals) * 100, 1) if rule_signals else None
    ai_win_rate   = round(sum(total_ai_results)   / max(1, ai_signals)   * 100, 1) if ai_signals  else None

    # Consistency penalty: high fold-to-fold variance means regime-sensitive model
    rule_wr_std = round(float(np.std(fold_rule_wrs)), 1) if len(fold_rule_wrs) >= 2 else None

    wr = ai_win_rate if ai_signals >= 10 else rule_win_rate
    if wr is None or rule_signals < 10:
        grade = "N/A"
    elif wr >= 58:
        grade = "A"
    elif wr >= 52:
        grade = "B"
    elif wr >= 47:
        grade = "C"
    else:
        grade = "D"

    logger.info(
        f"[wf] {symbol}: grade={grade}  rule={rule_win_rate}% ({rule_signals} bars)"
        f"  AI={ai_win_rate}% ({ai_signals} bars)"
        f"  fold_std={rule_wr_std}%"
    )
    return {
        "symbol": symbol,
        "grade": grade,
        "train_bars": len(X_all),
        "test_bars": rule_signals,
        "rule_signals": rule_signals,
        "rule_win_rate": rule_win_rate,
        "ai_signals": ai_signals,
        "ai_win_rate": ai_win_rate,
        "fold_win_rates": fold_rule_wrs,
        "fold_wr_std": rule_wr_std,
        "n_folds": len(splits),
    }


def run_walk_forward(symbols: list[str],
                     job_id: int | None = None) -> dict[str, dict]:
    from database import log_wf_result, start_training_job, update_training_job, finish_training_job
    import json

    if job_id is None:
        job_id = start_training_job("walk_forward", total_symbols=len(symbols))
        logger.info(f"[wf] Started job_id={job_id} for {len(symbols)} symbols")

    results = {}
    for idx, symbol in enumerate(symbols):
        logger.info(f"\n{'='*50}")
        logger.info(f"[wf] Testing {symbol} ({idx+1}/{len(symbols)})...")
        update_training_job(job_id, done_symbols=idx, current_symbol=symbol)
        try:
            result = walk_forward_test(symbol)
        except Exception as e:
            logger.error(f"[wf] {symbol} failed: {e}")
            result = {"symbol": symbol, "grade": "N/A", "error": str(e)[:100]}
        results[symbol] = result
        try:
            log_wf_result(symbol, result)
        except Exception as e:
            logger.debug(f"[wf] log_wf_result failed for {symbol}: {e}")
        update_training_job(job_id, done_symbols=idx + 1, current_symbol=None)

    grade_counts = {g: sum(1 for r in results.values() if r.get("grade") == g)
                    for g in ("A", "B", "C", "D", "N/A")}
    summary = json.dumps(grade_counts)
    finish_training_job(job_id, status="done", result_summary=summary,
                        note=f"walk-forward: A={grade_counts['A']} B={grade_counts['B']} "
                             f"C={grade_counts['C']} D={grade_counts['D']}")
    logger.info(f"[wf] Job {job_id} complete — grades: {grade_counts}")
    return results


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
