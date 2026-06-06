# TradeSignal — Project Summary & Roadmap

## Current Status (June 2026 — Post Quant Audit)

The core loop is complete and running with a full production-grade overhaul applied:

**Signal → Filter → Broadcast → Track → Retrain**

| Component | Status | Notes |
|-----------|--------|-------|
| Technical signals (RSI/MACD/EMA) | ✅ Live | 5m candles for stocks, 1h for crypto |
| EMA50 slope guard | ✅ Live | Prevents buying falling-knife entries |
| Multi-timeframe filter | ✅ Live | 1h bias must agree with 5m signal |
| ML confidence filter — 1h model | ✅ Live | GradientBoosting, trained on 1h candles for bias |
| ML confidence filter — 5m model | ✅ Live | Separate GBM trained on 60d of 5m data (~4,660 bars) |
| Probability calibration | ✅ Live | Platt sigmoid via CalibratedClassifierCV |
| TP/SL-aligned training labels | ✅ Live | Bar-by-bar simulation matching live outcome resolution |
| Sentiment suppression | ✅ Live | FinBERT via Finnhub news |
| Regime filter | ✅ Live | VIX spike / SPY drop / ATR spike suppression |
| Per-symbol AI capability gate | ✅ Live | Suppresses AI-only signals below 52% bootstrap WR |
| Bootstrap training | ✅ Done | Bar-by-bar honest labeling (no optimistic bias) |
| Walk-forward validation | ✅ Live | 5-fold purged-embargo, TP/SL objective |
| Performance brake | ✅ Live | Halves scan frequency if 20-trade WR drops below 42% |
| Concurrent signal limit | ✅ Live | Max 3 same-direction signals per scan cycle |
| Web dashboard | ✅ Live | Port 5002, training/signal/backtest/WF views |

---

## Quant Audit — What Was Fixed (June 2026)

A full quant systems review identified and fixed the following in 5 phases:

### Phase 1 — Broken Foundations

| Bug | Impact | Fix |
|-----|--------|-----|
| `_label_outcome()` used `max(high)/min(low)` — could not determine which barrier hit first | Bootstrap win rates inflated 10-15 pp | Bar-by-bar loop matching `build_labels_tpsl()` exactly |
| `StandardScaler.fit_transform(X)` before train/cal split | Calibration accuracy optimistic | Split X first, `scaler.fit(X_main)` only |
| `walk_forward_test()` used `build_labels()` (8-bar return) vs live `build_labels_tpsl()` | WF grade measured wrong objective | Replaced with `build_labels_tpsl` in WF |
| EMA50 guard defaulted to `True` when `<4 bars` | Fail-open instead of fail-safe | Changed to `False` — insufficient data blocks signal |
| 5× outcome row duplication | GBM memorised exact feature vectors | Replaced with `sample_weight=5.0` in `clf.fit()` |

**Measured impact:** Bootstrap win rates dropped from inflated ~50% to honest 34-41%, confirming the optimistic bias was real.

### Phase 2 — Timeframe Mismatch

The ML model was trained on 60d of 1h candles but used to confirm 5m rule signals — RSI/MACD/EMA distributions are statistically different across timeframes, making predictions uncorrelated with 5m outcomes.

**Fix:** `StockModel` now has two classifier pairs:
- `clf_buy / clf_sell` (1h model) — for bias/directional context
- `clf_buy_5m / clf_sell_5m` (5m model) — for confirming 5m rule signals

5m model fetches `period="60d", interval="5m"` → ~4,660 training bars (11× more than before). `bot.py` routes `model.predict(df_5m, timeframe="5m")` for rule confirmation.

### Phase 3 — Feature Engineering

**Removed** (correlated, no independent information):
- `ema_ratio`, `roc_3`, `roc_5`

**Added** (volatility regime, momentum quality, time structure):

| Feature | What It Captures |
|---------|-----------------|
| `atr_percentile` | Is vol expanding or contracting vs 90-bar history? |
| `gk_vol` | Garman-Klass realized vol using OHLC (more efficient than close-only) |
| `vol_of_vol` | Is the volatility regime stable or chaotic? |
| `mom_consistency` | What % of last 10 bars moved in the same direction? |
| `mom_accel` | Is momentum getting faster or slower? |
| `vwap_spread` | Proxy for institutional vs retail price positioning |
| `hour_sin / hour_cos` | Cyclical intraday pattern encoding |
| `dow_sin / dow_cos` | Day-of-week seasonality |

**Result:** 10 → 22 features. Max feature importance 12.5% (no dominance). Volatility features (`vol_of_vol`, `gk_vol`, `atr_pct`) in top 5 for 5m model.

### Phase 4 — Honest Validation

Replaced single 75/25 walk-forward split with **5-fold purged + embargo cross-validation**:
- **Purge:** removes training bars whose label window overlaps the test fold (prevents label leakage)
- **Embargo:** removes 10 bars after test fold (prevents market-state leakage)
- **Grade:** based on aggregate win rate across all 5 folds — requires consistent performance, not a lucky single period
- **Fold std reported:** high variance = regime-sensitive model

### Phase 5 — Risk Controls

| Control | Behaviour |
|---------|-----------|
| Performance brake | Tracks last 20 resolved outcomes; if WR < 42%, halves scan frequency until recovery |
| Concurrent signal limit | Max 3 same-direction signals per scan cycle — prevents correlated position accumulation |
| Retrain threshold | Raised from 3 to 25 new outcomes — below 25, the update is noise, not signal |

---

## Architecture (Updated)

```
yfinance (5m + 1h + SPY + VIX)
    │
    ▼
signals.py          ← RSI/MACD/EMA + EMA50 slope guard (5m candles)
    │
    ▼
ml_signals.py       ← 5m GBM (confirms 5m signals)
                       1h GBM (bias / directional context)
                       22-feature set: price + vol regime + momentum + time
                       Platt calibration (scaler fit on train only)
    │
    ▼
sentiment.py        ← FinBERT / Gemini / Claude news sentiment
    │
    ▼
bot.py              ← Regime filter + AI capability gate
                       Performance brake + concurrent limit
                       Telegram broadcast → @channel
    │
    ▼
check_outcomes      ← Bar-by-bar TP/SL resolution (honest)
    │
    ▼
bootstrap.py        ← 5-fold purged walk-forward validation
                       Bar-by-bar label resolution (matches live)
```

---

## Key Metrics

| Metric | Before Audit | After Audit | Target |
|--------|-------------|-------------|--------|
| Bootstrap win rate (reported) | ~50-55% (inflated) | 34-41% (honest) | Honest baseline |
| Training samples (5m model) | 0 (no 5m model) | ~4,660 bars | ✅ |
| Feature count | 10 | 22 | ✅ |
| WF validation folds | 1 (single split) | 5 (purged) | ✅ |
| Retrain trigger | 3 outcomes | 25 outcomes | ✅ |
| Max same-direction signals/scan | unlimited | 3 | ✅ |

---

## Remaining Roadmap

### Near-term
- [ ] **Meta-labeling** — train a secondary classifier to predict whether the primary classifier is correct; separates direction from confidence
- [ ] **Per-symbol feature pruning** — drop features where importance < 0.3× mean after each training run
- [ ] **Drift detection (PSI)** — compute Population Stability Index nightly; trigger retrain when PSI > 0.2
- [ ] **Regime classification** — 3-state HMM (bull/bear/volatile); store regime with each signal for attribution

### Medium-term
- [ ] **Sector ETF relative strength** — add XLK, XLF, XLE relative strength as features
- [ ] **Sentiment as feature** — inject FinBERT score as float feature into ML model instead of binary gate
- [ ] **Kelly position sizing** — compute half-Kelly fraction per signal using walk-forward win rate + RR
- [ ] **Earnings calendar blackout** — suppress signals ±2 days around earnings

### Long-term
- [ ] **Alternative data** — Reddit mention velocity, options flow, SEC filing alerts
- [ ] **Portfolio equity curve** — track running P&L of all signals as if traded at fixed size
- [ ] **Multi-user support** — per-user risk tolerance, symbol selection, language

---

## Infrastructure

- **Server:** Intel i7-4500U, 5.5GB RAM — adequate for current load
- **Data source:** yfinance (free) — reliable for intraday; consider Polygon.io at scale
- **Database:** SQLite at `/app/data/trading.db` (inside Docker container)
- **Deploy:** `git push origin main` → SSH → `git fetch && git reset --hard origin/main && docker compose restart bot`

---

## What This Is Not

- This is **not** financial advice
- This is **not** a fully automated trading system — it generates signals for humans to act on
- Honest win rates of 34-41% on the rule engine alone are below breakeven without position sizing and risk management
- All signals should be evaluated in the context of your own risk tolerance
