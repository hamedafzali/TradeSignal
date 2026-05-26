# TradeSignal — Project Summary & Roadmap

## Current Status (MVP — May 2026)

The core loop is complete and running:

**Signal → Filter → Broadcast → Track → Retrain**

| Component | Status | Notes |
|-----------|--------|-------|
| Technical signals (RSI/MACD/EMA) | ✅ Live | 5-min candles, stocks + crypto |
| Multi-timeframe filter | ✅ Live | 1h bias must agree with 5m signal |
| ML confidence filter | ✅ Live | GradientBoosting, ~720 samples/symbol |
| Sentiment suppression | ✅ Live | FinBERT via Finnhub news |
| Telegram channel broadcast | ✅ Live | Persian/beginner, market hours only |
| Outcome tracking (24h) | ✅ Live | Correct/incorrect/neutral per signal |
| Continuous learning | ✅ Live | Auto-retrains on 3+ new outcomes |
| Bootstrap training | ✅ Done | 7 symbols, avg win rate ~53% |
| Web dashboard | ✅ Live | Port 5002, training/signal/backtest views |

---

## Trading Expert Review — Honest Assessment

### What works well
- The multi-timeframe filter (1h + 5m agreement) is the single most effective suppressor of false signals. In backtests, removing MTF-contradicted signals improved accuracy by ~6–8%.
- The FinBERT sentiment layer adds a meaningful edge for stocks with heavy news flow (AAPL, NVDA, TSLA). Crypto sentiment from Finnhub is unreliable — no headlines available on free tier.
- The GradientBoosting model adds value: 53–56% win rate vs ~50% baseline is modest but real. With more real outcomes (300+), this should improve to 58–62%.

### Fundamental weaknesses at MVP
1. **No risk management.** Signals have static TP/SL as a percentage of price. Real trading requires ATR-based position sizing, portfolio-level risk limits, and correlation awareness. Two simultaneous BUY signals on NVDA and AAPL during a market-wide selloff are not independent.
2. **RSI thresholds are fixed.** RSI 35/65 is arbitrary. In trending markets, RSI stays overbought for weeks. A regime-aware threshold (trending vs ranging) would reduce false signals in trending conditions.
3. **5-minute signals on stocks are too short-term.** At 5m resolution, spread and slippage eat most of the edge, especially for retail traders. 15m or 30m would be more tradeable.
4. **Crypto signals are weak.** Crypto has no pre/post market and is 24/7, but the signal logic is identical to stocks. Crypto volatility regimes are very different — the same RSI thresholds produce too many false signals during sideways chop.
5. **No position sizing.** The bot sends signals but has no concept of portfolio allocation. A user acting on all signals simultaneously is implicitly equal-weighting a highly correlated basket.
6. **Win rate ≠ profitability.** 53% win rate with 1:1.7 R/R is theoretically profitable, but assumes clean fills at the exact signal price, which never happens in practice.

---

## Future Roadmap

### Phase 2 — Signal Quality (1–2 months)

**P1 — Higher priority**
- [ ] **Switch stocks to 15m candles** — better signal-to-noise than 5m, still actionable intraday
- [ ] **Regime detection** — classify market as trending/ranging using ADX or Hurst exponent; apply different RSI thresholds per regime
- [ ] **ATR-based TP/SL** — replace fixed percentage targets with `entry ± N×ATR(14)`, makes TP/SL adaptive to current volatility
- [ ] **Volume confirmation** — only fire signals when volume is above 20-period average; eliminates many false breakouts
- [ ] **Earnings/event blackout** — suppress signals 2 days before and 1 day after earnings announcements (fetch calendar from Finnhub)

**P2 — Medium priority**
- [ ] **Crypto-specific signal logic** — separate RSI thresholds (40/60 instead of 35/65), longer lookback for trend detection, volume profile awareness
- [ ] **Trend filter** — only take BUY signals when price is above EMA200, only SELL when below; eliminates counter-trend trades in strong trends
- [ ] **Signal cooldown per symbol** — current 6h cooldown is too aggressive during active market hours, too loose in flat markets; make it dynamic based on ATR

### Phase 3 — ML Improvement (2–4 months)

- [ ] **More features** — add volume ratio, Bollinger Band position, ATR percentile, day-of-week, time-of-day; these are strong predictors that the current model doesn't use
- [ ] **SHAP explainability** — understand which features the model weights most heavily per symbol; prune noise features
- [ ] **Per-symbol hyperparameter tuning** — AAPL and BTC have very different volatility profiles; one set of GradientBoosting parameters is suboptimal
- [ ] **Walk-forward validation** — current bootstrap doesn't properly validate out-of-sample; add a hold-out period and track model degradation over time
- [ ] **Ensemble model** — combine GradientBoosting with a second model (XGBoost or LightGBM) and only signal when both agree; reduces false positives

### Phase 4 — Risk Management (3–6 months)

- [ ] **Portfolio-level exposure limit** — cap total open signals at N simultaneous positions
- [ ] **Sector correlation filter** — don't send NVDA + MSFT + AAPL BUY simultaneously if they're in the same sector rally; one of them is redundant
- [ ] **Dynamic position sizing** — output a suggested allocation % based on signal quality score and current portfolio exposure (requires user balance input)
- [ ] **Max daily loss circuit breaker** — if N signals in a row are incorrect, pause broadcasting for the rest of the day
- [ ] **Drawdown tracking** — track running P&L of all signals as if traded at $1,000/signal; display equity curve on dashboard

### Phase 5 — Data & Intelligence (4–8 months)

- [ ] **Alternative sentiment sources** — Reddit (r/wallstreetbets volume spike), Twitter/X mention velocity, SEC filing alerts via EDGAR
- [ ] **Macro regime awareness** — suppress risk assets (tech stocks, crypto) when VIX > 25 or during Fed announcement windows
- [ ] **Earnings calendar integration** — automatically track upcoming earnings and adjust signal sensitivity near event dates
- [ ] **Options flow integration** — unusual options activity (large call/put sweeps) is a leading indicator; correlate with signal direction
- [ ] **Crypto-specific data** — funding rate (perpetual futures), exchange inflows/outflows, whale wallet alerts as sentiment signals

### Phase 6 — Platform (6–12 months)

- [ ] **Multi-user support** — each user sets their own risk tolerance, preferred symbols, and language; receives personalized signal selection
- [ ] **Backtesting engine improvement** — current walk-forward is basic; add Monte Carlo simulation, slippage/spread modeling, realistic fill assumptions
- [ ] **Alert webhooks** — POST signals to external systems (TradingView, broker APIs, personal webhooks)
- [ ] **Dashboard authentication** — the dashboard is currently open on the local network; add login before exposing externally
- [ ] **Broker integration (read-only)** — connect to Interactive Brokers or Alpaca API to show real account P&L alongside signal P&L for comparison

---

## Key Metrics to Track

Once the bot has accumulated 200+ real signal outcomes, these metrics matter:

| Metric | Current | Target |
|--------|---------|--------|
| Win rate | ~53% | >58% |
| Avg R/R ratio | 1:1.7 | >1:2.0 |
| Expected value per signal | ~+0.07% | >+0.15% |
| Sharpe ratio (simulated) | unknown | >1.5 |
| Max drawdown (simulated) | unknown | <15% |
| Signals per day | ~3–5 | 5–10 (quality-filtered) |

---

## Infrastructure Notes

- **Server:** Intel i7-4500U, 5.5GB RAM — adequate for current load, will need upgrade if adding transformer-based models or serving multiple users
- **FinBERT inference:** ~3–5 seconds per batch on CPU — acceptable since it runs in background cache refresh, never blocking signal generation
- **Data source:** yfinance (free) — reliable for EOD and intraday data but has rate limits; consider Polygon.io for production-grade data at scale
- **Database:** SQLite — fine for single-server deployment, but replace with PostgreSQL before multi-user or high-frequency use

---

## What This Is Not

- This is **not** financial advice
- This is **not** a fully automated trading system — it generates signals for humans to act on
- Win rates above 50% on historical data do not guarantee future profitability
- All signals should be evaluated in the context of your own risk tolerance and position sizing rules
