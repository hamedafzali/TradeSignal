# TradeSignal 📈

An autonomous trading signal bot that generates, filters, and broadcasts BUY/SELL signals to a Telegram channel. Built with Python, scikit-learn, FinBERT, and Docker.

---

## What It Does

- **Scans** stocks and crypto every 5 minutes using RSI, MACD, EMA crossovers, and EMA50 slope guard
- **Filters** signals with two per-symbol ML models: a 5m-trained model for rule confirmation and a 1h-trained model for directional bias
- **Calibrates** probabilities via Platt sigmoid so confidence scores reflect true win rates
- **Suppresses** signals that contradict current news sentiment (FinBERT or Gemini)
- **Validates** each symbol with 5-fold purged walk-forward using TP/SL-aligned labels
- **Broadcasts** to a Telegram channel in Persian or English, beginner or expert format
- **Tracks outcomes** via bar-by-bar TP/SL resolution — same logic as training labels
- **Retrains automatically** as real outcomes accumulate, with a performance brake for drawdown protection

---

## Architecture

```
yfinance (5m + 1h candles + SPY + VIX market context)
    │
    ▼
signals.py          ← RSI / MACD / EMA / EMA50 slope guard / multi-timeframe filter
    │
    ▼
ml_signals.py       ← 5m GBM (confirms 5m rule signals, ~4,660 training bars)
                       1h GBM (directional bias context)
                       22-feature set: direction + vol regime + momentum + time cyclical
                       Platt calibration (no scaler leakage)
    │
    ▼
sentiment.py        ← FinBERT / Gemini / Claude news sentiment check
    │
    ▼
bot.py              ← Regime filter (VIX/SPY/ATR)
                       AI capability gate (bootstrap WR < 52% → suppress AI-only)
                       Performance brake (WR < 42% over 20 trades → halve scan rate)
                       Concurrent limit (max 3 same-direction signals per scan)
                       Telegram broadcast → @channel
    │
    ▼
check_outcomes      ← Bar-by-bar TP/SL resolution (matches training labels exactly)
    │
    ▼
bootstrap.py        ← Historical training via 5-fold purged walk-forward validation
```

**Services (Docker Compose):**
| Container | Role | Port |
|-----------|------|------|
| `tradesignal-bot` | Telegram bot + scanner | — |
| `tradesignal-dashboard` | Web UI | 5002 |
| `tradesignal-finbert` | Local FinBERT sentiment API | 5004→5001 |

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Telegram bot token ([BotFather](https://t.me/BotFather))
- Telegram channel (bot must be admin)

### 1. Clone and configure
```bash
git clone https://github.com/hamedafzali/TradeSignal.git
cd TradeSignal
cp .env.example .env   # edit with your tokens
```

**.env**
```
BOT_TOKEN=your_telegram_bot_token
ADMIN_CHAT_ID=your_telegram_user_id
CHANNEL_ID=@YourChannelUsername
SYMBOLS=AAPL,TSLA,NVDA,MSFT,BTC-USD,ETH-USD
```

### 2. Start
```bash
docker compose up -d
```

### 3. Train models
Open the dashboard at `http://localhost:5002` → **Admin** → **Run Bootstrap**

This downloads 2 years of 1h history + 60 days of 5m history per symbol, trains both the 1h and 5m ML models, and runs walk-forward validation (~15 min for 6 symbols).

### 4. Enable sentiment (optional)
In the dashboard Admin panel:
- **Sentiment Provider**: `Local FinBERT` (already running) or `Gemini 1.5 Flash` (free tier)
- **News Provider**: `Finnhub` (free key at [finnhub.io](https://finnhub.io))

---

## Signal Logic

### Rule Engine (5m candles)

A rule signal fires when **RSI is mandatory** + **at least 2 of 3** conditions are met, volume is sufficient, and the EMA50 slope guard passes:

| Indicator | BUY condition | SELL condition |
|-----------|--------------|----------------|
| RSI (14) | < 40 (oversold) | > 60 (overbought) |
| MACD | above signal line | below signal line |
| EMA | EMA9 above EMA21 | EMA9 below EMA21 |
| EMA50 slope | rising (no falling knife) | falling (no short squeeze) |

### Signal Strength Levels

| Strength | Condition |
|----------|-----------|
| `STRONG` | Rule signal + 5m ML model agree on direction |
| `RULE` | Rule signal only (ML neutral or not trained) |
| `AI` | ML signal only at ≥70% confidence + 1h bias confirmation |

### Filters Applied in Order

1. **Market hours** — no signals outside NYSE/XETRA open (crypto 24/7)
2. **Multi-timeframe** — 1h bias must agree with signal direction
3. **EMA200 trend** — no BUY below EMA200, no SELL above EMA200
4. **Sentiment** — strong contradicting news suppresses signal
5. **AI capability gate** — AI-only signals suppressed for symbols with bootstrap WR < 52%
6. **Regime filter** — suppressed during VIX spike / SPY drop / ATR spike
7. **Concurrent limit** — max 3 same-direction signals per scan cycle
8. **Performance brake** — scan frequency halved if 20-trade rolling WR < 42%

---

## ML Models

Each symbol has two independent model pairs:

### 5m Model (rule confirmation)
- Trained on 60 days of 5m candles (~4,660 bars per symbol)
- Labels: bar-by-bar TP/SL simulation — `buy_result=1` when TP hit before SL within 24 bars
- Scaler: fit on training partition only (no calibration-set leakage)
- Used in `bot.py` to confirm 5m rule signals — matches the timeframe being evaluated

### 1h Model (bias context)
- Trained on 60 days of 1h candles (~420 bars)
- Same label logic, same feature set
- Used for directional bias and `get_1h_bias()`

### Features (22 total)

| Category | Features |
|----------|---------|
| Direction | RSI, MACD histogram, Bollinger Band position |
| Volatility regime | ATR%, ATR percentile (90-bar), Garman-Klass vol, vol-of-vol |
| Momentum quality | ROC(10), ROC(20), momentum consistency, momentum acceleration |
| Price structure | Volume ratio, VWAP spread |
| Time cyclical | hour_sin, hour_cos, dow_sin, dow_cos |
| Market context | SPY relative strength (1h/8h), SPY regime, SPY BB position, VIX norm |

---

## Continuous Learning

1. Every signal is stored with its entry price, TP, and SL
2. 24 hours later, `check_outcomes` resolves each signal bar-by-bar (same logic as training labels)
3. When **25+ new outcomes** arrive since the last retrain, both models retrain automatically
4. Real outcomes are blended into training via `sample_weight=5.0` (not row duplication)
5. Performance brake engages if rolling 20-trade WR drops below 42%

---

## Walk-Forward Validation

Symbols are graded A/B/C/D using **5-fold purged + embargo cross-validation**:

- **Purge:** training bars whose label window overlaps the test fold are excluded
- **Embargo:** 10 bars after each test fold are also excluded
- **Labels:** `build_labels_tpsl` — same TP/SL bar-by-bar logic as live training
- **Grade:** based on aggregate win rate across all 5 folds + fold standard deviation

| Grade | Win Rate | Interpretation |
|-------|----------|---------------|
| A | ≥ 58% | Strong edge |
| B | 52–58% | Positive expectancy |
| C | 47–52% | Marginal / monitor |
| D | < 47% | Below breakeven |

---

## Dashboard

Access at `http://your-server:5002`

| Tab | What you see |
|-----|-------------|
| Overview | Live prices, recent signals, open positions |
| Signals | Full signal history with outcomes and accuracy |
| Learning Activity | Training log, job progress, win rate per symbol |
| ML | Walk-forward grades, fold win rates, feature stats |
| Admin | Symbol management, sentiment settings, bootstrap |

---

## Configuration

All runtime settings stored in the database, changeable from Admin panel without restart:

| Setting | Default | Description |
|---------|---------|-------------|
| `sentiment_provider` | `disabled` | `disabled` / `local_finbert` / `gemini` / `claude` |
| `sentiment_suppress_threshold` | `0.35` | Score above this suppresses conflicting signal |
| `news_provider` | `disabled` | `disabled` / `finnhub` |
| `finnhub_api_key` | — | Free key from finnhub.io |
| `gemini_api_key` | — | Google AI Studio (1,500 free calls/day) |
| `outcome_notify_admin` | `false` | Send outcome results to admin DM |

---

## Deployment (Server)

```bash
# On local machine
git push origin main

# On server
git fetch && git reset --hard origin/main && docker compose restart bot
```

---

## Project Structure

```
TradeSignal/
├── bot.py               # Telegram bot, scheduler, outcome checker, continuous learning
│                        # regime filter, performance brake, concurrent limit
├── signals.py           # Technical indicator logic (RSI, MACD, EMA, EMA50 slope guard, MTF)
├── ml_signals.py        # GradientBoosting models per symbol (5m + 1h)
│                        # 22-feature set, TP/SL labels, Platt calibration
├── sentiment.py         # Provider abstraction: FinBERT / Gemini / Claude
├── bootstrap.py         # Historical training + 5-fold purged walk-forward validation
├── backtest.py          # Additional backtesting utilities
├── database.py          # SQLite schema + all DB helpers
├── dashboard.py         # Flask web dashboard
├── lang.py              # Persian / English translations
├── sentiment_service/   # FinBERT Flask microservice (Docker)
└── docker-compose.yml
```

---

## Security

- Never commit `.env` — all secrets are injected via environment variables
- Sensitive settings (API keys) are stored in the database, not in code
- The dashboard has no authentication — run it behind a VPN or reverse proxy

---

## License

MIT
