# TradeSignal 📈

An autonomous trading signal bot that generates, filters, and broadcasts BUY/SELL signals to a Telegram channel. Built with Python, scikit-learn, FinBERT, and Docker.

---

## What It Does

- **Scans** stocks and crypto every 5 minutes using RSI, MACD, and EMA crossovers
- **Filters** signals with a per-symbol ML model (GradientBoosting) trained on 2 years of historical data
- **Suppresses** signals that contradict current news sentiment (via FinBERT or Gemini)
- **Broadcasts** to a Telegram channel in Persian or English, beginner or expert format
- **Tracks outcomes** 24 hours after each signal to measure accuracy
- **Retrains automatically** as real outcomes accumulate — models improve over time

---

## Architecture

```
yfinance (price data)
    │
    ▼
signals.py          ← RSI / MACD / EMA / multi-timeframe filter
    │
    ▼
ml_signals.py       ← GradientBoosting confidence filter (per symbol)
    │
    ▼
sentiment.py        ← FinBERT / Gemini / Claude news sentiment check
    │
    ▼
bot.py              ← Telegram broadcast → @channel
    │
    ▼
check_outcomes      ← 24h later: was the signal correct?
    │
    ▼
continuous_learning ← auto-retrain when 3+ new outcomes arrive
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

This downloads 2 years of price history per symbol and trains the ML models (~10 min).

### 4. Enable sentiment (optional)
In the dashboard Admin panel:
- **Sentiment Provider**: `Local FinBERT` (already running) or `Gemini 1.5 Flash` (free tier)
- **News Provider**: `Finnhub` (free key at [finnhub.io](https://finnhub.io))

---

## Signal Logic

A signal fires when **at least 2 of 3** technical conditions are met:

| Indicator | BUY condition | SELL condition |
|-----------|--------------|----------------|
| RSI (14) | < 35 (oversold) | > 65 (overbought) |
| MACD | crosses above signal line | crosses below signal line |
| EMA | EMA9 crosses above EMA21 | EMA9 crosses below EMA21 |

**Multi-timeframe filter:** 5m signal must agree with 1h trend direction. Disagreements are suppressed.

**ML filter:** GradientBoosting classifier trained on historical outcomes predicts whether the current pattern has >50% probability of being correct. Low-confidence signals are dropped.

**Sentiment filter:** If news sentiment strongly contradicts the signal direction (score > 0.35 threshold), the signal is suppressed.

**Signal quality (1–5 stars):** Computed from number of confirming indicators, AI confidence, and multi-timeframe agreement.

---

## Continuous Learning

The bot is self-improving:

1. Every signal is stored with its entry price, TP, and SL
2. 24 hours later, `check_outcomes` resolves each signal as **correct**, **incorrect**, or **neutral**
3. Every 10 minutes, `continuous_learning` checks if 3+ new outcomes arrived since the last training run
4. If yes, the model retrains immediately using blended historical + real outcome data
5. Models get better as more real signals are observed

---

## Dashboard

Access at `http://your-server:5002`

| Tab | What you see |
|-----|-------------|
| Overview | Live prices, recent signals, open positions |
| Signals | Full signal history with outcomes and accuracy |
| Learning Activity | Training log, job progress, win rate per symbol |
| Backtest | Walk-forward backtest for any symbol |
| Admin | Symbol management, sentiment settings, bootstrap |

---

## Configuration

All runtime settings are stored in the database and can be changed from the Admin panel **without restart**:

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

The project uses a **bind-mount** (`.:/app`), so updates deploy without rebuilding the image:

```bash
git pull
docker restart tradesignal-bot tradesignal-dashboard
```

---

## Project Structure

```
TradeSignal/
├── bot.py               # Telegram bot, scheduler, outcome checker, continuous learning
├── signals.py           # Technical indicator logic (RSI, MACD, EMA, MTF)
├── ml_signals.py        # GradientBoosting model per symbol
├── sentiment.py         # Provider abstraction: FinBERT / Gemini / Claude
├── bootstrap.py         # Historical training data generator
├── backtest.py          # Walk-forward backtesting
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
