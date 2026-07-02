import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from backtest import format_backtest_message, run_backtest
from lang import t, signal_msg, tp_sl_alert, sell_alert as _sell_alert_msg
from database import (
    complete_action_request,
    close_position,
    get_setting,
    set_setting,
    create_action_request,
    get_active_symbols,
    get_all_open_positions,
    get_all_subscriber_ids,
    get_last_signal_action,
    get_latest_signal,
    get_outcome_count,
    get_outcome_training_data,
    get_pending_action_requests,
    get_recent_action_requests,
    get_pending_outcomes,
    get_pending_outcomes_recent,
    get_pending_shadow_signals,
    get_postmortem_stats,
    get_llm_usage_stats,
    get_shadow_stats,
    get_swing_enabled_symbols,
    log_shadow_signal,
    update_shadow_outcome,
    get_stats,
    get_subscribers_for_symbol,
    get_user_open_positions,
    get_user_pnl,
    get_user_pref,
    get_user_subscriptions,
    get_users_with_open_position,
    get_weekly_stats,
    init_db,
    log_learning_cycle,
    log_scan,
    log_signal,
    log_user,
    open_position,
    seed_symbols,
    set_user_pref,
    start_training_job,
    finish_training_job,
    subscribe_user,
    unsubscribe_user,
    update_outcome,
    get_symbol_name,
    update_symbol_name,
)
from ml_signals import StockModel, build_features, get_predict_market_context, _get_sector_ctx
from sentiment import get_sentiment, refresh_cache as refresh_sentiment
from signals import (
    _atr, _macd, _rsi,
    combine_signals,
    fetch_ohlc,
    get_1h_bias,
    get_signal,
    get_swing_signal,
    is_crypto,
    is_market_open,
    market_session,
)

load_dotenv()

# ── Training notification helper ──────────────────────────────────────────────
def _train_summary_line(symbol: str, outcome_samples: int, trigger: str) -> str:
    """One-line summary for a trained symbol with live accuracy."""
    trigger_icon = {"outcomes": "⚡", "time": "⏱", "manual": "🖱", "bootstrap": "🚀"}.get(trigger, "🔄")
    try:
        from database import get_ml_accuracy_stats
        d = get_ml_accuracy_stats()["per_symbol"].get(symbol, {})
        ai_total = d.get("ai_total", 0)
        ai_acc   = d.get("ai_accuracy")
        if ai_total >= 3 and ai_acc is not None:
            acc_str = f" · AI {ai_acc}% ({ai_total} sig)"
        elif ai_total > 0:
            acc_str = f" · {ai_total} sig (accumulating)"
        else:
            acc_str = " · no live outcomes yet"
        blend = f"{outcome_samples} blended" if outcome_samples else "synthetic only"
        return f"`{symbol}` {trigger_icon} {blend}{acc_str}"
    except Exception:
        return f"`{symbol}` {trigger_icon} {outcome_samples} outcomes blended"


# ── Currency helpers ───────────────────────────────────────────────────────────
_CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "CHF": "Fr"}
_fx_cache: dict = {}  # {currency: (rate, timestamp)}
_FX_CACHE_TTL = 300   # seconds

def _get_fx_rate(currency: str) -> float:
    """Return the USD→currency conversion rate, cached for 5 minutes."""
    if currency == "USD":
        return 1.0
    import time
    cached = _fx_cache.get(currency)
    if cached and time.time() - cached[1] < _FX_CACHE_TTL:
        return cached[0]
    try:
        ticker = yf.Ticker(f"USD{currency}=X")
        rate = ticker.fast_info.last_price
        if rate and rate > 0:
            _fx_cache[currency] = (rate, time.time())
            return rate
    except Exception:
        pass
    return 1.0

def _get_display_currency() -> tuple[str, str, float]:
    """Return (currency_code, symbol, fx_rate) from settings."""
    currency = get_setting("display_currency", "USD").upper()
    symbol = _CURRENCY_SYMBOLS.get(currency, currency + " ")
    rate = _get_fx_rate(currency)
    return currency, symbol, rate

# yfinance quotes each ticker in its exchange's native currency. A blanket
# USD→display rate double-converts European stocks (their prices are already
# EUR), so conversion must be native→display per symbol.
_NATIVE_CCY_SUFFIX = {
    ".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".MI": "EUR", ".AS": "EUR",
    ".MC": "EUR", ".BR": "EUR", ".LS": "EUR", ".VI": "EUR", ".HE": "EUR",
    ".SW": "CHF", ".CO": "DKK", ".ST": "SEK", ".OL": "NOK", ".L": "GBP",
}

def _native_currency(symbol: str) -> str:
    s = symbol.upper()
    for sfx, ccy in _NATIVE_CCY_SUFFIX.items():
        if s.endswith(sfx):
            return ccy
    return "USD"  # US stocks and crypto pairs quote in USD

def _symbol_fx(symbol: str) -> tuple[str, float]:
    """Return (display currency sign, native→display conversion rate)."""
    display = get_setting("display_currency", "USD").upper()
    cs = _CURRENCY_SYMBOLS.get(display, display + " ")
    native = _native_currency(symbol)
    if native == display:
        return cs, 1.0
    usd_to_native = _get_fx_rate(native)
    if usd_to_native <= 0:
        return cs, 1.0
    return cs, _get_fx_rate(display) / usd_to_native


BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHANNEL_ID = os.getenv("CHANNEL_ID", "")
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "")
SYMBOLS = [s.strip() for s in os.getenv(
    "SYMBOLS", "AAPL,TSLA,NVDA,MSFT,BTC-USD,ETH-USD"
).split(",")]
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_MINUTES", "5")) * 60
OUTCOME_CHECK_HOURS = int(os.getenv("OUTCOME_CHECK_HOURS", "24"))
OUTCOME_CHECK_INTERVAL = int(os.getenv("OUTCOME_CHECK_INTERVAL_MINUTES", "10")) * 60
LEARNING_CHECK_INTERVAL = int(os.getenv("LEARNING_CHECK_INTERVAL_MINUTES", "10")) * 60
ACTION_QUEUE_INTERVAL = int(os.getenv("ACTION_QUEUE_INTERVAL_SECONDS", "60"))
ENABLE_SUBSCRIBER_DM_SIGNALS = os.getenv("ENABLE_SUBSCRIBER_DM_SIGNALS", "false").lower() == "true"

ET  = ZoneInfo("America/New_York")
CET = ZoneInfo("Europe/Berlin")

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
# httpx logs every getUpdates long-poll at INFO (~1 line/10s, 100K+ lines/week)
# — that flood is what made `docker logs` appear silent. Warnings still surface.
logging.getLogger("httpx").setLevel(logging.WARNING)

models: dict[str, StockModel] = {s: StockModel(s) for s in SYMBOLS}
BOT_USERNAME = ""


def _get_symbols() -> list[str]:
    """Active symbol list from DB; falls back to env SYMBOLS if DB is empty."""
    db_syms = get_active_symbols()
    return db_syms if db_syms else SYMBOLS


async def refresh_symbols(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sync models dict with the current active symbols from the DB."""
    current = set(_get_symbols())
    existing = set(models.keys())
    added = current - existing
    removed = existing - current
    for sym in added:
        models[sym] = StockModel(sym)
        logger.info(f"[symbols] Added model for {sym}")
    for sym in removed:
        del models[sym]
        logger.info(f"[symbols] Removed model for {sym}")


# ── Per-symbol AI capability cache ───────────────────────────────────────────
import time as _time
_ai_capable_cache: dict = {"ts": 0.0, "data": {}}
_AI_CAPABLE_TTL = 3600  # re-read bootstrap rates once per hour
_AI_MIN_BOOTSTRAP_WR = 52.0  # disable AI-only for symbols below this bootstrap win rate

# ── Performance brake ─────────────────────────────────────────────────────────
# Track last 20 signal outcomes. If win rate drops below 42% for 20 consecutive
# resolved signals, halve the scan frequency (fire every other scan cycle).
_PERF_BRAKE_WINDOW    = 20
_PERF_BRAKE_THRESHOLD = 0.42   # below this → engage brake (normal)
_PERF_BRAKE_DD_THRESHOLD = 0.35  # tightened threshold when drawdown > 5%
_PERF_BRAKE_DD_LIMIT  = 5.0   # drawdown % that triggers the tighter brake
_perf_history: list[bool] = []  # True = correct, False = incorrect
_perf_brake_engaged       = False
_perf_scan_counter        = 0   # counts scan_and_alert() calls; odd calls skipped when brake on

# ── Concurrent signal limit ───────────────────────────────────────────────────
# Prevent firing more than 3 signals in the same direction per scan cycle.
# Correlated positions (e.g. all tech stocks BUY at open) amplify drawdowns.
_MAX_SAME_DIRECTION_PER_SCAN = 3

# ── Sentiment history for momentum computation ────────────────────────────────
# Tracks (unix_timestamp, score) per symbol to compute 3h sentiment momentum.
_sentiment_history: dict[str, list[tuple[float, float]]] = {}
_SENTIMENT_MOMENTUM_WINDOW = 10800  # 3 hours in seconds


# ── Earnings guard ────────────────────────────────────────────────────────────
# Earnings are binary-event risk no technical indicator can see coming.
# Suppressed signals are logged as shadow signals (source='earnings_guard') so
# the guard's value is measured, not assumed: if the hypothetical win rate of
# suppressed signals turns out HIGHER than live, the guard is hurting and
# should be removed (see docs/llm-evaluation.md).
_earnings_cache: dict[str, tuple[float, "object"]] = {}  # symbol -> (fetched_at, next_date|None)
_EARNINGS_CACHE_TTL = 24 * 3600
_EARNINGS_WINDOW_DAYS = 3


def _near_earnings(symbol: str) -> bool:
    """True when the symbol reports earnings within the guard window.
    Missing/unavailable earnings data never blocks a signal."""
    if is_crypto(symbol):
        return False
    now = _time.time()
    cached = _earnings_cache.get(symbol)
    if cached and now - cached[0] < _EARNINGS_CACHE_TTL:
        next_date = cached[1]
    else:
        next_date = None
        try:
            dates = yf.Ticker(symbol).get_earnings_dates(limit=8)
            if dates is not None and not dates.empty:
                today = datetime.now(timezone.utc).date()
                future = [d.date() for d in dates.index.to_pydatetime()
                          if d.date() >= today]
                next_date = min(future) if future else None
        except Exception as e:
            logger.debug(f"[earnings_guard] no earnings data for {symbol}: {e}")
        _earnings_cache[symbol] = (now, next_date)
    if next_date is None:
        return False
    days_until = (next_date - datetime.now(timezone.utc).date()).days
    return 0 <= days_until <= _EARNINGS_WINDOW_DAYS


_AI_MIN_HOLDOUT_AUC = 0.60  # AI-only allowed when holdout buy-AUC demonstrates skill


def _is_ai_capable(symbol: str) -> bool:
    """
    Gate AI-only signals on demonstrated out-of-sample skill.
    Primary: holdout buy-AUC from the last train (>= 0.60 = real signal;
    0.5 = coin flip). Falls back to bootstrap win rate for models trained
    before AUC tracking existed.
    """
    model = models.get(symbol)
    auc = getattr(model, "val_auc_buy", None) if model else None
    if auc is not None:
        return auc >= _AI_MIN_HOLDOUT_AUC
    now = _time.time()
    if now - _ai_capable_cache["ts"] > _AI_CAPABLE_TTL:
        try:
            from database import get_ml_accuracy_stats
            stats = get_ml_accuracy_stats()
            _ai_capable_cache["data"] = {
                sym: (d.get("bootstrap_win_rate") or 0.0)
                for sym, d in stats["per_symbol"].items()
            }
            _ai_capable_cache["ts"] = now
        except Exception:
            pass
    wr = _ai_capable_cache["data"].get(symbol, 100.0)  # unknown = allow
    return wr >= _AI_MIN_BOOTSTRAP_WR or wr == 0.0  # 0.0 = never bootstrapped = allow


# ── Regime filter ─────────────────────────────────────────────────────────────

def get_market_regime(spy_df: "pd.DataFrame | None" = None,
                      vix_df: "pd.DataFrame | None" = None) -> str:
    """
    3-state market regime: 'bull' | 'bear' | 'volatile'.
    bull:     SPY above EMA50 AND VIX < 20
    volatile: VIX ≥ 20 OR VIX 2-day spike > 20%
    bear:     SPY below EMA50 (while not volatile)
    """
    try:
        if spy_df is None or vix_df is None:
            spy_df, vix_df = get_predict_market_context()
    except Exception:
        pass
    if spy_df is None and vix_df is None:
        return "unknown"
    spy_above_ema = None  # None = unknown, treated as neutral (no bull bias)
    try:
        if spy_df is not None:
            spy_c = spy_df["Close"].squeeze()
            spy_ema50 = spy_c.ewm(span=50, adjust=False).mean()
            spy_above_ema = float(spy_c.iloc[-1]) >= float(spy_ema50.iloc[-1])
    except Exception:
        pass

    try:
        vix_c = vix_df["Close"].squeeze()
        current_vix = float(vix_c.iloc[-1])
        prior_vix   = float(vix_c.iloc[-3]) if len(vix_c) >= 3 else current_vix
        vix_high    = current_vix >= 20
        vix_spike   = prior_vix > 0 and (current_vix - prior_vix) / prior_vix > 0.20
    except Exception:
        current_vix = 15.0
        vix_high = False
        vix_spike = False

    if vix_high or vix_spike:
        return "volatile"
    if spy_above_ema is True:
        return "bull"
    if spy_above_ema is False:
        return "bear"
    return "unknown"  # spy_df was unavailable but vix was fine (rare)


def _is_hostile_regime(spy_df: "pd.DataFrame | None", vix_df: "pd.DataFrame | None",
                       symbol_df_1h: "pd.DataFrame | None") -> tuple[bool, str]:
    """
    Returns (hostile: bool, reason: str).
    Three independent conditions — any one triggers suppression:
      1. VIX 2-day spike >30% while VIX >20 (fear spike)
      2. SPY down >1.5% in last 8 hourly bars (intraday market sweep)
      3. Symbol's 5-day ATR > 2× its 90-day ATR (abnormal volatility)
    """
    try:
        if vix_df is not None and not vix_df.empty:
            vix_c = vix_df["Close"].squeeze()
            if len(vix_c) >= 3:
                current_vix = float(vix_c.iloc[-1])
                prior_vix = float(vix_c.iloc[-3])
                if prior_vix > 0 and current_vix > 20:
                    if (current_vix - prior_vix) / prior_vix > 0.30:
                        return True, f"VIX spike {current_vix:.1f} (+{(current_vix-prior_vix)/prior_vix*100:.0f}%)"
    except Exception:
        pass

    try:
        if spy_df is not None and not spy_df.empty:
            spy_c = spy_df["Close"].squeeze()
            if len(spy_c) >= 9:
                spy_now = float(spy_c.iloc[-1])
                spy_8h  = float(spy_c.iloc[-9])
                if spy_8h > 0 and (spy_now - spy_8h) / spy_8h < -0.015:
                    return True, f"SPY 8h drop {(spy_now-spy_8h)/spy_8h*100:.1f}%"
    except Exception:
        pass

    try:
        if symbol_df_1h is not None and not symbol_df_1h.empty and len(symbol_df_1h) >= 90 * 6:
            from signals import _atr
            atr_series = _atr(symbol_df_1h)
            if not atr_series.empty and len(atr_series) >= 90 * 6:
                atr_5d  = float(atr_series.iloc[-5*6:].mean())
                atr_90d = float(atr_series.iloc[-90*6:].mean())
                if atr_90d > 0 and atr_5d > 2.0 * atr_90d:
                    return True, f"ATR spike {atr_5d/atr_90d:.1f}× normal"
    except Exception:
        pass

    return False, ""


# ── Company name helpers ──────────────────────────────────────────────────────

_name_cache: dict[str, str] = {}  # symbol → company name, in-process cache


def _get_company_name(symbol: str) -> str | None:
    if symbol in _name_cache:
        return _name_cache[symbol]
    name = get_symbol_name(symbol)
    if name:
        _name_cache[symbol] = name
    return name


def _populate_symbol_names() -> None:
    """Fetch company names from yfinance for symbols missing them. Runs once at startup."""
    import threading
    def _run():
        for sym in get_active_symbols():
            if get_symbol_name(sym):
                continue
            try:
                info = yf.Ticker(sym).info
                name = (info.get("shortName") or info.get("longName") or "").strip()
                # Strip exchange suffixes yfinance appends (e.g. "Eni S.p.A.")
                if name:
                    update_symbol_name(sym, name)
                    _name_cache[sym] = name
                    logger.debug(f"[names] {sym} → {name}")
            except Exception as e:
                logger.debug(f"[names] failed for {sym}: {e}")
    threading.Thread(target=_run, daemon=True, name="symbol_names").start()


# ── ML helpers ────────────────────────────────────────────────────────────────

def _current_features(df, symbol: str | None = None) -> dict:
    try:
        spy_df, vix_df = get_predict_market_context()
        sector_df = _get_sector_ctx(symbol) if symbol else None
        return build_features(df, spy_df=spy_df, vix_df=vix_df, sector_df=sector_df).dropna().iloc[-1].to_dict()
    except Exception:
        return {}


def _resolve_signal_outcome(sig: dict) -> dict | None:
    """
    Resolve a signal using post-entry candle ranges instead of only the latest price.
    Returns None when the signal should remain pending.
    """
    sent_at = datetime.fromisoformat(sig["sent_at"])
    if sent_at.tzinfo is None:
        sent_at = sent_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    # Swing signals hold for days: check TP/SL touches on 1h candles and only
    # time out after ~10 trading days (14 calendar). Intraday keeps 24h/5m.
    is_swing = sig.get("mode") == "swing"
    if is_swing:
        interval = "1h"
    else:
        interval = "1h" if is_crypto(sig["symbol"]) else "5m"
    pad = timedelta(hours=2) if interval == "1h" else timedelta(minutes=30)
    end = now + timedelta(minutes=5)

    # Hardened fetch: outcome labels feed model training — a transient yfinance
    # failure must delay resolution (return None → stay pending), never mislabel.
    df = fetch_ohlc(
        sig["symbol"],
        start=sent_at - pad,
        end=end,
        interval=interval,
    )
    if df is None or df.empty:
        return None

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[df.index >= sent_at]
    if df.empty:
        return None

    current_price = float(df["Close"].iloc[-1])
    tp = sig.get("tp")
    sl = sig.get("sl")
    age_hours = (now - sent_at).total_seconds() / 3600
    entry = float(sig["price"])

    highs = df["High"].astype(float)
    lows = df["Low"].astype(float)
    if sig["action"] == "BUY":
        max_favorable_pct = ((float(highs.max()) - entry) / entry) * 100
        max_adverse_pct = ((float(lows.min()) - entry) / entry) * 100
    else:
        max_favorable_pct = ((entry - float(lows.min())) / entry) * 100
        max_adverse_pct = ((entry - float(highs.max())) / entry) * 100

    def _meta(resolution_time: datetime | None,
              resolution_reason: str,
              touched_tp: bool = False,
              touched_sl: bool = False) -> dict:
        minutes = None
        if resolution_time is not None:
            minutes = round((resolution_time - sent_at).total_seconds() / 60, 1)
        return {
            "resolution_reason": resolution_reason,
            "touched_tp": touched_tp,
            "touched_sl": touched_sl,
            "resolution_minutes": minutes,
            "max_favorable_pct": round(max_favorable_pct, 2),
            "max_adverse_pct": round(max_adverse_pct, 2),
        }

    if tp and sl:
        for ts, row in df.iterrows():
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            if sig["action"] == "BUY":
                hit_tp = high >= tp
                hit_sl = low <= sl
            else:
                hit_tp = low <= tp
                hit_sl = high >= sl

            if hit_tp and hit_sl:
                return {
                    "outcome": "neutral",
                    "price": close,
                    "current_price": current_price,
                    "metadata": _meta(ts, "both_hit_same_candle", touched_tp=True, touched_sl=True),
                }
            if hit_tp:
                return {
                    "outcome": "correct",
                    "price": float(tp),
                    "current_price": current_price,
                    "metadata": _meta(ts, "tp_hit", touched_tp=True),
                }
            if hit_sl:
                return {
                    "outcome": "incorrect",
                    "price": float(sl),
                    "current_price": current_price,
                    "metadata": _meta(ts, "sl_hit", touched_sl=True),
                }

        horizon_hours = 14 * 24 if is_swing else OUTCOME_CHECK_HOURS
        if age_hours < horizon_hours:
            return None
        return {
            "outcome": "neutral",
            "price": current_price,
            "current_price": current_price,
            "metadata": _meta(now, "timeout_no_hit"),
        }

    threshold = 0.005
    if sig["action"] == "BUY":
        outcome = ("correct" if current_price > entry * (1 + threshold)
                   else "incorrect" if current_price < entry * (1 - threshold)
                   else "neutral")
        reason = "threshold_up" if outcome == "correct" else "threshold_down" if outcome == "incorrect" else "threshold_flat"
    else:
        outcome = ("correct" if current_price < entry * (1 - threshold)
                   else "incorrect" if current_price > entry * (1 + threshold)
                   else "neutral")
        reason = "threshold_down" if outcome == "correct" else "threshold_up" if outcome == "incorrect" else "threshold_flat"
    return {
        "outcome": outcome,
        "price": current_price,
        "current_price": current_price,
        "metadata": _meta(now, reason),
    }


async def _ensure_models_trained(bot=None) -> None:
    trained_lines = []
    for symbol, model in models.items():
        current_count = get_outcome_count(symbol)
        if model.needs_retrain(current_outcome_count=current_count):
            outcome_data = get_outcome_training_data(symbol=symbol)
            ok = model.train_gated(outcome_data=outcome_data)
            if ok:
                blended = getattr(model, "last_blend_count", 0)
                trigger = "outcomes" if blended >= 3 else "time"
                line = _train_summary_line(symbol, blended, trigger)
                if getattr(model, "last_promotion", "") == "rejected":
                    line += " · ⛔ challenger rejected, champion kept"
                trained_lines.append(line)
    if trained_lines and bot and ADMIN_CHAT_ID:
        try:
            await bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="🧠 *Models retrained*\n\n" + "\n".join(trained_lines),
                parse_mode="Markdown",
            )
        except Exception:
            pass


# ── Outcome checker ───────────────────────────────────────────────────────────

async def check_outcomes(context: ContextTypes.DEFAULT_TYPE) -> None:
    # Combine long-window (configured hours) and short-window (1h+) signals
    long_window = get_pending_outcomes(older_than_hours=OUTCOME_CHECK_HOURS)
    recent = get_pending_outcomes_recent(min_age_hours=1.0)
    # Deduplicate by id
    seen = set()
    pending = []
    for sig in long_window + recent:
        if sig["id"] not in seen:
            seen.add(sig["id"])
            pending.append(sig)
    if not pending:
        return
    logger.info(f"Checking outcomes for {len(pending)} signals...")
    for sig in pending:
        try:
            resolved = _resolve_signal_outcome(sig)
            if resolved is None:
                continue
            current_price = resolved["current_price"]
            tp = sig.get("tp")
            sl = sig.get("sl")
            outcome = resolved["outcome"]

            update_outcome(sig["id"], outcome, resolved["price"], metadata=resolved.get("metadata"))

            # Feed outcome into performance brake tracker
            if outcome in ("correct", "incorrect"):
                _update_perf_brake(outcome == "correct")

                # LLM post-mortem tag (no-op without an API key); threaded so
                # the HTTP call never blocks the outcome loop
                try:
                    import asyncio
                    from postmortem import run_postmortem
                    await asyncio.to_thread(run_postmortem, sig, outcome, resolved)
                except Exception as e:
                    logger.debug(f"[postmortem] skipped: {e}")

            entry = sig["price"]
            pct = (current_price - entry) / entry * 100
            sign = "+" if pct >= 0 else ""
            emoji = {"correct": "✅", "incorrect": "❌", "neutral": "➖"}.get(outcome, "❓")

            if ADMIN_CHAT_ID and get_setting("outcome_notify_admin", "false") == "true":
                cs, fxr = _symbol_fx(sig["symbol"])
                tp_text = (f"\nTP: `{cs}{tp * fxr:.2f}` | SL: `{cs}{sig.get('sl', 0) * fxr:.2f}`"
                           if tp else "")
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=(
                        f"{emoji} *Outcome — {sig['symbol']} {sig['action']}*\n"
                        f"Entry: `{cs}{entry * fxr:.2f}` → Now: `{cs}{current_price * fxr:.2f}` ({sign}{pct:.2f}%)"
                        f"{tp_text}\n"
                        f"Result: *{outcome.upper()}*"
                    ),
                    parse_mode="Markdown",
                )
        except Exception as e:
            logger.error(f"Outcome check error signal {sig['id']}: {e}")

    # Resolve hypothetical outcomes for shadow-suppressed AI-only signals.
    # Same resolver, same TP/SL-touch semantics — no notifications, just data.
    try:
        shadow_pending = get_pending_shadow_signals(min_age_hours=1.0)
        n_resolved = 0
        for srow in shadow_pending:
            sig_like = dict(srow)
            sig_like["sent_at"] = srow["created_at"]
            resolved = _resolve_signal_outcome(sig_like)
            if resolved is None:
                continue
            update_shadow_outcome(
                srow["id"], resolved["outcome"], resolved["price"],
                resolution_reason=resolved.get("metadata", {}).get("resolution_reason"),
            )
            n_resolved += 1
        if n_resolved:
            logger.info(f"[shadow] resolved {n_resolved} hypothetical outcomes")
    except Exception as e:
        logger.error(f"[shadow] outcome resolution failed: {e}")


# ── Continuous learning ───────────────────────────────────────────────────────

async def refresh_sentiment_cache(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Refresh news sentiment for all symbols every 30 min. Never blocks scan."""
    if get_setting("sentiment_provider", "disabled") == "disabled":
        return
    symbols = _get_symbols()
    for symbol in symbols:
        try:
            refresh_sentiment(symbol)
        except Exception as e:
            logger.debug(f"[sentiment] refresh failed for {symbol}: {e}")


async def ml_health_report(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Weekly ML health report to the admin: live-outcome progress toward the
    blend threshold, holdout-AUC ranking, shadow-mode activity, 7d win rate.
    Surfaces the evidence that otherwise only lives in logs."""
    if not ADMIN_CHAT_ID:
        return
    try:
        from ml_signals import MIN_LIVE_BLEND

        # Outcome progress toward the live-blend threshold
        counts = {sym: get_outcome_count(sym) for sym in get_active_symbols()}
        with_outcomes = sorted(
            ((s, c) for s, c in counts.items() if c > 0),
            key=lambda x: -x[1],
        )
        total_outcomes = sum(counts.values())
        progress_lines = [
            f"  `{s}`: {c}/{MIN_LIVE_BLEND}" for s, c in with_outcomes[:8]
        ] or ["  none yet"]

        # Holdout AUC ranking (skill evidence per symbol)
        aucs = sorted(
            ((s, m.val_auc_buy) for s, m in models.items()
             if getattr(m, "val_auc_buy", None) is not None),
            key=lambda x: -x[1],
        )
        auc_top = [f"  🟢 `{s}` {a:.2f}" for s, a in aucs[:5]]
        auc_bottom = [f"  🔴 `{s}` {a:.2f}" for s, a in aucs[-3:]] if len(aucs) > 5 else []
        n_skilled = sum(1 for _, a in aucs if a >= _AI_MIN_HOLDOUT_AUC)

        # Shadow-mode hypothetical performance: would the suppressed AI-only
        # signals have won? This is the evidence for lifting shadow mode.
        sh = get_shadow_stats()
        if sh["resolved"] > 0:
            shadow_line = (
                f"`{sh['week']}` suppressed this week · hypothetical record "
                f"`{sh['correct']}W/{sh['incorrect']}L` (`{sh['win_rate']}%`) "
                f"· `{sh['pending']}` pending"
            )
        else:
            shadow_line = (f"`{sh['week']}` suppressed this week · "
                           f"`{sh['pending']}` awaiting hypothetical resolution")

        wk = get_weekly_stats()
        wr_line = (f"`{wk['accuracy']}%` ({wk['correct']}/{wk['resolved']})"
                   if wk["resolved"] > 0 else "no resolved signals")

        # Earnings guard: hypothetical record of what it suppressed. If the
        # suppressed win rate beats live, the guard is hurting — remove it.
        eg = get_shadow_stats(source="earnings_guard")
        if eg["resolved"] > 0:
            eg_line = (f"`{eg['total']}` suppressed · hypothetical "
                       f"`{eg['correct']}W/{eg['incorrect']}L` (`{eg['win_rate']}%`)")
        else:
            eg_line = f"`{eg['total']}` suppressed · `{eg['pending']}` pending resolution"

        # Post-mortem tags: failure clusters become filters at n>=30
        pm = get_postmortem_stats()
        if pm["total"] > 0:
            top_loss = sorted(pm["loss_causes"].items(), key=lambda x: -x[1])[:3]
            pm_line = (f"`{pm['total']}` tagged · top loss causes: "
                       + ", ".join(f"{c} ({n})" for c, n in top_loss))
        else:
            pm_line = "none yet (needs LLM API key)"

        usage = get_llm_usage_stats(days=7)
        if usage:
            u_line = " · ".join(
                f"{t}: {v['ok']}/{v['calls']} ok, {(v['input_tokens'] + v['output_tokens']) / 1000:.0f}K tok"
                for t, v in usage.items())
        else:
            u_line = "no LLM calls this week"

        msg = (
            "🩺 *Weekly ML Health Report*\n\n"
            f"*Live outcomes:* `{total_outcomes}` total — blend gate opens at "
            f"`{MIN_LIVE_BLEND}`/symbol/direction\n"
            + "\n".join(progress_lines) + "\n\n"
            f"*Holdout AUC* ({n_skilled}/{len(aucs)} symbols ≥ {_AI_MIN_HOLDOUT_AUC}):\n"
            + "\n".join(auc_top + auc_bottom) + "\n\n"
            f"*Shadow mode:* {shadow_line}\n"
            f"*Earnings guard:* {eg_line}\n"
            f"*Post-mortems:* {pm_line}\n"
            f"*LLM usage (7d):* {u_line}\n"
            f"*7d win rate:* {wr_line}"
        )

        # Optional AI analyst commentary — one reasoning pass over the same
        # numbers, human-in-the-loop by design (recommendations only; any
        # strategy change still goes through the backtest gate)
        try:
            import asyncio
            import json as _json
            from llm import llm_complete
            analyst_input = _json.dumps({
                "total_live_outcomes": total_outcomes,
                "blend_gate": MIN_LIVE_BLEND,
                "outcome_progress": dict(with_outcomes[:10]),
                "holdout_auc_top": aucs[:5], "holdout_auc_bottom": aucs[-3:],
                "shadow_ai": sh, "earnings_guard": eg,
                "postmortems": pm, "week_stats": wk,
            }, default=str)
            analyst = await asyncio.to_thread(
                llm_complete, "analyst",
                f"Weekly stats for a rule-based trading signal bot with an "
                f"ML layer in shadow mode:\n{analyst_input}",
                "You are a cautious quantitative analyst reviewing one week of "
                "data from a small trading-signal system. Sample sizes are tiny "
                "— never draw confident conclusions from n<30. Write max 5 "
                "short lines, plain text without any markdown: (1) the single "
                "most important observation, (2) any emerging pattern worth "
                "watching, (3) at most one concrete experiment to consider, "
                "which must be backtestable. If the data supports no "
                "conclusion, say exactly that.",
                False, 400,
            )
            if analyst:
                safe = str(analyst).replace("*", "").replace("_", "").replace("`", "")[:900]
                msg += f"\n\n🧠 *Analyst notes:*\n{safe}"
        except Exception as e:
            logger.debug(f"[ml_health] analyst section skipped: {e}")

        await context.bot.send_message(
            chat_id=ADMIN_CHAT_ID, text=msg, parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"[ml_health] report failed: {e}")


async def weekly_pooled_retrain(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Weekly refresh of the pooled cross-symbol 5m model (Sunday, market
    closed). Runs in a thread; predict() picks up the new pickle via mtime."""
    try:
        import asyncio
        from ml_signals import train_pooled_5m
        symbols = get_active_symbols()
        logger.info(f"[pooled] weekly retrain starting ({len(symbols)} symbols)")
        payload = await asyncio.to_thread(train_pooled_5m, symbols)
        if payload and ADMIN_CHAT_ID:
            per = payload["per_symbol"]
            n_use = sum(1 for v in per.values() if v["use_pooled"])
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=(f"🧬 *Pooled 5m model retrained*\n"
                      f"`{payload['n_samples']:,}` samples · active for "
                      f"`{n_use}/{len(per)}` symbols"),
                parse_mode="Markdown",
            )
    except Exception as e:
        logger.error(f"[pooled] weekly retrain failed: {e}")


async def nightly_db_backup(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Daily online backup of trading.db (the labeled outcome data is the one
    thing this system can't regenerate). Keeps the newest 7 in data/backups/."""
    try:
        from database import backup_db
        path = backup_db(keep=7)
        logger.info(f"[backup] DB backed up to {path}")
    except Exception as e:
        logger.error(f"[backup] nightly DB backup failed: {e}")
        if ADMIN_CHAT_ID:
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=f"⚠️ Nightly DB backup failed: {e}",
                )
            except Exception:
                pass


async def drift_detection_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Nightly PSI drift check. Compares current 5m feature distributions against
    training baselines stored in each model. Logs a WARNING for features with
    PSI > 0.20 (significant distribution shift) and flags the symbol for retraining
    attention. Does not retrain automatically — drift is an alert, not a trigger.
    """
    symbols = _get_symbols()
    spy_df, vix_df = get_predict_market_context()
    for symbol in symbols:
        model = models.get(symbol)
        if model is None or not model.trained:
            continue
        try:
            df = yf.download(symbol, period="2d", interval="5m",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) < 50:
                continue
            result = model.compute_psi(df, spy_df=spy_df, vix_df=vix_df)
            drifted = result.get("drifted", [])
            if drifted:
                psi_str = ", ".join(
                    f"{f}={result['psi'][f]:.3f}" for f in drifted
                )
                logger.warning(
                    f"[drift] {symbol}: significant feature drift detected — {psi_str}"
                )
            else:
                logger.debug(f"[drift] {symbol}: no significant drift")
        except Exception as e:
            logger.debug(f"[drift] {symbol} check failed: {e}")


async def continuous_learning(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Runs every 20 min. For each active symbol, checks whether new resolved
    outcomes have accumulated since the last training run. If so, retrains
    immediately rather than waiting for the 6-hour fallback timer.
    """
    symbols = _get_symbols()
    retrained = []
    for symbol in symbols:
        model = models.get(symbol)
        if model is None:
            continue
        try:
            current_count = get_outcome_count(symbol)
            new_outcomes = current_count - getattr(model, "outcome_count_at_train", 0)
            needs_retrain = model.needs_retrain(current_outcome_count=current_count)
            if not needs_retrain:
                log_learning_cycle(
                    symbol,
                    retrain_needed=False,
                    retrained=False,
                    resolved_outcomes=current_count,
                    new_outcomes=max(0, new_outcomes),
                    note="check_only",
                )
                continue

            outcome_data = get_outcome_training_data(symbol=symbol)
            ok = model.train_gated(outcome_data or None)
            trigger = "outcomes" if new_outcomes >= 3 else "time"
            promo = getattr(model, "last_promotion", "")
            log_learning_cycle(
                symbol,
                retrain_needed=True,
                retrained=ok,
                resolved_outcomes=current_count,
                new_outcomes=max(0, new_outcomes),
                note=("challenger_rejected" if promo == "rejected"
                      else "retrained" if ok else "retrain_failed"),
            )
            if ok:
                retrained.append((symbol, getattr(model, "last_blend_count", 0), trigger))
                logger.info(
                    f"[ML] Retrained {symbol} — outcomes since last train: "
                    f"{current_count - model.outcome_count_at_train}"
                )
        except Exception as e:
            log_learning_cycle(
                symbol,
                retrain_needed=False,
                retrained=False,
                resolved_outcomes=0,
                new_outcomes=0,
                note=f"error:{type(e).__name__}",
            )
            logger.error(f"[ML] continuous_learning error for {symbol}: {e}")

    if retrained and ADMIN_CHAT_ID:
        try:
            lines = [_train_summary_line(sym, n, trig) for sym, n, trig in retrained]
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text="🧠 *Auto-retrain complete*\n\n" + "\n".join(lines),
                parse_mode="Markdown",
            )
        except Exception:
            pass


async def process_action_requests(context: ContextTypes.DEFAULT_TYPE) -> None:
    pending = get_pending_action_requests()
    if not pending:
        return
    for req in pending:
        action = req["action"]
        symbol = req.get("symbol")
        try:
            if action == "scan_now":
                await scan_and_alert(context)
                note = "manual scan completed"
            elif action == "check_outcomes_now":
                await check_outcomes(context)
                note = "manual outcome check completed"
            elif action == "retrain_all":
                # trained=False is the only flag needs_retrain() honors
                # unconditionally (trained_at is ignored below blend threshold)
                for model in models.values():
                    model.trained = False
                await _ensure_models_trained(context.bot)
                note = "all models retrained"
            elif action == "retrain_symbol":
                if not symbol:
                    raise ValueError("symbol required")
                model = models.get(symbol)
                if model is None:
                    raise ValueError(f"unknown symbol {symbol}")
                ok = model.train_gated(get_outcome_training_data(symbol=symbol) or None)
                if not ok:
                    raise RuntimeError(f"{symbol} retrain failed")
                promo = getattr(model, "last_promotion", "")
                note = (f"{symbol} retrained — challenger rejected, champion kept"
                        if promo == "rejected" else f"{symbol} retrained")
            elif action == "run_bootstrap":
                # Guard: refuse if a bootstrap thread is already alive
                if any(t.name == "bootstrap" for t in threading.enumerate()):
                    note = "bootstrap already running — wait for it to finish"
                else:
                    symbols_to_boot = [symbol] if symbol else _get_symbols()
                    if not symbols_to_boot:
                        raise ValueError("no symbols to bootstrap")
                    job_id = start_training_job("bootstrap", total_symbols=len(symbols_to_boot))

                    def _run(jid=job_id, syms=symbols_to_boot):
                        try:
                            from bootstrap import run_bootstrap
                            run_bootstrap(syms, years=2, job_id=jid)
                        except Exception as exc:
                            finish_training_job(jid, status="failed", note=str(exc)[:200])
                            logger.error(f"[bootstrap] thread error: {exc}", exc_info=True)

                    threading.Thread(target=_run, daemon=True, name="bootstrap").start()
                    note = f"bootstrap started in background (job_id={job_id})"
            elif action == "run_walk_forward":
                if any(t.name == "walk_forward" for t in threading.enumerate()):
                    note = "walk-forward test already running — wait for it to finish"
                else:
                    symbols_wf = [symbol] if symbol else _get_symbols()
                    if not symbols_wf:
                        raise ValueError("no symbols for walk-forward test")
                    job_id = start_training_job("walk_forward", total_symbols=len(symbols_wf))

                    def _run_wf(jid=job_id, syms=symbols_wf):
                        try:
                            from bootstrap import run_walk_forward
                            run_walk_forward(syms, job_id=jid)
                        except Exception as exc:
                            finish_training_job(jid, status="failed", note=str(exc)[:200])
                            logger.error(f"[walk_forward] thread error: {exc}", exc_info=True)

                    threading.Thread(target=_run_wf, daemon=True, name="walk_forward").start()
                    note = f"walk-forward test started in background (job_id={job_id})"
            else:
                raise ValueError(f"unsupported action {action}")
            complete_action_request(req["id"], "completed", note)
        except Exception as e:
            logger.error(f"Action request {req['id']} failed: {e}")
            complete_action_request(req["id"], "failed", str(e)[:200])


# ── Live TP/SL monitor ────────────────────────────────────────────────────────

async def monitor_positions(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check all open tracked positions against current price; alert on TP/SL hit."""
    positions = get_all_open_positions()
    if not positions:
        return

    # Batch price fetches per symbol
    symbols_needed = list({p["symbol"] for p in positions
                           if p.get("tp") or p.get("sl")})
    prices: dict[str, float] = {}
    for sym in symbols_needed:
        try:
            df = yf.download(sym, period="1d", interval="5m", progress=False, auto_adjust=True)
            if not df.empty:
                prices[sym] = float(df["Close"].iloc[-1])
        except Exception as e:
            logger.warning(f"Price fetch failed for {sym}: {e}")

    for pos in positions:
        sym = pos["symbol"]
        current = prices.get(sym)
        if current is None:
            continue
        tp = pos.get("tp")
        sl = pos.get("sl")
        if not tp and not sl:
            continue

        hit_tp = tp and current >= tp
        hit_sl = sl and current <= sl
        if not hit_tp and not hit_sl:
            continue

        # Auto-close the position
        result = close_position(pos["user_id"], sym, current)
        if not result:
            continue

        pnl = result["pnl_pct"]
        pref = get_user_pref(pos["user_id"])
        msg = tp_sl_alert(sym, result["entry"], current, pnl,
                          hit_tp=hit_tp, lang=pref["lang"])
        try:
            await context.bot.send_message(
                chat_id=pos["user_id"],
                text=msg,
                parse_mode="Markdown",
            )
        except Exception as e:
            logger.warning(f"Could not DM user {pos['user_id']} for {sym} TP/SL: {e}")


# ── Scanner ───────────────────────────────────────────────────────────────────

def _update_perf_brake(outcome_correct: bool) -> None:
    """Call after each resolved outcome to maintain the performance brake state."""
    global _perf_brake_engaged
    _perf_history.append(outcome_correct)
    if len(_perf_history) > _PERF_BRAKE_WINDOW:
        _perf_history.pop(0)
    if len(_perf_history) >= _PERF_BRAKE_WINDOW:
        wr = sum(_perf_history) / len(_perf_history)
        # Check current drawdown — tighten brake threshold during deep drawdowns
        try:
            from database import get_equity_curve
            eq = get_equity_curve()
            in_drawdown = eq["max_drawdown_pct"] >= _PERF_BRAKE_DD_LIMIT
        except Exception:
            in_drawdown = False
        threshold = _PERF_BRAKE_DD_THRESHOLD if in_drawdown else _PERF_BRAKE_THRESHOLD
        _perf_brake_engaged = wr < threshold
        if _perf_brake_engaged:
            logger.warning(
                f"[perf_brake] ENGAGED: last {_PERF_BRAKE_WINDOW} outcomes WR={wr:.1%} "
                f"threshold={threshold:.0%} drawdown={in_drawdown} "
                f"— halving signal frequency until performance recovers"
            )


async def scan_and_alert(context: ContextTypes.DEFAULT_TYPE) -> None:
    global _perf_scan_counter
    _perf_scan_counter += 1
    scan_start = _time.time()

    # Performance brake: skip every other scan when recent win rate is critically low
    if _perf_brake_engaged and (_perf_scan_counter % 2 == 0):
        logger.info("[perf_brake] scan skipped — brake engaged")
        return

    await _ensure_models_trained(context.bot)

    # Concurrent direction counters — reset each scan cycle
    direction_count: dict[str, int] = {"BUY": 0, "SELL": 0}

    all_symbols = _get_symbols()
    n_scanned = 0
    n_skipped_market = 0
    n_skipped_brake = 0
    n_signals = 0
    scan_regime: str | None = None

    for symbol in all_symbols:
        # Skip non-crypto outside regular market hours (9:30am–4:00pm ET)
        if not is_crypto(symbol):
            session = market_session(symbol)
            if session != "open":
                logger.debug(f"{symbol}: market {session} — skip")
                n_skipped_market += 1
                continue

        n_scanned += 1
        try:
            # Multi-timeframe: get 1h bias first
            bias_1h, df_1h = get_1h_bias(symbol)

            rule_sig, df_5m = get_signal(symbol, df_1h=df_1h)

            model = models.get(symbol)

            # Get live sentiment and compute momentum before calling predict.
            # sentiment_score flows into ML probability adjustment (soft, continuous)
            # instead of the old hard binary suppress gate.
            sentiment = get_sentiment(symbol)
            s_score = float(sentiment.get("score", 0.0))
            now_ts  = _time.time()
            hist = _sentiment_history.setdefault(symbol, [])
            hist.append((now_ts, s_score))
            # Trim entries older than 3h
            _sentiment_history[symbol] = [(t, s) for t, s in hist
                                          if now_ts - t <= _SENTIMENT_MOMENTUM_WINDOW]
            if len(_sentiment_history[symbol]) >= 2:
                s_momentum = s_score - _sentiment_history[symbol][0][1]
            else:
                s_momentum = 0.0

            # Use 5m model to confirm 5m rule signals — the 1h model was trained on
            # hourly patterns and produces uncorrelated predictions on 5m bars.
            # Falls back to 1h model if 5m model not yet trained.
            if model and df_5m is not None and not df_5m.empty:
                ml_result = model.predict(df_5m, timeframe="5m",
                                          sentiment_score=s_score,
                                          sentiment_momentum=s_momentum)
            elif model and df_1h is not None and not df_1h.empty:
                ml_result = model.predict(df_1h, timeframe="1h",
                                          sentiment_score=s_score,
                                          sentiment_momentum=s_momentum)
            else:
                ml_result = {"buy_prob": None, "sell_prob": None, "ai_signal": None}

            # Compute actual RSI from 5m data for AI-only signals (not the 50.0 default)
            if rule_sig:
                rsi = rule_sig["rsi"]
            elif df_5m is not None and len(df_5m) >= 14:
                try:
                    rsi = float(_rsi(df_5m["Close"].squeeze()).iloc[-1])
                except Exception:
                    rsi = 50.0
            else:
                rsi = 50.0

            price = (rule_sig["price"] if rule_sig
                     else float(df_5m["Close"].squeeze().iloc[-1]) if df_5m is not None else 0.0)

            sig = combine_signals(
                rule_sig, ml_result, symbol, price, rsi,
                df_5m=df_5m,
                df_1h=df_1h,
                bias_1h=bias_1h,
            )
            if sig is None:
                continue

            if get_last_signal_action(symbol, within_hours=6) == sig["action"]:
                continue

            # Sentiment is now a soft ML probability adjustment (applied in predict()),
            # not a hard gate. Append score to reasons for transparency when non-neutral.
            if sentiment.get("label") not in ("neutral", None) and sentiment.get("source") not in ("disabled", "cache_miss"):
                momentum_str = f", Δ{s_momentum:+.2f}" if abs(s_momentum) > 0.05 else ""
                sig.setdefault("reasons", []).append(
                    f"News sentiment: {sentiment['label']} ({s_score:+.2f}{momentum_str})"
                )

            # Suppress AI-only signals for symbols with poor bootstrap accuracy
            if sig.get("strength") == "AI" and not _is_ai_capable(symbol):
                logger.info(f"[quality] suppressed AI-only {sig['action']} {symbol}: no demonstrated skill (AUC < {_AI_MIN_HOLDOUT_AUC} / bootstrap WR < {_AI_MIN_BOOTSTRAP_WR}%)")
                n_skipped_brake += 1
                continue

            # Regime filter — suppress during hostile macro conditions
            spy_df_ctx, vix_df_ctx = get_predict_market_context()
            hostile, regime_reason = _is_hostile_regime(spy_df_ctx, vix_df_ctx, df_1h)
            if hostile:
                logger.info(f"[regime] suppressed {sig['action']} {symbol}: {regime_reason}")
                continue

            # Earnings guard — binary-event risk; suppressed signals are
            # shadow-tracked so the guard's value is measured, not assumed
            if (get_setting("earnings_guard_enabled", "true") == "true"
                    and _near_earnings(symbol)):
                logger.info(f"[earnings_guard] suppressed {sig['action']} {symbol}: "
                            f"earnings within {_EARNINGS_WINDOW_DAYS}d")
                try:
                    log_shadow_signal(sig, source="earnings_guard")
                except Exception:
                    pass
                continue

            # Tag signal with current 3-state regime before persisting
            sig["regime"] = get_market_regime(spy_df_ctx, vix_df_ctx)
            scan_regime = sig["regime"]

            # Concurrent direction limit: max 3 same-direction signals per scan cycle.
            # Prevents correlated position accumulation (e.g. 8 tech stocks all BUY).
            action = sig["action"]
            direction_count[action] = direction_count.get(action, 0) + 1
            if direction_count[action] > _MAX_SAME_DIRECTION_PER_SCAN:
                logger.info(
                    f"[concurrent] suppressed {action} {symbol}: "
                    f"already {_MAX_SAME_DIRECTION_PER_SCAN} {action} signals this scan"
                )
                continue

            # Use 1h bars for saved features — clf_buy/clf_sell are 1h-trained models;
            # 5m bars produce same column names but different scale/lookback values.
            feat = _current_features(df_1h, symbol=symbol) if df_1h is not None else (
                   _current_features(df_5m, symbol=symbol) if df_5m is not None else {})
            sig["company_name"] = _get_company_name(symbol)
            log_signal(sig, features=feat)
            n_signals += 1

            session = market_session(symbol)
            await _broadcast_signal(context.bot, sig, session)

            if sig["action"] == "SELL":
                for uid in get_users_with_open_position(symbol):
                    await _send_private_sell(context.bot, uid, sig)

            # Optional private signal sharing for explicit user subscriptions
            if ENABLE_SUBSCRIBER_DM_SIGNALS:
                open_users = set(get_users_with_open_position(symbol))
                for uid in get_subscribers_for_symbol(symbol):
                    if uid not in open_users:
                        await _send_subscriber_alert(context.bot, uid, sig)

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

    scan_duration_ms = (_time.time() - scan_start) * 1000
    try:
        log_scan(
            scanned_at=datetime.utcnow().isoformat(),
            symbols_total=len(all_symbols),
            symbols_scanned=n_scanned,
            symbols_skipped_market=n_skipped_market,
            symbols_skipped_brake=n_skipped_brake,
            signals_generated=n_signals,
            scan_duration_ms=round(scan_duration_ms, 1),
            regime=scan_regime,
            perf_brake_engaged=_perf_brake_engaged,
        )
    except Exception as e:
        logger.debug(f"[scan_log] failed to log scan: {e}")


# ── Swing scan (daily-bar strategy) ───────────────────────────────────────────

async def swing_scan_and_alert(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Hourly scan of the swing (daily-bar) strategy — only for symbols that
    passed the backtest gate (symbols.swing_enabled, set by backtest_swing.py
    --apply). Daily bars are immune to yfinance's 15-min delay on European
    quotes, and the multi-day horizon matches how outcomes actually resolve.
    """
    symbols = get_swing_enabled_symbols()
    if not symbols:
        return

    n_signals = 0
    for symbol in symbols:
        try:
            # Only fire while the market is open so entry price is actionable
            if market_session(symbol) != "open":
                continue
            # Dedup: one swing signal per symbol per 5 days, and don't stack
            # on top of a recent intraday signal in the same direction
            if get_last_signal_action(symbol, within_hours=120) == "BUY":
                continue

            # Live entry price (falls back to last daily close inside)
            entry_price = None
            try:
                px = fetch_ohlc(symbol, period="1d", interval="5m", retries=1)
                if px is not None and not px.empty:
                    entry_price = float(px["Close"].iloc[-1])
            except Exception:
                pass

            sig = get_swing_signal(symbol, entry_price=entry_price)
            if sig is None:
                continue

            # Same earnings guard as intraday — swing holds span report dates
            if (get_setting("earnings_guard_enabled", "true") == "true"
                    and _near_earnings(symbol)):
                logger.info(f"[earnings_guard] suppressed swing BUY {symbol}")
                try:
                    log_shadow_signal(sig, source="earnings_guard")
                except Exception:
                    pass
                continue

            sig["regime"] = get_market_regime()
            sig["company_name"] = _get_company_name(symbol)
            log_signal(sig, features={})
            n_signals += 1
            await _broadcast_signal(context.bot, sig, session="open")

            if ENABLE_SUBSCRIBER_DM_SIGNALS:
                open_users = set(get_users_with_open_position(symbol))
                for uid in get_subscribers_for_symbol(symbol):
                    if uid not in open_users:
                        await _send_subscriber_alert(context.bot, uid, sig)
        except Exception as e:
            logger.error(f"[swing] error scanning {symbol}: {e}")

    if n_signals:
        logger.info(f"[swing] scan complete — {n_signals} signal(s) fired")


# ── Weekly recap ──────────────────────────────────────────────────────────────

async def weekly_recap(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Post weekly performance summary to the channel every Monday."""
    if not CHANNEL_ID:
        return
    stats = get_weekly_stats()
    total = stats["total"]
    acc = stats["accuracy"]
    resolved = stats["resolved"]
    correct = stats["correct"]

    acc_emoji = "🟢" if acc >= 60 else ("🟡" if acc >= 50 else "🔴")
    lines = [
        "📅 *Weekly Signal Performance*",
        f"Signals sent: `{total}`  ·  Resolved: `{resolved}`",
        f"Win rate: {acc_emoji} `{acc}%`  (`{correct}/{resolved}`)",
        "",
        "Per symbol:",
    ]
    for sym, v in sorted(stats["per_symbol"].items()):
        if v["resolved"] > 0:
            sym_acc = round(v["correct"] / v["resolved"] * 100, 1)
            lines.append(f"  *{sym}*: {v['total']} signals  ·  `{sym_acc}%` win rate")

    lines += [
        "",
        "_Follow the channel for daily signals. Tap 'Track this trade' on any BUY to get private TP/SL alerts._",
    ]
    try:
        await context.bot.send_message(
            chat_id=CHANNEL_ID,
            text="\n".join(lines),
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error(f"Weekly recap failed: {e}")


# ── Message formatting ────────────────────────────────────────────────────────

def _trend_label(trend: str, action: str) -> str:
    icons = {"up": "📈 Uptrend", "down": "📉 Downtrend", "sideways": "➡️ Sideways", "unknown": "— Unknown"}
    label = icons.get(trend, "—")
    if (action == "BUY" and trend == "down") or (action == "SELL" and trend == "up"):
        label += "  ⚠️ Counter-trend"
    return label


def _channel_message(sig: dict, session: str) -> str:
    action = sig["action"]
    symbol = sig["symbol"]
    price = sig["price"]
    tp = sig.get("tp", 0)
    sl = sig.get("sl", 0)
    tp_pct = sig.get("tp_pct", 0)
    sl_pct = sig.get("sl_pct", 0)
    rr = sig.get("rr", 0)
    rsi = sig["rsi"]
    trend = sig.get("trend", "unknown")
    quality = sig.get("quality", 0)
    stars = sig.get("stars", "")
    ai_conf = sig.get("ai_confidence")
    strength = sig.get("strength", "RULE")
    vol_spike = sig.get("vol_spike", False)
    mtf = sig.get("mtf_confirmed", False)
    crypto = is_crypto(symbol)

    strength_labels = {"STRONG": "Rules + AI ✦", "AI": "AI only", "RULE": "Technical"}
    action_emoji = {"BUY": "🟢", "SELL": "🔴"}.get(action, "⚪")
    if strength == "STRONG":
        action_emoji = "🔥" if action == "BUY" else "💀"

    reasons = sig.get("reasons", [])
    ind_parts = []
    if any("RSI" in r for r in reasons):
        ind_parts.append(f"RSI {rsi:.0f}")
    if any("MACD" in r for r in reasons):
        ind_parts.append("MACD " + ("↑" if action == "BUY" else "↓"))
    if any("EMA" in r for r in reasons):
        ind_parts.append("EMA " + ("↑" if action == "BUY" else "↓"))
    if vol_spike:
        ind_parts.append("Vol 🔥")
    ind_line = "  ·  ".join(ind_parts) if ind_parts else "AI pattern"

    tp_sign = "+" if tp_pct >= 0 else ""
    sl_sign = "+" if sl_pct >= 0 else ""
    ai_line = f"\nAI: `{ai_conf * 100:.0f}%` confident" if ai_conf else ""
    mtf_line = "\n✅ Multi-timeframe confirmed (1h + 5m)" if mtf else ""
    session_label = "  ·  🌅 Pre-market" if session == "pre" else ""
    asset_type = "₿ Crypto · 24/7" if crypto else "5-min"
    et_time = datetime.now(ET).strftime("%H:%M ET")

    return (
        f"{action_emoji} *{action} — ${symbol}*  ·  {asset_type}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Entry:       `${price:.2f}`\n"
        f"🎯 Target:  `${tp:.2f}`  (`{tp_sign}{tp_pct:.1f}%`)\n"
        f"🛑 Stop:    `${sl:.2f}`  (`{sl_sign}{sl_pct:.1f}%`)\n"
        f"⚖️  R/R:     `1 : {rr}`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Indicators: {ind_line}\n"
        f"Trend: {_trend_label(trend, action)}\n"
        f"Type: _{strength_labels.get(strength, strength)}_"
        f"{ai_line}"
        f"{mtf_line}\n"
        f"Quality: {stars}  `({quality}/5)`\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"⏱ `{et_time}`{session_label}  ·  #{symbol}"
    )


async def _broadcast_signal(bot, sig: dict, session: str = "open") -> None:
    if not CHANNEL_ID:
        return
    # Channel is always English expert — language personalisation is for private DMs only
    cs, fx = _symbol_fx(sig["symbol"])
    body = signal_msg(sig, lang="en", mode="expert", currency_symbol=cs, fx_rate=fx)
    cet_time = datetime.now(CET).strftime("%H:%M")
    tz_label = "CEST" if datetime.now(CET).dst() else "CET"
    session_label = "  ·  🌅 Pre-market" if session == "pre" else ""
    body += f"\n━━━━━━━━━━━━━━━━━━━\n⏱ `{cet_time} {tz_label}`{session_label}  ·  #{sig['symbol']}"
    if sig["action"] == "BUY" and BOT_USERNAME:
        deep_link = (
            f"https://t.me/{BOT_USERNAME}?start="
            f"track_{sig['symbol']}_{sig['price']}_{sig.get('tp', 0)}_{sig.get('sl', 0)}"
        )
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("📊 Track this trade", url=deep_link)
        ]])
        await bot.send_message(chat_id=CHANNEL_ID, text=body, parse_mode="Markdown", reply_markup=keyboard)
    else:
        await bot.send_message(chat_id=CHANNEL_ID, text=body, parse_mode="Markdown")


async def _send_private_sell(bot, user_id: str, sig: dict) -> None:
    positions = get_user_open_positions(user_id)
    pos = next((p for p in positions if p["symbol"] == sig["symbol"]), None)
    if not pos:
        return

    symbol = sig["symbol"]
    pref = get_user_pref(user_id)
    lang = pref["lang"]
    body = _sell_alert_msg(
        symbol, pos["entry_price"], sig["price"],
        sig.get("tp", 0), sig.get("sl", 0), lang=lang,
    )
    btn_label = t("btn_close_position", lang=lang)
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton(btn_label, callback_data=f"close:{symbol}")
    ]])
    try:
        await bot.send_message(chat_id=user_id, text=body, parse_mode="Markdown", reply_markup=keyboard)
    except Exception as e:
        logger.warning(f"Could not DM user {user_id}: {e}")


async def _send_subscriber_alert(bot, user_id: str, sig: dict) -> None:
    pref = get_user_pref(user_id)
    cs, fx = _symbol_fx(sig["symbol"])
    msg = signal_msg(sig, lang=pref["lang"], mode=pref["mode"], currency_symbol=cs, fx_rate=fx)
    try:
        await bot.send_message(chat_id=user_id, text=msg, parse_mode="Markdown")
    except Exception as e:
        logger.warning(f"Subscriber alert failed for {user_id}: {e}")


# ── Callbacks ─────────────────────────────────────────────────────────────────

async def close_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = str(query.from_user.id)
    _, symbol = query.data.split(":", 1)
    pref = get_user_pref(user_id)
    lang = pref["lang"]

    df = yf.download(symbol, period="1d", interval="5m", progress=False, auto_adjust=True)
    exit_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
    result = close_position(user_id, symbol, exit_price)
    await query.edit_message_reply_markup(reply_markup=None)

    if result:
        pnl = result["pnl_pct"]
        sign = "+" if pnl >= 0 else ""
        emoji = "🟢" if pnl >= 0 else "🔴"
        await query.message.reply_text(
            t("close_confirmed", lang=lang, emoji=emoji, symbol=symbol,
              entry=result["entry"], exit=result["exit"], sign=sign, pnl=pnl),
            parse_mode="Markdown",
        )
    else:
        await query.message.reply_text(t("close_not_found", lang=lang, symbol=symbol))


# ── Menus ─────────────────────────────────────────────────────────────────────

def _reply_keyboard() -> ReplyKeyboardMarkup:
    """Persistent button bar shown at the bottom of the chat."""
    rows = [
        ["📊 Positions", "📈 P&L History"],
        ["🔍 Scan Market", "📉 Stats"],
        ["📬 Subscriptions", "🧪 Backtest"],
        ["⚙️ Settings", "📋 Menu"],
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, input_field_placeholder="Choose an action…")


def _main_menu_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Positions", callback_data="menu:status"),
            InlineKeyboardButton("📈 P&L History", callback_data="menu:pnl"),
        ],
        [
            InlineKeyboardButton("🔍 Scan Market", callback_data="menu:scan"),
            InlineKeyboardButton("📉 Stats", callback_data="menu:stats"),
        ],
        [
            InlineKeyboardButton("📬 Subscriptions", callback_data="menu:subscribe"),
            InlineKeyboardButton("🧪 Backtest", callback_data="menu:backtest_prompt"),
        ],
        [
            InlineKeyboardButton("❓ Help", callback_data="menu:help"),
        ],
    ])


async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_user(update.effective_user.id, update.effective_user.username or "")
    await update.message.reply_text(
        "📋 *Main Menu* — choose an action:",
        parse_mode="Markdown",
        reply_markup=_main_menu_inline(),
    )


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    action = query.data.split(":", 1)[1]

    if action == "status":
        await query.message.reply_text("Loading positions…")
        await cmd_status.__wrapped__(update, context) if hasattr(cmd_status, "__wrapped__") else None
        positions = get_user_open_positions(str(query.from_user.id))
        if not positions:
            await query.message.reply_text("You have no open positions.")
            return
        lines = []
        for p in positions:
            try:
                df = yf.download(p["symbol"], period="1d", interval="5m", progress=False, auto_adjust=True)
                current = float(df["Close"].iloc[-1]) if not df.empty else p["entry_price"]
            except Exception:
                current = p["entry_price"]
            pct = (current - p["entry_price"]) / p["entry_price"] * 100
            sign = "+" if pct >= 0 else ""
            emoji = "🟢" if pct >= 0 else "🔴"
            tp_line = f"\n   TP: `${p['tp']:.2f}`  SL: `${p['sl']:.2f}`" if p.get("tp") and p.get("sl") else ""
            lines.append(
                f"{emoji} *{p['symbol']}*\n"
                f"   Entry: `${p['entry_price']:.2f}` → Now: `${current:.2f}`\n"
                f"   P&L: `{sign}{pct:.2f}%`{tp_line}"
            )
        await query.message.reply_text(
            "*Your open positions:*\n\n" + "\n\n".join(lines),
            parse_mode="Markdown",
        )

    elif action == "pnl":
        history = get_user_pnl(str(query.from_user.id))
        if not history:
            await query.message.reply_text(
                "No closed trades yet.\n\nTrack a BUY signal from the channel to get started."
            )
            return
        total = sum(t["pnl_pct"] for t in history)
        wins = sum(1 for t in history if t["pnl_pct"] > 0)
        avg = total / len(history)
        sign = "+" if avg >= 0 else ""
        lines = []
        for t in history:
            s = "+" if t["pnl_pct"] >= 0 else ""
            e = "🟢" if t["pnl_pct"] >= 0 else "🔴"
            lines.append(
                f"{e} *{t['symbol']}* `{s}{t['pnl_pct']:.2f}%`  —  "
                f"`${t['entry_price']:.2f}` → `${t['exit_price']:.2f}`"
            )
        await query.message.reply_text(
            f"📊 *Your Trade History*\n"
            f"{len(history)} trades  ·  {wins} wins  ·  Avg `{sign}{avg:.2f}%`\n\n"
            + "\n".join(lines),
            parse_mode="Markdown",
        )

    elif action == "scan":
        # Delegate to cmd_scan (no-arg mode = interesting symbols only)
        class _FakeUpdate:
            message = query.message
            effective_user = query.from_user
        class _FakeCtx:
            args = []
        await cmd_scan(_FakeUpdate(), _FakeCtx())

    elif action == "stats":
        s = get_stats()
        acc = (f"{s['accuracy']}%  ({s['correct']}/{s['resolved']} resolved)"
               if s["resolved"] > 0 else "N/A — no resolved signals yet")
        sym_lines = "\n".join(
            f"  {sym}: {v['total']} signals  ·  {v['accuracy']}% accuracy"
            for sym, v in s["per_symbol"].items()
        ) or "  No signals yet"
        await query.message.reply_text(
            f"📊 *Signal Performance*\n\n"
            f"Subscribers: `{s['total_users']}`\n"
            f"Total signals: `{s['total_signals']}`\n"
            f"Today: `{s['today_signals']}`\n"
            f"Accuracy: `{acc}`\n\n"
            f"Per symbol:\n{sym_lines}",
            parse_mode="Markdown",
        )

    elif action == "subscribe":
        subs = get_user_subscriptions(str(query.from_user.id))
        all_symbols = ", ".join(f"`{s}`" for s in SYMBOLS)
        if subs:
            current = ", ".join(f"`{s}`" for s in subs)
            await query.message.reply_text(
                f"📬 *Your subscriptions:* {current}\n\n"
                f"To add more: /subscribe AAPL BTC-USD\n"
                f"To remove all: /unsubscribe\n\n"
                f"Available: {all_symbols}",
                parse_mode="Markdown",
            )
        else:
            # Show subscribe buttons for each symbol
            buttons = [
                InlineKeyboardButton(sym, callback_data=f"sub:{sym}")
                for sym in _get_symbols()
            ]
            rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
            await query.message.reply_text(
                f"📬 *Subscribe to signal alerts*\n\nTap symbols to subscribe:",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(rows),
            )

    elif action == "backtest_prompt":
        buttons = [
            InlineKeyboardButton(sym, callback_data=f"bt:{sym}")
            for sym in _get_symbols()
        ]
        rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
        await query.message.reply_text(
            "🧪 *Backtest* — choose a symbol:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(rows),
        )

    elif action == "help":
        class _FU:
            message = query.message
            effective_user = query.from_user
        class _FC:
            args = []
        await cmd_help(_FU(), _FC())


async def subscribe_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle tap on a symbol subscribe button from the menu."""
    query = update.callback_query
    await query.answer()
    symbol = query.data.split(":", 1)[1]
    subscribe_user(str(query.from_user.id), [symbol])
    await query.edit_message_text(
        f"✅ Subscribed to *{symbol}*\n\nYou'll get DM alerts when signals fire.\n\n"
        f"Use /unsubscribe to remove, or /subscribe to add more.",
        parse_mode="Markdown",
    )


async def backtest_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle tap on a symbol backtest button from the menu."""
    query = update.callback_query
    await query.answer()
    symbol = query.data.split(":", 1)[1]
    await query.edit_message_text(f"⏳ Running 2-year backtest for *{symbol}*…", parse_mode="Markdown")
    try:
        result = run_backtest(symbol, years=2)
        await query.message.reply_text(format_backtest_message(result), parse_mode="Markdown")
    except Exception as e:
        await query.message.reply_text(f"❌ Backtest failed: {e}")


# Text button handler — maps reply keyboard labels to commands
_TEXT_ACTIONS: dict[str, str] = {
    "📊 positions": "status",
    "📈 p&l history": "pnl",
    "🔍 scan market": "scan",
    "📉 stats": "stats",
    "📬 subscriptions": "subscribe",
    "🧪 backtest": "backtest_prompt",
    "⚙️ settings": "settings",
    "📋 menu": "open_menu",
}


async def text_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route reply-keyboard button taps to the right action."""
    text = update.message.text.strip().lower()
    action = _TEXT_ACTIONS.get(text)
    if not action:
        return
    log_user(update.effective_user.id, update.effective_user.username or "")
    if action == "open_menu":
        await update.message.reply_text(
            "📋 *Main Menu* — choose an action:",
            parse_mode="Markdown",
            reply_markup=_main_menu_inline(),
        )
    else:
        # Simulate a callback query by firing the appropriate command handler
        fake_update = update
        if action == "status":
            await cmd_status(fake_update, context)
        elif action == "pnl":
            await cmd_pnl(fake_update, context)
        elif action == "scan":
            await cmd_scan(fake_update, context)
        elif action == "stats":
            await cmd_stats(fake_update, context)
        elif action == "subscribe":
            await cmd_subscribe(fake_update, context)
        elif action == "settings":
            await cmd_settings(update, context)
        elif action == "backtest_prompt":
            buttons = [
                InlineKeyboardButton(sym, callback_data=f"bt:{sym}")
                for sym in _get_symbols()
            ]
            rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
            await update.message.reply_text(
                "🧪 *Backtest* — choose a symbol:",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(rows),
            )


# ── Settings ──────────────────────────────────────────────────────────────────

def _settings_keyboard(lang: str, mode: str) -> InlineKeyboardMarkup:
    check = lambda val, cur: "✓ " if val == cur else ""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(f"{check('en', lang)}🇬🇧 English",  callback_data="set:lang:en"),
            InlineKeyboardButton(f"{check('fa', lang)}🇮🇷 فارسی",   callback_data="set:lang:fa"),
        ],
        [
            InlineKeyboardButton(f"{check('beginner', mode)}📗 Beginner / مبتدی", callback_data="set:mode:beginner"),
            InlineKeyboardButton(f"{check('expert', mode)}📘 Expert / حرفه‌ای",   callback_data="set:mode:expert"),
        ],
    ])


async def cmd_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    log_user(uid, update.effective_user.username or "")
    pref = get_user_pref(uid)
    await update.message.reply_text(
        t("settings_title", lang=pref["lang"]),
        parse_mode="Markdown",
        reply_markup=_settings_keyboard(pref["lang"], pref["mode"]),
    )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    uid = str(query.from_user.id)
    _, field, value = query.data.split(":")
    set_user_pref(uid, lang=value if field == "lang" else None,
                  mode=value if field == "mode" else None)
    pref = get_user_pref(uid)

    if field == "lang":
        key = "settings_saved_lang_en" if value == "en" else "settings_saved_lang_fa"
    else:
        key = "settings_saved_beginner" if value == "beginner" else "settings_saved_expert"

    await query.edit_message_text(
        t(key, lang=pref["lang"]) + "\n\n" + t("settings_title", lang=pref["lang"]),
        parse_mode="Markdown",
        reply_markup=_settings_keyboard(pref["lang"], pref["mode"]),
    )


# ── Commands ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    log_user(user.id, user.username or "")

    pref = get_user_pref(user.id)
    lang = pref["lang"]

    if context.args and context.args[0].startswith("track_"):
        parts = context.args[0].split("_")
        if len(parts) in (3, 5):
            _, symbol, price_str, *rest = parts
            try:
                price = float(price_str)
                tp = None
                sl = None
                if len(rest) == 2:
                    tp = float(rest[0]) if float(rest[0]) > 0 else None
                    sl = float(rest[1]) if float(rest[1]) > 0 else None
                else:
                    latest_buy = get_latest_signal(symbol, action="BUY")
                    if latest_buy:
                        tp = latest_buy.get("tp")
                        sl = latest_buy.get("sl")
                open_position(str(user.id), symbol, price, tp=tp, sl=sl)
                await update.message.reply_text(
                    t("position_tracked", lang=lang, symbol=symbol, price=price),
                    parse_mode="Markdown",
                )
                return
            except ValueError:
                pass

    channel_text = f"\n📢 Channel: {CHANNEL_ID}" if CHANNEL_ID else ""
    await update.message.reply_text(
        t("welcome", lang=lang) + channel_text,
        parse_mode="Markdown",
        reply_markup=_reply_keyboard(),
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    log_user(uid, update.effective_user.username or "")
    pref = get_user_pref(uid)
    lang = pref["lang"]
    positions = get_user_open_positions(str(uid))
    if not positions:
        await update.message.reply_text(t("status_no_positions", lang=lang))
        return

    lines = []
    for p in positions:
        try:
            df = yf.download(p["symbol"], period="1d", interval="5m", progress=False, auto_adjust=True)
            current = float(df["Close"].iloc[-1]) if not df.empty else p["entry_price"]
        except Exception:
            current = p["entry_price"]
        pct = (current - p["entry_price"]) / p["entry_price"] * 100
        sign = "+" if pct >= 0 else ""
        emoji = "🟢" if pct >= 0 else "🔴"
        if lang == "fa":
            tp_line = f"\n   هدف: `${p['tp']:.2f}`  حد ضرر: `${p['sl']:.2f}`" if p.get("tp") and p.get("sl") else ""
            lines.append(
                f"{emoji} *{p['symbol']}*\n"
                f"   ورود: `${p['entry_price']:.2f}` ← اکنون: `${current:.2f}`\n"
                f"   سود/زیان: `{sign}{pct:.2f}٪`{tp_line}"
            )
        else:
            tp_line = f"\n   TP: `${p['tp']:.2f}`  SL: `${p['sl']:.2f}`" if p.get("tp") and p.get("sl") else ""
            lines.append(
                f"{emoji} *{p['symbol']}*\n"
                f"   Entry: `${p['entry_price']:.2f}` → Now: `${current:.2f}`\n"
                f"   P&L: `{sign}{pct:.2f}%`{tp_line}"
            )
    header = t("status_header", lang=lang)
    await update.message.reply_text(f"{header}\n\n" + "\n\n".join(lines), parse_mode="Markdown")


async def cmd_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    log_user(uid, update.effective_user.username or "")
    pref = get_user_pref(uid)
    lang = pref["lang"]
    history = get_user_pnl(str(uid))
    if not history:
        await update.message.reply_text(t("pnl_empty", lang=lang))
        return

    total = sum(tr["pnl_pct"] for tr in history)
    wins = sum(1 for tr in history if tr["pnl_pct"] > 0)
    avg = total / len(history)
    sign = "+" if avg >= 0 else ""
    lines = []
    for tr in history:
        s = "+" if tr["pnl_pct"] >= 0 else ""
        e = "🟢" if tr["pnl_pct"] >= 0 else "🔴"
        pnl_unit = "٪" if lang == "fa" else "%"
        lines.append(
            f"{e} *{tr['symbol']}* `{s}{tr['pnl_pct']:.2f}{pnl_unit}`  —  "
            f"`${tr['entry_price']:.2f}` → `${tr['exit_price']:.2f}`"
        )
    header = t("pnl_header", lang=lang, n=len(history), wins=wins, sign=sign, avg=avg)
    await update.message.reply_text(header + "\n\n" + "\n".join(lines), parse_mode="Markdown")


async def cmd_close(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    log_user(uid, update.effective_user.username or "")
    pref = get_user_pref(uid)
    lang = pref["lang"]

    if not context.args:
        # No arg — show open positions as inline close buttons
        positions = get_user_open_positions(str(uid))
        if not positions:
            await update.message.reply_text(t("status_no_positions", lang=lang))
            return
        buttons = []
        for p in positions:
            label = f"❌ {p['symbol']}"
            buttons.append([InlineKeyboardButton(label, callback_data=f"close:{p['symbol']}")])
        prompt = ("کدام موقعیت را می‌خواهید ببندید؟" if lang == "fa"
                  else "Which position do you want to close?")
        await update.message.reply_text(
            prompt,
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return

    symbol = context.args[0].upper()
    df = yf.download(symbol, period="1d", interval="5m", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    exit_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
    result = close_position(str(uid), symbol, exit_price)
    if result:
        pnl = result["pnl_pct"]
        sign = "+" if pnl >= 0 else ""
        emoji = "🟢" if pnl >= 0 else "🔴"
        await update.message.reply_text(
            t("cancel_closed", lang=lang, emoji=emoji, symbol=symbol, sign=sign, pnl=pnl),
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(t("cancel_not_found", lang=lang, symbol=symbol))


# Backward-compatible alias
async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_close(update, context)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    log_user(uid, update.effective_user.username or "")
    pref = get_user_pref(uid)
    lang = pref["lang"]
    s = get_stats()
    if s["resolved"] > 0:
        acc = t("stats_accuracy", lang=lang, pct=s["accuracy"],
                correct=s["correct"], resolved=s["resolved"])
    else:
        acc = t("stats_no_data", lang=lang)
    pct_label = "٪" if lang == "fa" else "%"
    sym_lines = "\n".join(
        f"  {sym}: {v['total']}  ·  {v['accuracy']}{pct_label}"
        for sym, v in s["per_symbol"].items()
    ) or ("  هنوز سیگنالی نیست" if lang == "fa" else "  No signals yet")
    body = t("stats_body", lang=lang, users=s["total_users"], total=s["total_signals"],
             today=s["today_signals"], acc=acc, sym_lines=sym_lines)
    await update.message.reply_text(
        t("stats_header", lang=lang) + "\n\n" + body, parse_mode="Markdown"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_user(update.effective_user.id, update.effective_user.username or "")
    pref = get_user_pref(update.effective_user.id)
    lang = pref["lang"]
    if lang == "fa":
        text = (
            "📋 *دستورات موجود*\n\n"
            "/start — خوش‌آمدگویی\n"
            "/menu — منوی اصلی\n"
            "/settings — زبان و سطح تجربه\n\n"
            "*معاملات*\n"
            "/status — موقعیت‌های باز با سود/زیان زنده\n"
            "/pnl — تاریخچه معاملات بسته‌شده\n"
            "/close — بستن موقعیت (یا /close SYMBOL)\n\n"
            "*تحلیل*\n"
            "/scan — نمادهای قابل توجه (RSI حدی)\n"
            "/scan SYMBOL — تحلیل دقیق یک نماد\n"
            "/backtest SYMBOL [years] — بک‌تست تاریخی\n"
            "/stats — دقت سیگنال‌ها\n\n"
            "*اطلاع‌رسانی*\n"
            "/subscribe SYMBOL — دریافت هشدار DM\n"
            "/unsubscribe — حذف اشتراک‌ها\n"
        )
    else:
        text = (
            "📋 *Available Commands*\n\n"
            "/start — Welcome & onboarding\n"
            "/menu — Main menu\n"
            "/settings — Language & experience level\n\n"
            "*Positions*\n"
            "/status — Open positions with live P&L\n"
            "/pnl — Closed trade history\n"
            "/close — Close a position (or /close SYMBOL)\n\n"
            "*Analysis*\n"
            "/scan — Notable symbols (RSI extremes)\n"
            "/scan SYMBOL — Detailed analysis of one symbol\n"
            "/backtest SYMBOL \\[years\\] — Historical backtest\n"
            "/stats — Signal accuracy stats\n\n"
            "*Alerts*\n"
            "/subscribe SYMBOL — Get DM alerts for a symbol\n"
            "/unsubscribe — Remove subscriptions\n"
        )
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    log_user(user.id, user.username or "")
    user_id = str(user.id)
    pref = get_user_pref(user_id)
    lang = pref["lang"]
    active = _get_symbols()
    all_fmt = ", ".join(f"`{s}`" for s in active)

    if not context.args:
        subs = get_user_subscriptions(user_id)
        # Always show inline buttons so user can tap to subscribe/manage
        buttons = [
            InlineKeyboardButton(
                ("✅ " if s in subs else "") + s,
                callback_data=f"sub:{s}"
            )
            for s in active
        ]
        rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
        prompt = ("اشتراک‌های شما با ✅ نشان داده شده‌اند. برای افزودن روی نماد بزنید:" if lang == "fa"
                  else "Your active subscriptions are marked ✅. Tap a symbol to subscribe:")
        current_note = ""
        if subs:
            current_note = (f"\n\nاشتراک فعال: {', '.join(f'`{s}`' for s in subs)}" if lang == "fa"
                            else f"\n\nActive: {', '.join(f'`{s}`' for s in subs)}")
        await update.message.reply_text(
            prompt + current_note,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        return

    requested = [s.strip().upper() for s in context.args]
    valid = [s for s in requested if s in [sym.upper() for sym in active]]
    invalid = [s for s in requested if s not in valid]
    if valid:
        subscribe_user(user_id, valid)
        added = ", ".join(f"`{s}`" for s in valid)
        msg = t("subscribe_added", lang=lang, added=added)
        if invalid:
            warn = (f"\n\n⚠️ یافت نشد: {', '.join(invalid)}" if lang == "fa"
                    else f"\n\n⚠️ Not found: {', '.join(invalid)}")
            msg += warn
        await update.message.reply_text(msg, parse_mode="Markdown")
    else:
        not_found = ("❌ هیچ‌کدام از نمادها یافت نشد.\n\nموجود: " if lang == "fa"
                     else "❌ None of those symbols are tracked.\n\nAvailable: ")
        await update.message.reply_text(not_found + all_fmt, parse_mode="Markdown")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    pref = get_user_pref(user_id)
    lang = pref["lang"]
    if context.args:
        sym = context.args[0].upper()
        unsubscribe_user(user_id, sym)
        await update.message.reply_text(t("unsubscribed_sym", lang=lang, symbol=sym),
                                        parse_mode="Markdown")
    else:
        unsubscribe_user(user_id)
        await update.message.reply_text(t("unsubscribed", lang=lang))


async def cmd_backtest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_user(update.effective_user.id, update.effective_user.username or "")
    if not context.args:
        active = _get_symbols()
        buttons = [InlineKeyboardButton(sym, callback_data=f"bt:{sym}") for sym in active]
        rows = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
        await update.message.reply_text(
            "📊 *Backtest* — choose a symbol:\n_(or: /backtest SYMBOL \\[years\\])_",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        return

    symbol = context.args[0].upper()
    years = 2
    if len(context.args) >= 2:
        try:
            years = max(1, min(int(context.args[1]), 5))
        except ValueError:
            pass

    await update.message.reply_text(
        f"⏳ Running {years}-year backtest for *{symbol}*…", parse_mode="Markdown"
    )
    try:
        result = run_backtest(symbol, years=years)
        msg = format_backtest_message(result)
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Backtest failed: {e}")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    log_user(update.effective_user.id, update.effective_user.username or "")
    session = market_session()
    session_labels = {
        "open": "🟢 Market open", "pre": "🌅 Pre-market",
        "after": "🌙 After-hours", "closed": "⛔ Market closed",
    }
    session_str = session_labels.get(session, "")

    # /scan SYMBOL — detailed single-symbol analysis
    if context.args:
        symbol = context.args[0].upper()
        await update.message.reply_text(f"🔍 Scanning *{symbol}*…", parse_mode="Markdown")
        try:
            interval = "1h" if is_crypto(symbol) else "5m"
            period = "30d" if is_crypto(symbol) else "5d"
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 50:
                await update.message.reply_text(f"❌ No data for *{symbol}*", parse_mode="Markdown")
                return
            close = df["Close"].squeeze()
            rsi_val = float(_rsi(close).iloc[-1])
            macd_line, sig_line = _macd(close)
            ema9  = close.ewm(span=9,  adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            price = float(close.iloc[-1])
            atr_series = _atr(df)
            atr_val = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            vol = df["Volume"].squeeze().astype(float)
            vol_avg = float(vol.iloc[-20:].mean()) if len(vol) >= 20 else 0.0
            vol_now = float(vol.iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

            macd_val = float(macd_line.iloc[-1])
            sig_val  = float(sig_line.iloc[-1])
            macd_dir = "↑ bullish" if macd_val > sig_val else "↓ bearish"
            ema_dir  = "↑ bullish" if float(ema9.iloc[-1]) > float(ema21.iloc[-1]) else "↓ bearish"

            rsi_label = ("🔴 Overbought" if rsi_val > 65 else
                         "🟢 Oversold"   if rsi_val < 35 else
                         "⚪ Neutral")

            _, df_1h = get_1h_bias(symbol)
            bias_1h, _ = get_1h_bias(symbol)

            model = models.get(symbol)
            ml_result = {}
            if model and not df.empty:
                ml_result = model.predict(df, timeframe=interval)
            buy_prob  = ml_result.get("buy_prob")
            sell_prob = ml_result.get("sell_prob")
            ai_signal = ml_result.get("ai_signal")

            ai_lines = []
            if buy_prob is not None:
                bar = "█" * int(buy_prob * 10) + "░" * (10 - int(buy_prob * 10))
                ai_lines.append(f"  AI Buy:  `{bar}` `{buy_prob*100:.0f}%`")
            if sell_prob is not None:
                bar = "█" * int(sell_prob * 10) + "░" * (10 - int(sell_prob * 10))
                ai_lines.append(f"  AI Sell: `{bar}` `{sell_prob*100:.0f}%`")
            if ai_signal:
                ai_lines.append(f"  AI Signal: *{ai_signal}*")

            last_sig = get_last_signal_action(symbol, within_hours=6)
            recent_line = f"\n⚡ Recent signal: *{last_sig}* (last 6h)" if last_sig else ""

            name = _get_company_name(symbol)
            display = f"{name} ({symbol})" if name else symbol
            crypto_tag = " ₿" if is_crypto(symbol) else ""

            msg = (
                f"🔍 *{display}*{crypto_tag}  {session_str}\n\n"
                f"💰 Price:  `${price:.2f}`\n"
                f"📊 RSI:    `{rsi_val:.1f}` — {rsi_label}\n"
                f"📈 MACD:   {macd_dir}\n"
                f"📉 EMA:    {ema_dir}\n"
                f"📏 ATR:    `{atr_val:.2f}` ({atr_val/price*100:.2f}%)\n"
                f"📦 Volume: `{vol_ratio:.1f}×` avg"
                + (f"\n🕐 1h bias: *{bias_1h}*" if bias_1h else "")
                + ("\n\n*AI Analysis:*\n" + "\n".join(ai_lines) if ai_lines else "")
                + recent_line
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Error scanning {symbol}: {e}")
        return

    # /scan (no arg) — show only "interesting" symbols: RSI extremes or recent signal
    await update.message.reply_text(f"🔍 Scanning watchlist… {session_str}")
    interesting = []
    for symbol in _get_symbols():
        try:
            interval = "1h" if is_crypto(symbol) else "5m"
            period = "30d" if is_crypto(symbol) else "5d"
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty or len(df) < 50:
                continue
            close = df["Close"].squeeze()
            rsi_val = float(_rsi(close).iloc[-1])
            price = float(close.iloc[-1])
            macd_line, sig_line = _macd(close)
            ema9  = close.ewm(span=9,  adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            macd_dir = "↑" if float(macd_line.iloc[-1]) > float(sig_line.iloc[-1]) else "↓"
            ema_dir  = "↑" if float(ema9.iloc[-1]) > float(ema21.iloc[-1]) else "↓"

            flags = []
            if rsi_val < 35:
                flags.append(f"🟢 RSI {rsi_val:.0f} oversold")
            elif rsi_val > 65:
                flags.append(f"🔴 RSI {rsi_val:.0f} overbought")

            last_sig = get_last_signal_action(symbol, within_hours=6)
            if last_sig:
                flags.append(f"⚡ {last_sig} signal fired")

            if not flags:
                continue

            model = models.get(symbol)
            ai_buy = None
            if model and not df.empty:
                ai_buy = model.predict(df, timeframe=interval).get("buy_prob")
            ai_text = f"  AI `{ai_buy*100:.0f}%`" if ai_buy is not None else ""
            crypto_tag = " ₿" if is_crypto(symbol) else ""
            flag_str = "  ·  ".join(flags)
            interesting.append(
                f"*{symbol}*{crypto_tag}  `${price:.2f}`  ·  MACD {macd_dir}  EMA {ema_dir}{ai_text}\n"
                f"  {flag_str}"
            )
        except Exception:
            continue

    if interesting:
        await update.message.reply_text(
            f"{session_str}\n\n" + "\n\n".join(interesting),
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            f"{session_str}\n\n✅ Nothing extreme right now — all RSI readings are neutral.\n"
            f"Use `/scan SYMBOL` for a detailed view.",
            parse_mode="Markdown",
        )


async def cmd_train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if str(update.effective_user.id) != ADMIN_CHAT_ID:
        return
    await update.message.reply_text("🧠 Retraining all models with latest outcome data...")
    # trained=False is the only flag needs_retrain() honors unconditionally —
    # trained_at=0 is ignored below the live-outcome blend threshold
    for model in models.values():
        model.trained = False
    await _ensure_models_trained(context.bot)
    await update.message.reply_text("✅ All models retrained.")


async def cmd_test(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if str(update.effective_user.id) != ADMIN_CHAT_ID:
        return

    uid = update.effective_user.id
    pref = get_user_pref(uid)
    lang = pref["lang"]

    sample_sig = {
        "symbol": "AAPL", "action": "BUY", "price": 192.50,
        "tp": 196.40, "sl": 190.60, "tp_pct": 2.0, "sl_pct": -1.0, "rr": 1.7,
        "rsi": 38.2, "trend": "up", "quality": 4, "stars": "⭐⭐⭐⭐☆",
        "ai_confidence": 0.74, "strength": "STRONG", "vol_spike": True,
        "mtf_confirmed": True,
        "reasons": ["RSI oversold (38.2)", "MACD bullish crossover", "Volume spike (+50% above avg)",
                    "Multi-timeframe confirmed (1h+5m)"],
    }

    if not CHANNEL_ID:
        # No channel — just show personal preview
        await update.message.reply_text(
            signal_msg(sample_sig, lang=lang, mode=pref["mode"]),
            parse_mode="Markdown",
        )
        return

    try:
        await _broadcast_signal(context.bot, sample_sig, session="open")
        # Send confirmation + personal DM preview
        await update.message.reply_text(
            t("test_channel_ok", lang=lang, channel=CHANNEL_ID),
            parse_mode="Markdown",
        )
        # Personal preview — shows how DM signals look with their language/mode
        await update.message.reply_text(
            signal_msg(sample_sig, lang=lang, mode=pref["mode"]),
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(
            t("test_channel_fail", lang=lang, channel=CHANNEL_ID, error=str(e)),
            parse_mode="Markdown",
        )


# ── App setup ─────────────────────────────────────────────────────────────────

async def post_init(app: Application) -> None:
    global BOT_USERNAME
    me = await app.bot.get_me()
    BOT_USERNAME = me.username
    init_db()
    # Seed symbols from env into DB on first run (no-op if already present)
    seed_symbols(SYMBOLS)
    # Sync models dict with DB in case symbols were added/removed via dashboard
    for sym in get_active_symbols():
        if sym not in models:
            models[sym] = StockModel(sym)

    # Fetch company names from yfinance for any symbol that doesn't have one yet
    _populate_symbol_names()

    # Register commands so they appear in Telegram's "/" command list
    await app.bot.set_my_commands([
        BotCommand("menu",        "Show main menu"),
        BotCommand("help",        "List all commands"),
        BotCommand("settings",    "Language & experience level / زبان و سطح"),
        BotCommand("status",      "Open positions with live P&L"),
        BotCommand("pnl",         "Closed trade history"),
        BotCommand("close",       "Close a position (tap or /close SYMBOL)"),
        BotCommand("subscribe",   "Get DM alerts for a symbol"),
        BotCommand("unsubscribe", "Remove subscriptions"),
        BotCommand("scan",        "Notable symbols, or /scan SYMBOL for detail"),
        BotCommand("backtest",    "Historical backtest for a symbol"),
        BotCommand("stats",       "Signal accuracy stats"),
    ])

    if ADMIN_CHAT_ID:
        active_syms = get_active_symbols()
        crypto_syms = [s for s in active_syms if is_crypto(s)]
        stock_syms  = [s for s in active_syms if not is_crypto(s)]
        watch_lines = []
        if stock_syms:
            watch_lines.append(f"Stocks: `{', '.join(stock_syms)}`")
        if crypto_syms:
            watch_lines.append(f"Crypto: `{', '.join(crypto_syms)}`")
        # Show session for each unique market type among active symbols
        labels = {"open": "🟢 Open", "pre": "🌅 Pre-market", "after": "🌙 After-hours", "closed": "⛔ Closed"}
        sessions = {market_session(s) for s in active_syms}
        session_str = " | ".join(labels.get(s, s) for s in sorted(sessions))
        await app.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=(
                f"✅ *Bot live* @{BOT_USERNAME}\n"
                f"Channel: `{CHANNEL_ID or 'not set'}`\n"
                + ("\n".join(watch_lines) + "\n" if watch_lines else "")
                + f"Scan: every {SCAN_INTERVAL // 60} min\n"
                f"Market: {session_str or '⛔ Closed'}"
            ),
            parse_mode="Markdown",
        )


def main() -> None:
    if not BOT_TOKEN or BOT_TOKEN == "your_bot_token_here":
        raise SystemExit("Set BOT_TOKEN in .env before running.")

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    for cmd, fn in [
        ("start", cmd_start), ("menu", cmd_menu), ("help", cmd_help),
        ("settings", cmd_settings),
        ("status", cmd_status), ("pnl", cmd_pnl),
        ("close", cmd_close), ("cancel", cmd_cancel),
        ("stats", cmd_stats), ("scan", cmd_scan),
        ("train", cmd_train), ("test", cmd_test),
        ("subscribe", cmd_subscribe), ("unsubscribe", cmd_unsubscribe),
        ("backtest", cmd_backtest),
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    app.add_handler(CallbackQueryHandler(close_callback, pattern=r"^close:"))
    app.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^menu:"))
    app.add_handler(CallbackQueryHandler(subscribe_callback, pattern=r"^sub:"))
    app.add_handler(CallbackQueryHandler(backtest_callback, pattern=r"^bt:"))
    app.add_handler(CallbackQueryHandler(settings_callback, pattern=r"^set:"))

    # Reply keyboard text buttons
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, text_button_handler
    ))

    # Refresh symbol list from DB every 5 minutes
    app.job_queue.run_repeating(refresh_symbols, interval=300, first=30)
    # Scan every SCAN_INTERVAL seconds
    app.job_queue.run_repeating(scan_and_alert, interval=SCAN_INTERVAL, first=10)
    # Check pending outcomes frequently so resolved trades feed learning faster
    app.job_queue.run_repeating(check_outcomes, interval=OUTCOME_CHECK_INTERVAL, first=180)
    # Continuous learning: re-check all symbols on a shorter cadence
    app.job_queue.run_repeating(continuous_learning, interval=LEARNING_CHECK_INTERVAL, first=120)
    # Operator action queue from dashboard controls
    app.job_queue.run_repeating(process_action_requests, interval=ACTION_QUEUE_INTERVAL, first=30)
    # Monitor open positions for TP/SL every 5 minutes
    app.job_queue.run_repeating(monitor_positions, interval=300, first=60)
    # Refresh news sentiment cache every 30 minutes
    app.job_queue.run_repeating(refresh_sentiment_cache, interval=1800, first=60)
    # Weekly recap: every Monday at 9:00 UTC
    app.job_queue.run_daily(weekly_recap, time=datetime.strptime("09:00", "%H:%M").time(),
                            days=(0,))  # 0 = Monday
    # Nightly PSI drift detection: 02:00 UTC daily
    app.job_queue.run_daily(drift_detection_check,
                            time=datetime.strptime("02:00", "%H:%M").time())
    # Nightly DB backup: 03:00 UTC daily — the labeled outcomes in trading.db
    # are the one thing this system can't regenerate
    app.job_queue.run_daily(nightly_db_backup,
                            time=datetime.strptime("03:00", "%H:%M").time())
    # Weekly ML health report: Monday 09:05 UTC (right after the weekly recap)
    app.job_queue.run_daily(ml_health_report,
                            time=datetime.strptime("09:05", "%H:%M").time(),
                            days=(0,))
    # Pooled cross-symbol 5m model refresh: Sunday 02:30 UTC (markets closed)
    app.job_queue.run_daily(weekly_pooled_retrain,
                            time=datetime.strptime("02:30", "%H:%M").time(),
                            days=(6,))
    # Swing (daily-bar) strategy scan: hourly, gated by symbols.swing_enabled
    app.job_queue.run_repeating(swing_scan_and_alert, interval=3600, first=90)

    logger.info(f"Bot @{BOT_USERNAME} started — channel: {CHANNEL_ID}")
    app.run_polling()


if __name__ == "__main__":
    main()
