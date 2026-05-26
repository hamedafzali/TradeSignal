"""
Sentiment provider abstraction.

Provider is read from DB settings at call time — switch without restart:
  disabled       — always returns neutral, no external calls
  local_finbert  — calls FinBERT sidecar at SENTIMENT_LOCAL_URL
  claude         — calls Claude API (requires claude_api_key in settings)

News is fetched from Finnhub (free tier) when finnhub_api_key is set.
Results are cached in the sentiment_cache table for CACHE_MINUTES.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

CACHE_MINUTES = 30
_NEUTRAL = {"score": 0.0, "label": "neutral", "headlines": [], "source": "disabled"}


# ── Settings helpers ──────────────────────────────────────────────────────────

def _setting(key: str, default: str = "") -> str:
    try:
        from database import get_setting
        return get_setting(key, default)
    except Exception:
        return default


# ── News fetching ─────────────────────────────────────────────────────────────

def fetch_headlines(symbol: str) -> list[str]:
    """Fetch recent headlines from Finnhub. Returns [] on any failure."""
    api_key = _setting("finnhub_api_key")
    if not api_key:
        return []
    try:
        hours = int(_setting("news_lookback_hours", "6"))
        to_dt = datetime.utcnow()
        from_dt = to_dt - timedelta(hours=hours)
        # Finnhub uses stock tickers without exchange suffix for news
        ticker = symbol.replace("-USD", "").replace("-", "")
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={urllib.parse.quote(ticker)}"
            f"&from={from_dt.strftime('%Y-%m-%d')}"
            f"&to={to_dt.strftime('%Y-%m-%d')}"
            f"&token={api_key}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            articles = json.loads(resp.read())
        headlines = [a["headline"] for a in articles if a.get("headline")][:8]
        logger.debug(f"[sentiment] {symbol}: {len(headlines)} headlines")
        return headlines
    except Exception as e:
        logger.warning(f"[sentiment] news fetch failed for {symbol}: {e}")
        return []


# ── Provider implementations ──────────────────────────────────────────────────

def _score_local_finbert(headlines: list[str]) -> dict:
    url = _setting("sentiment_local_url", "http://finbert:5001")
    try:
        payload = json.dumps({"texts": headlines}).encode()
        req = urllib.request.Request(
            f"{url.rstrip('/')}/sentiment",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"[sentiment] local_finbert error: {e}")
        return _NEUTRAL.copy()


def _score_claude(headlines: list[str], symbol: str) -> dict:
    api_key = _setting("claude_api_key")
    if not api_key:
        return _NEUTRAL.copy()
    try:
        joined = "\n".join(f"- {h}" for h in headlines)
        prompt = (
            f"Analyze these news headlines for {symbol} and return a JSON object with:\n"
            f"- score: float from -1.0 (very negative) to 1.0 (very positive)\n"
            f"- label: one of positive / negative / neutral\n\n"
            f"Headlines:\n{joined}\n\n"
            f"Return only valid JSON, no explanation."
        )
        payload = json.dumps({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        text = data["content"][0]["text"].strip()
        result = json.loads(text)
        return {
            "score": float(result.get("score", 0.0)),
            "label": result.get("label", "neutral"),
            "headlines": headlines,
            "source": "claude",
        }
    except Exception as e:
        logger.warning(f"[sentiment] claude error: {e}")
        return _NEUTRAL.copy()


# ── Public API ────────────────────────────────────────────────────────────────

def analyze(headlines: list[str], symbol: str = "") -> dict:
    """Score headlines with the configured provider. Always returns a safe dict."""
    if not headlines:
        return {**_NEUTRAL, "source": "no_headlines"}

    provider = _setting("sentiment_provider", "disabled")

    if provider == "local_finbert":
        result = _score_local_finbert(headlines)
    elif provider == "claude":
        result = _score_claude(headlines, symbol)
    else:
        return {**_NEUTRAL, "source": "disabled"}

    result["headlines"] = headlines
    result["source"] = provider
    return result


def refresh_cache(symbol: str) -> dict:
    """Fetch news, score, store in DB. Returns the sentiment dict."""
    try:
        from database import set_sentiment_cache
    except ImportError:
        return _NEUTRAL.copy()

    headlines = fetch_headlines(symbol)
    result = analyze(headlines, symbol=symbol)
    try:
        set_sentiment_cache(symbol, result)
    except Exception as e:
        logger.warning(f"[sentiment] cache write failed for {symbol}: {e}")
    return result


def get_sentiment(symbol: str) -> dict:
    """
    Returns cached sentiment if fresh, otherwise returns neutral immediately
    (background refresh handles staleness — never blocks scan).
    """
    provider = _setting("sentiment_provider", "disabled")
    if provider == "disabled":
        return {**_NEUTRAL, "source": "disabled"}

    try:
        from database import get_sentiment_cache
        cached = get_sentiment_cache(symbol)
        if cached:
            return cached
    except Exception:
        pass

    # Cache miss — return neutral so scan isn't blocked
    # The background refresh_all_sentiment job will populate it soon
    return {**_NEUTRAL, "source": "cache_miss"}


def should_suppress(signal_action: str, sentiment: dict) -> bool:
    """
    Returns True if sentiment strongly contradicts the signal direction.
    Threshold is configurable in DB settings.
    """
    score = sentiment.get("score", 0.0) or 0.0
    label = sentiment.get("label", "neutral")
    source = sentiment.get("source", "disabled")

    if source in ("disabled", "cache_miss", "no_headlines"):
        return False

    threshold = float(_setting("sentiment_suppress_threshold", "0.35"))

    if signal_action == "BUY" and label == "negative" and score < -threshold:
        return True
    if signal_action == "SELL" and label == "positive" and score > threshold:
        return True
    return False
