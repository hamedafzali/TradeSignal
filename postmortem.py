"""
LLM post-mortem tagging for resolved trades.

Each resolved (correct/incorrect) signal gets one structured "why" tag with
news context — information the technical features can't see. Tags accumulate
in trade_postmortems; clusters become filters (a cause covering >=25% of
tagged losses at n>=30 is actionable — see docs/llm-evaluation.md).

Fully optional: without an LLM API key nothing runs and nothing breaks.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

CAUSES = [
    "earnings_event",     # earnings report moved the price
    "news_event",         # company-specific news (not earnings)
    "market_wide_move",   # index/macro move dragged the stock along
    "sector_move",        # sector rotation, peer contagion
    "stop_too_tight",     # thesis fine, SL inside normal noise
    "target_too_far",     # direction right, TP beyond the move
    "good_setup",         # clean technical win, no external driver
    "unclear",            # no identifiable cause
]

_SYSTEM = (
    "You are a trading post-mortem analyst. Given the facts of one resolved "
    "trade and recent news headlines, identify the most likely primary cause "
    "of the outcome. Be skeptical: only attribute to news/earnings when the "
    "headlines actually support it; otherwise prefer the technical causes or "
    "'unclear'. Respond ONLY with a JSON object:\n"
    '{"cause": "<one of: ' + ", ".join(CAUSES) + '>", '
    '"market_related": <bool>, "earnings_related": <bool>, '
    '"explanation": "<max 25 words>"}'
)


def run_postmortem(sig: dict, outcome: str, resolution: dict) -> None:
    """Tag one resolved trade. Sync (call via asyncio.to_thread). Never raises."""
    try:
        from llm import is_enabled, llm_complete, resolved_model
        if not is_enabled():
            return

        headlines: list[str] = []
        try:
            from sentiment import fetch_headlines
            headlines = (fetch_headlines(sig["symbol"]) or [])[:8]
        except Exception:
            pass

        meta = resolution.get("metadata", {}) or {}
        facts = {
            "symbol": sig["symbol"],
            "action": sig["action"],
            "outcome": outcome,
            "entry": sig.get("price"),
            "tp": sig.get("tp"),
            "sl": sig.get("sl"),
            "signal_time_utc": sig.get("sent_at"),
            "resolution_reason": meta.get("resolution_reason"),
            "hold_minutes": meta.get("resolution_minutes"),
            "max_favorable_pct": meta.get("max_favorable_pct"),
            "max_adverse_pct": meta.get("max_adverse_pct"),
            "signal_mode": sig.get("mode", "intraday"),
        }
        user = (
            f"Trade facts:\n{json.dumps(facts, default=str)}\n\n"
            f"Recent headlines for {sig['symbol']}:\n"
            + ("\n".join(f"- {h}" for h in headlines) if headlines else "(none available)")
        )

        result = llm_complete("postmortem", user=user, system=_SYSTEM,
                              want_json=True, max_tokens=300)
        if not result or result.get("cause") not in CAUSES:
            if result:
                logger.warning(f"[postmortem] invalid cause for signal {sig.get('id')}: "
                               f"{result.get('cause')}")
            return

        from database import save_postmortem
        provider, model = resolved_model("postmortem")
        save_postmortem(
            signal_id=sig["id"], symbol=sig["symbol"], outcome=outcome,
            cause=result["cause"],
            market_related=bool(result.get("market_related")),
            earnings_related=bool(result.get("earnings_related")),
            explanation=str(result.get("explanation", ""))[:200],
            provider=provider, model=model,
        )
        logger.info(f"[postmortem] {sig['symbol']} {outcome}: {result['cause']}")
    except Exception as e:
        logger.error(f"[postmortem] failed for signal {sig.get('id')}: {e}")
