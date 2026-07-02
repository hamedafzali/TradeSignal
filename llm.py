"""
Provider-agnostic LLM layer.

The LLM is an operational component (trade post-mortems, weekly analysis) —
never a price predictor. Providers are swappable via the `llm_provider`
setting ('mistral' default, 'anthropic', or 'off'); per-task models via
`llm_model_<task>` settings. API keys come from the environment
(MISTRAL_API_KEY / ANTHROPIC_API_KEY, set through the Container Manager UI).

Everything degrades gracefully: with no key configured, llm_complete()
returns None and the app runs exactly as before. Every call is logged to the
llm_calls table (tokens, latency, ok) so the component can be evaluated on
cost and reliability — see docs/llm-evaluation.md.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# task -> default model per provider. Tagging is a cheap classification job;
# the weekly analyst is the one reasoning task worth a stronger model.
_DEFAULT_MODELS = {
    "mistral": {
        "postmortem": "mistral-small-latest",
        "analyst": "mistral-large-latest",
    },
    "anthropic": {
        "postmortem": "claude-haiku-4-5",
        "analyst": "claude-opus-4-8",
    },
}

_API_KEYS = {
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_warned_disabled = False


def _setting(key: str, default: str = "") -> str:
    try:
        from database import get_setting
        return get_setting(key, default)
    except Exception:
        return default


def is_enabled() -> bool:
    provider = _setting("llm_provider", "mistral").lower()
    if provider in ("off", "disabled", ""):
        return False
    return bool(os.getenv(_API_KEYS.get(provider, ""), "").strip())


def resolved_model(task: str) -> tuple[str, str]:
    """(provider, model) that llm_complete would use for this task."""
    provider = _setting("llm_provider", "mistral").lower()
    model = _setting(f"llm_model_{task}",
                     _DEFAULT_MODELS.get(provider, {}).get(task, ""))
    return provider, model


def _post_json(url: str, headers: dict, body: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _call_mistral(model: str, system: str, user: str,
                  want_json: bool, max_tokens: int) -> tuple[str, int, int]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": user}],
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
    data = _post_json(
        "https://api.mistral.ai/v1/chat/completions",
        {"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY'].strip()}"},
        body,
    )
    usage = data.get("usage", {})
    return (data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def _call_anthropic(model: str, system: str, user: str,
                    want_json: bool, max_tokens: int) -> tuple[str, int, int]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        body["system"] = system
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"].strip(),
         "anthropic-version": "2023-06-01"},
        body,
    )
    text = "".join(b.get("text", "") for b in data.get("content", [])
                   if b.get("type") == "text")
    usage = data.get("usage", {})
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def _extract_json(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating fences/preamble."""
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return None


def llm_complete(task: str, user: str, system: str = "",
                 want_json: bool = False, max_tokens: int = 1000):
    """
    Run one LLM call for a named task. Returns parsed dict (want_json=True),
    text, or None when disabled/failed. Never raises.
    """
    global _warned_disabled
    if not is_enabled():
        if not _warned_disabled:
            logger.info("[llm] no provider configured — LLM features dormant "
                        "(set MISTRAL_API_KEY or ANTHROPIC_API_KEY)")
            _warned_disabled = True
        return None

    provider = _setting("llm_provider", "mistral").lower()
    model = _setting(f"llm_model_{task}",
                     _DEFAULT_MODELS.get(provider, {}).get(task, ""))
    if not model:
        logger.warning(f"[llm] no model for task '{task}' on provider '{provider}'")
        return None

    t0 = time.time()
    ok, in_tok, out_tok, result = False, 0, 0, None
    response_text = None
    request_text = (f"[system]\n{system}\n\n[user]\n{user}" if system
                    else f"[user]\n{user}")
    try:
        call = _call_mistral if provider == "mistral" else _call_anthropic
        text, in_tok, out_tok = call(model, system, user, want_json, max_tokens)
        response_text = text
        result = _extract_json(text) if want_json else text
        ok = result is not None
        if not ok:
            logger.warning(f"[llm] {task}: unparseable JSON response")
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:500]
        response_text = f"[HTTP {e.code}] {err}"
        logger.error(f"[llm] {task} HTTP {e.code}: {err[:200]}")
    except Exception as e:
        response_text = f"[error] {e}"
        logger.error(f"[llm] {task} failed: {e}")

    try:
        from database import log_llm_call
        log_llm_call(task, provider, model, in_tok, out_tok, ok,
                     round((time.time() - t0) * 1000),
                     request_text=request_text, response_text=response_text)
    except Exception:
        pass
    return result
