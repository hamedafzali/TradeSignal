# LLM Components — Evaluation Criteria

Added 2026-07-02. Every LLM/guard component ships with a measurable success
criterion and a kill criterion. **Scheduled review: ~2026-08-15** (or earlier
once thresholds below are reached). The weekly ML health report carries the
running numbers; this file is the contract for judging them.

Principles: the LLM is an operational component, never a price predictor.
Components must be measured against their own hypothetical/tagged data, and
any strategy change they suggest still goes through the backtest gate.

---

## 1. Earnings guard (`_near_earnings`, bot.py)

Suppresses signals within 3 days of an earnings report. Suppressed signals
are logged to `shadow_signals` with `source='earnings_guard'` and resolved
hypothetically like real signals.

| | Criterion |
|---|---|
| **Data needed** | ≥ 15 resolved earnings-guard shadow signals |
| **KEEP** | Hypothetical win rate of suppressed signals **below** the live win rate of the same period (the guard is filtering out worse-than-average trades) |
| **KILL** | Hypothetical win rate **at or above** live win rate at n ≥ 15 (the guard suppresses trades as good as the ones we take — remove it) |
| **Tune** | Window is 3 days (`_EARNINGS_WINDOW_DAYS`); swing signals may need a wider window (holds span ~10 trading days). Revisit if swing signals go live. |

Query: `get_shadow_stats(source='earnings_guard')` vs `get_weekly_stats()`.

## 2. Post-mortem tagging (postmortem.py → `trade_postmortems`)

One structured cause tag per resolved trade, with news headlines as context.

| | Criterion |
|---|---|
| **Data needed** | ≥ 30 tagged losses |
| **SUCCESS** | Any single cause ≥ 25% of tagged losses → design a filter for it and backtest it (e.g. `market_wide_move` cluster → tighten regime filter) |
| **QUALITY CHECK** | Spot-check 10 random tags against the actual trades. If > 3 are clearly wrong, tighten the prompt or upgrade the model before trusting clusters |
| **KILL** | After 60+ tags, distribution ≈ uniform or dominated by `unclear` (> 50%) → tags carry no signal; stop paying for them |

Query: `get_postmortem_stats()`.

## 3. Weekly AI analyst (ml_health_report, bot.py)

One reasoning pass over the weekly numbers, appended to the Monday report.

| | Criterion |
|---|---|
| **SUCCESS** | Its observations are correct (verifiable against the data) and at least occasionally surface something the raw numbers didn't make obvious; suggested experiments are sane and backtestable |
| **KILL** | Consistently generic, wrong about the data, or overconfident on tiny samples despite the prompt → drop the section (keep the raw stats) |

Judgment call by the operator/Claude at review time — read 4+ consecutive
Monday reports before deciding.

## 4. Cost & reliability (all components, `llm_calls` table)

| | Criterion |
|---|---|
| **Budget** | Total LLM spend < €5/month at current volumes (expected: well under €1 on Mistral Small/Large) |
| **Reliability** | ok-rate ≥ 90% per task. Persistent parse failures → fix prompt or switch model |

Query: `get_llm_usage_stats(days=30)` — tokens by task; multiply by provider
prices for spend.

---

## Review procedure (for future-Claude)

1. Pull the four queries above on the server.
2. Score each component against its table.
3. KILL criteria met → remove the component (they are all cleanly removable:
   settings `earnings_guard_enabled=false`, `llm_provider=off`).
4. SUCCESS criteria met for post-mortems → design the indicated filter and
   run it through `backtest_swing.py`-style gating before enabling live.
5. Update this file with findings and set the next review date.
