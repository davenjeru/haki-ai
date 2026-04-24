"""
Supervisor router \u2014 tier 1 of the two-tier agent system.

Given the conversation so far, a single Haiku call returns a JSON object
like:
    {"agents": ["employment"], "reason": "labour question"}

The supervisor never produces the user-facing answer itself; it only
decides who handles the turn. A dedicated router prompt (prompts.SUPERVISOR_PROMPT)
constrains the output to a fixed set of agent names so the caller can rely
on a canonical routing list.

Failure modes:
  - Malformed JSON           \u2192 default to ["chat"]. ``chat`` replies
                                conversationally and bilingually-refuses
                                off-topic turns, so it is the safest
                                no-op route when we can't trust the model's
                                output. Previously this fell back to
                                ``["employment"]`` which produced
                                hallucinated statute answers for
                                conversational turns (see "Naitwa nani?"
                                regression in backend/tests/test_unit.py).
  - Empty agents array       \u2192 same fallback.
  - Unknown agent name       \u2192 dropped from the routing list.

The router is intentionally stateless \u2014 the only information it needs is
the last few turns of the conversation. We truncate to the same window the
classifier uses (prompts.CLASSIFIER_PROMPT) so the model has comparable
context in both codepaths.
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from prompts import SUPERVISOR_PROMPT


# Canonical agent names the supervisor may select. Anything else is dropped.
# Kept in lockstep with ``agents.specialists.AGENT_REGISTRY`` \u2014 adding a new
# domain specialist requires an entry in both places.
KNOWN_AGENTS = frozenset({
    "constitution",
    "employment",
    "land",
    "criminal",
    "family",
    "contracts",
    "chat",
})

# Fallback when the model returns nothing we can use. ``chat`` is the only
# safe default: it replies conversationally / bilingually-refuses for
# off-topic questions without pulling statute citations that may be
# irrelevant to the user's actual intent. Defaulting to a statute specialist
# here used to cause hallucinated "employment" answers for conversational
# Swahili turns like ``Naitwa nani?`` whenever the Bedrock routing call
# flaked. See backend/tests/test_unit.py::TestSupervisorFallback.
_FALLBACK = ["chat"]

_MAX_CONTEXT_CHARS = 4000
_RECENT_TURN_LIMIT = 6
_MAX_AGENTS = 3

# One retry before falling back. Haiku 4.5 has a 10 RPM account quota and
# transient ``ThrottlingException``s were showing up in the "Naitwa nani?"
# trace, so a cheap retry meaningfully reduces the fallback rate without
# materially increasing p95 latency on the happy path.
_BEDROCK_RETRY_ATTEMPTS = 2
_BEDROCK_RETRY_BASE_DELAY_S = 0.3
_BEDROCK_RETRY_MAX_DELAY_S = 1.5


def _format_transcript(messages: list[dict]) -> str:
    """Renders the last N turns as a `U:/A:` transcript (matches classifier)."""
    tail = messages[-_RECENT_TURN_LIMIT:]
    lines: list[str] = []
    for m in tail:
        role = "U" if m.get("role") == "user" else "A"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    transcript = "\n".join(lines)
    if len(transcript) > _MAX_CONTEXT_CHARS:
        transcript = transcript[-_MAX_CONTEXT_CHARS:]
    return transcript


def _parse_routing(text: str) -> tuple[list[str], str]:
    """Pulls the JSON object out of the model reply and normalises the output."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return list(_FALLBACK), "parse_failure"
    try:
        obj: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError:
        return list(_FALLBACK), "parse_failure"

    raw_agents = obj.get("agents") or []
    if not isinstance(raw_agents, list):
        return list(_FALLBACK), "parse_failure"

    agents = [a for a in raw_agents if isinstance(a, str) and a in KNOWN_AGENTS]
    if not agents:
        return list(_FALLBACK), "unknown_agents"

    # "chat" is exclusive \u2014 never combine with statute specialists.
    if "chat" in agents:
        agents = ["chat"]

    # Cap at 3 to keep fan-out bounded.
    agents = agents[:_MAX_AGENTS]

    reason = obj.get("reason") if isinstance(obj.get("reason"), str) else ""
    return agents, reason or "routed"


def route_supervisor(
    messages: list[dict],
    bedrock_runtime,
    model_id: str,
) -> tuple[list[str], str]:
    """
    Returns (selected_agents, reason) for the latest user turn.

    Defensive semantics: any Bedrock failure returns the fallback routing
    so the pipeline degrades gracefully. The caller is responsible for
    emitting an observability event when the fallback kicks in.
    """
    transcript = _format_transcript(messages)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 80,
        "system": SUPERVISOR_PROMPT,
        "messages": [{"role": "user", "content": transcript or "(empty)"}],
    }

    last_err: Exception | None = None
    for attempt in range(_BEDROCK_RETRY_ATTEMPTS):
        try:
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            break
        except Exception as err:  # noqa: BLE001 - we log and degrade gracefully
            last_err = err
            if attempt + 1 < _BEDROCK_RETRY_ATTEMPTS:
                # Exponential backoff with a small amount of jitter so
                # simultaneous retries across concurrent Lambda invocations
                # don't stampede the 10 RPM Haiku quota in lockstep.
                delay = min(
                    _BEDROCK_RETRY_BASE_DELAY_S * (2 ** attempt),
                    _BEDROCK_RETRY_MAX_DELAY_S,
                )
                time.sleep(delay + random.uniform(0, delay / 2))
                continue
            print(
                f"[supervisor] Bedrock call failed after {attempt + 1} "
                f"attempt(s), using fallback route {_FALLBACK}: {err}"
            )
            return list(_FALLBACK), "bedrock_error"
    else:  # pragma: no cover - loop always breaks or returns above
        return list(_FALLBACK), "bedrock_error"

    result = json.loads(response["body"].read())
    text = (result.get("content", [{}]) or [{}])[0].get("text", "")
    return _parse_routing(text)
