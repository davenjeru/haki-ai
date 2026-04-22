"""
LangSmith observability bootstrap + per-turn tracing wrapper.

LangSmith and CloudWatch play complementary roles:
  - CloudWatch: aggregate metrics (latency p95, error rate), alarms, SNS.
  - LangSmith: per-invocation traces with the full LangGraph node tree,
    Bedrock InvokeModel prompts/completions, token counts, and routing
    decisions. Used for debugging quality and drift.

LangChain/LangGraph auto-trace every graph node and LLM call when
LANGSMITH_TRACING=true + LANGSMITH_API_KEY are set in the environment.
This module:

  1. Fetches the API key from SSM Parameter Store at Lambda cold start
     (idempotent across warm invocations). On any failure it disables
     tracing rather than crashing the request — observability must not
     be a hard dependency of the app path.

  2. Wraps graph.invoke with @traceable(name="haki_turn") so each request
     has a single root span carrying rich metadata attached AFTER the
     graph returns (session_id, language, needs_rag, blocked, etc.).

Imports of this module MUST happen before `from graph import ...` so
LANGSMITH_API_KEY is set before LangChain initialises its tracer.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Process-level flag: the SSM round-trip happens exactly once per container.
_BOOTSTRAPPED = False


def bootstrap_langsmith(config) -> None:
    """
    Cold-start-only: if a LANGSMITH_API_KEY_SSM_PARAMETER is configured,
    fetch the key from SSM and set os.environ["LANGSMITH_API_KEY"]. No-op
    if tracing is already bootstrapped, if no parameter is configured, or
    if the API key is already set directly in the environment (local dev).
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # Direct env-var wins (local dev via .env file). Nothing to fetch.
    if os.environ.get("LANGSMITH_API_KEY"):
        _BOOTSTRAPPED = True
        return

    param_name = getattr(config, "langsmith_ssm_parameter", "") or ""
    if not param_name:
        _BOOTSTRAPPED = True
        return

    try:
        from clients import make_ssm

        ssm = make_ssm(config)
        resp = ssm.get_parameter(Name=param_name, WithDecryption=True)
        value = resp.get("Parameter", {}).get("Value", "")
        if not value:
            raise RuntimeError(f"SSM parameter {param_name} is empty")
        os.environ["LANGSMITH_API_KEY"] = value
    except Exception as err:
        # Tracing is best-effort. Disable it so LangChain becomes a no-op
        # and the request path stays unaffected by observability failures.
        log.warning("LangSmith bootstrap failed; disabling tracing: %s", err)
        os.environ["LANGSMITH_TRACING"] = "false"
    finally:
        _BOOTSTRAPPED = True


def _tracing_enabled() -> bool:
    # LangChain accepts either name; we honour both so a user who set
    # only the legacy variable in their .env still gets traces.
    flag = (
        os.environ.get("LANGSMITH_TRACING", "")
        or os.environ.get("LANGCHAIN_TRACING_V2", "")
    ).lower()
    return flag == "true" and bool(os.environ.get("LANGSMITH_API_KEY"))


def run_traced_turn(
    graph,
    initial_state: dict,
    graph_config: dict,
    *,
    session_id: str,
    env: str,
    message_length: int,
) -> dict:
    """
    Wraps `graph.invoke(initial_state, config=graph_config)` in a single
    top-level LangSmith run called "haki_turn" and attaches rich metadata
    to the root span AFTER the graph returns (language, needs_rag,
    blocked, citations_count).

    Falls through to plain graph.invoke when tracing is disabled — nothing
    observable to attach to, so the @traceable overhead is wasted.
    """
    if not _tracing_enabled():
        return graph.invoke(initial_state, config=graph_config)

    # Local imports so the module can be used in unit tests that have no
    # langsmith installed / configured without paying the import cost.
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree

    @traceable(name="haki_turn", tags=[f"env:{env}", "haki-ai"])
    def _invoke(user_session_id: str, initial_message_length: int) -> dict:
        # Parameters show up as the trace's "inputs" in LangSmith so you
        # can search/filter by session from the UI.
        final_state = graph.invoke(initial_state, config=graph_config)

        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata(_trace_metadata(final_state, user_session_id, env, initial_message_length))
        return final_state

    return _invoke(session_id, message_length)


def _trace_metadata(
    final_state: dict, session_id: str, env: str, message_length: int
) -> dict[str, Any]:
    """
    Shape of the metadata attached to every haki_turn root span. Keeping
    this a pure function makes it unit-testable without LangSmith.
    """
    citations = final_state.get("citations") or []
    return {
        "session_id": session_id,
        "env": env,
        "message_length": message_length,
        "language": final_state.get("language"),
        "needs_rag": final_state.get("needs_rag"),
        "blocked": bool(final_state.get("blocked", False)),
        "citations_count": len(citations),
    }
