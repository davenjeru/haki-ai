"""
Specialist agents \u2014 tier 2 of the two-tier system.

Each specialist wraps the Phase 1 advanced-RAG pipeline with a corpus
filter and a display name. Specialists share the same build function so
adding a new statute (e.g. Marriage Act) is a one-line addition to
`AGENT_REGISTRY`.

Agents:
  - constitution  \u2192 filter(source=\"Constitution of Kenya 2010\")
  - employment    \u2192 filter(source=\"Employment Act 2007\")
  - land          \u2192 filter(source=\"Land Act 2012\")
  - faq           \u2192 filter(corpus=\"faq\") \u2014 Phase 4a populates this corpus
                    from SheriaPlex Q&As + KenyaLaw case summaries. Until
                    the FAQ corpus exists, the agent falls back to the
                    full KB unfiltered so it still produces answers.
  - chat          \u2192 does NOT use RAG; runs chat_node.invoke_chat.

A specialist call returns:
    {
      \"agent\":     \"employment\",
      \"text\":      \"The Employment Act 2007 Section 40...\",
      \"citations\": [ {source, chapter, section, title, chunkId, pageImageKey?}, ... ],
      \"blocked\":   False,
    }

The pipeline orchestrator in `graph.py` appends specialist outputs to
state via a custom reducer (`_specialist_outputs_reducer`). The reducer
concatenates within a turn but treats ``None`` as a reset sentinel that
the supervisor emits each turn \u2014 this prevents the checkpointer from
carrying stale outputs from prior turns into the current synthesis.
"""

from __future__ import annotations

from agents.chat import invoke_chat
from prompts import BILINGUAL_REFUSAL, build_system_prompt
from rag import (
    blocked_response,
    check_guardrail_block,
    extract_citations,
    run_rag,
)


# Substrings of BILINGUAL_REFUSAL that uniquely identify a model-emitted
# out-of-scope refusal. We can't do an exact-equality check because the
# model occasionally appends trailing whitespace / punctuation, and chat
# agent output is composed from tokens streamed by Bedrock.
_REFUSAL_MARKERS = (
    "Mimi ni msaidizi wa kisheria wa Kenya tu",
    "I can only help with Kenyan legal matters",
)


def _is_model_emitted_refusal(text: str) -> bool:
    """
    Returns True iff ``text`` contains the canonical bilingual refusal.

    The synthesizer treats ``blocked=True`` specialists as a hard-stop
    fast-path (no merging with other specialists, no citations). Without
    this detection, a chat agent that correctly refused an off-topic turn
    would still be merged with e.g. the previous turn's employment answer,
    surfacing confusing mixed output. See the "Naitwa nani?" trace for the
    exact failure mode this guards against.
    """
    if not text:
        return False
    return all(marker in text for marker in _REFUSAL_MARKERS)


# Registry mapping agent name \u2192 (display_name, retrieval_filter).
# `retrieval_filter` is applied to BOTH dense and BM25 retrieval so the
# specialist only sees chunks from its statute/corpus.

AGENT_REGISTRY: dict[str, dict] = {
    "constitution": {
        "display_name": "Constitution",
        "filter": {"source": "Constitution of Kenya 2010"},
    },
    "employment": {
        "display_name": "Employment",
        "filter": {"source": "Employment Act 2007"},
    },
    "land": {
        "display_name": "Land",
        "filter": {"source": "Land Act 2012"},
    },
    "faq": {
        "display_name": "FAQ",
        # Phase 4a populates faq-chunks/ with crawled SheriaPlex Q&A and
        # KenyaLaw case summaries, all tagged corpus="faq" in S3 sidecar
        # metadata. Filtering retrieval on corpus="faq" keeps this
        # specialist focused on lay/procedural answers while the statute
        # specialists handle Act/Constitution lookups via `source` filters.
        "filter": {"corpus": "faq"},
    },
    "chat": {
        "display_name": "Chat",
        "filter": None,  # unused \u2014 chat agent never queries the KB
    },
}


def _specialist_output(
    agent: str,
    *,
    text: str,
    citations: list[dict],
    blocked: bool = False,
) -> dict:
    return {
        "agent": agent,
        "text": text,
        "citations": citations,
        "blocked": blocked,
    }


def build_specialist(
    agent: str,
    *,
    rag_adapter,
    bedrock_runtime,
    model_id: str,
    s3_client,
    s3_bucket: str,
):
    """
    Returns a callable `specialist(state) -> dict` suitable for use as a
    LangGraph node. The closure captures the runtime dependencies so graph
    construction stays clean.
    """
    if agent not in AGENT_REGISTRY:
        raise ValueError(f"unknown agent: {agent!r} (known: {sorted(AGENT_REGISTRY)})")

    config = AGENT_REGISTRY[agent]
    retrieval_filter = config["filter"]

    def _run(state: dict) -> dict:
        # Don't run unless the supervisor selected this agent.
        if agent not in (state.get("selected_agents") or []):
            return {}

        language = state.get("language", "english")
        # Find the latest user message in the state.
        user_msg = ""
        for m in reversed(state.get("messages", []) or []):
            role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
            if role in ("user", "human"):
                user_msg = (
                    m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                ) or ""
                break

        if agent == "chat":
            text = invoke_chat(
                [
                    {
                        "role": m.get("role") if isinstance(m, dict) else {"human": "user", "ai": "assistant"}.get(
                            getattr(m, "type", None), ""
                        ),
                        "content": (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or "",
                    }
                    for m in (state.get("messages") or [])
                ],
                language,
                bedrock_runtime,
                model_id,
            )
            return {"specialist_outputs": [_specialist_output(
                agent,
                text=text,
                citations=[],
                blocked=_is_model_emitted_refusal(text),
            )]}

        system_prompt = build_system_prompt(language)
        result = run_rag(
            query=user_msg,
            system_prompt=system_prompt,
            model_id=model_id,
            rag_adapter=rag_adapter,
            metadata_filter=retrieval_filter,
        )
        if check_guardrail_block(result):
            return {
                "specialist_outputs": [_specialist_output(
                    agent,
                    text=blocked_response(language),
                    citations=[],
                    blocked=True,
                )]
            }

        citations = extract_citations(result, s3_client=s3_client, bucket=s3_bucket)
        text = result.get("output", {}).get("text", "")
        # A statute specialist that declines the turn (e.g. the user's
        # question is genuinely off-topic for this statute) emits the
        # bilingual refusal verbatim per the prompt. Flag it so the
        # synthesizer's blocked fast-path handles it, drops citations,
        # and doesn't interleave this specialist with a peer that happens
        # to have an answer.
        refusal = _is_model_emitted_refusal(text)
        return {
            "specialist_outputs": [_specialist_output(
                agent,
                text=text,
                citations=[] if refusal else citations,
                blocked=refusal,
            )]
        }

    _run.__name__ = f"{agent}_agent"
    return _run
