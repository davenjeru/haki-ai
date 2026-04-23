"""
LangGraph orchestration for Haki AI (Phase 2 two-tier multi-agent).

                         +------------------+
                         |  detect_language |
                         +--------+---------+
                                  |
                                  v
                         +------------------+
                         |    supervisor    |  Haiku routing \u2192 selected_agents
                         +--------+---------+
                                  |
           +----------+-----------+------+--------+
           |          |           |              |
           v          v           v              v
      constitution  employment   land           faq       chat
      specialist    specialist   specialist     specialist agent
           \\         |           |              /          |
            \\________+___________+_____________/           |
                            |                              |
                            v                              v
                     +-------------+               +--------------+
                     | synthesizer |               | synthesizer  |
                     +------+------+               +------+-------+
                            |                             |
                            +--------------+--------------+
                                           |
                                           v
                                       [ END ]

- Parallel fan-out is implemented via conditional edges that Send() the
  supervisor\u2019s state to every selected specialist. Only selected
  specialists run; others are skipped by the dispatcher.
- Results are accumulated into `specialist_outputs: list[dict]` via the
  `operator.add` reducer (LangGraph native list concat).
- The `synthesizer` node merges multi-specialist outputs (or passes a
  single specialist through verbatim) into the final response_text +
  citations + blocked flag.
- For the chat path, a single `chat` specialist runs and the synthesizer
  simply passes its output through \u2014 no extra LLM call.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from clients.adapters import BedrockRAGAdapter, ComprehendAdapter, LocalRAGAdapter
from agents import build_specialist, route_supervisor, synthesize
from agents.specialists import AGENT_REGISTRY
from memory.checkpointer import DynamoDBSaver
from clients import (
    make_bedrock_agent_runtime,
    make_bedrock_runtime,
    make_comprehend,
    make_dynamodb_table,
    make_s3,
    make_sagemaker_runtime,
)
from rag import refresh_presigned_urls


_VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), ".local-vectorstore")

# Match handler.detect_language thresholds so behaviour is identical.
_DOMINANCE_THRESHOLD = 0.85


# ── State ─────────────────────────────────────────────────────────────────────

def _specialist_outputs_reducer(
    existing: list[dict] | None,
    incoming: list[dict] | None,
) -> list[dict]:
    """
    Reducer for ``specialist_outputs``.

    Within a turn we fan out to multiple specialists in parallel and each
    branch returns a one-item list. The natural reducer for that is list
    concatenation (``operator.add``) \u2014 which is what we used to have.

    But the LangGraph checkpointer persists ``AgentState`` across turns
    within a thread, so a plain-concat reducer silently accumulates every
    specialist output from every prior turn. By turn N the synthesizer saw
    N-1 stale outputs plus the current turn's and merged them into nonsense
    (see the ``Naitwa nani?`` regression: ``specialist_outputs`` contained
    both the previous COVID-termination answer *and* the current turn's
    bilingual refusal, and the synthesizer happily blended them).

    This reducer fixes that by treating ``None`` as a per-turn reset
    sentinel. The supervisor node emits ``None`` at the start of each turn;
    specialist branches still return one-item lists that get appended as
    before.
    """
    if incoming is None:
        return []
    return (existing or []) + incoming


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], add_messages]
    language: str
    # Phase 2 \u2014 supervisor output
    selected_agents: list[str]
    routing_reason: str
    # See ``_specialist_outputs_reducer`` above for why this is not a plain
    # ``operator.add`` reducer. The supervisor emits ``None`` each turn to
    # reset the list before fan-out begins.
    specialist_outputs: Annotated[list[dict], _specialist_outputs_reducer]
    # Phase 2 \u2014 final merged output (written by synthesizer)
    citations: list[dict]
    blocked: bool
    response_text: str
    # Back-compat field some tests read. Derived from the synthesizer output:
    # True iff the synthesizer produced a non-chat answer with citations.
    needs_rag: bool


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_language(text: str, comprehend: ComprehendAdapter) -> str:
    languages = comprehend.detect_dominant_language(text)
    scores = {lang["LanguageCode"]: lang["Score"] for lang in languages}
    top = languages[0] if languages else {}
    top_code = top.get("LanguageCode")
    top_score = top.get("Score", 0.0)
    if top_code == "en" and top_score >= _DOMINANCE_THRESHOLD:
        return "english"
    if top_code == "sw" and top_score >= _DOMINANCE_THRESHOLD:
        return "swahili"
    if "en" in scores and "sw" in scores:
        return "mixed"
    if top_code == "sw":
        return "swahili"
    return "english"


def _normalize_message(m) -> dict:
    if isinstance(m, dict):
        return {"role": m.get("role", ""), "content": m.get("content", "") or ""}
    msg_type = getattr(m, "type", None)
    role = {"human": "user", "ai": "assistant"}.get(msg_type, msg_type or "")
    content = getattr(m, "content", "") or ""
    return {"role": role, "content": content}


def _message_metadata(m) -> dict:
    if isinstance(m, dict):
        return {}
    return dict(getattr(m, "additional_kwargs", {}) or {})


def _message_id(m) -> str | None:
    if isinstance(m, dict):
        return m.get("id")
    return getattr(m, "id", None)


def _as_role_content(messages: list) -> list[dict]:
    return [_normalize_message(m) for m in messages]


def _latest_user_message(messages: list) -> str:
    for m in reversed(_as_role_content(messages)):
        if m["role"] == "user":
            return (m["content"] or "").strip()
    return ""


def _make_rag_adapter(
    config,
    *,
    s3_client,
    bedrock_runtime,
    bedrock_agent_runtime,
    sagemaker_runtime=None,
):
    if config.is_local:
        return LocalRAGAdapter(
            bedrock_runtime=bedrock_runtime,
            bedrock_agent_runtime=bedrock_agent_runtime,
            embed_model=config.embedding_model_id,
            vectorstore_path=_VECTORSTORE_PATH,
            s3_client=s3_client,
            s3_bucket=config.s3_bucket,
            aws_region=config.aws_region,
            chroma_host=config.chroma_host,
            chroma_port=config.chroma_port,
            guardrail_id=config.guardrail_id,
            guardrail_version=config.guardrail_version,
            sagemaker_runtime=sagemaker_runtime,
            sagemaker_endpoint_name=config.sagemaker_endpoint_name,
            use_finetuned_model=config.use_finetuned_model,
        )
    return BedrockRAGAdapter(
        bedrock_agent_runtime=bedrock_agent_runtime,
        bedrock_runtime=bedrock_runtime,
        config=config,
        s3_client=s3_client,
        sagemaker_runtime=sagemaker_runtime,
    )


def _assistant_message(
    *, content: str, citations: list[dict], language: str, blocked: bool
) -> AIMessage:
    """
    Builds the AIMessage we persist via LangGraph\u2019s DynamoDB checkpointer.
    Stripping pageImageUrl keeps the stored shape small (and presigned URLs
    expire anyway \u2014 load_history re-signs each citation on read).
    """
    persistable_citations = [
        {k: v for k, v in c.items() if k != "pageImageUrl"}
        for c in citations
    ]
    return AIMessage(
        content=content,
        id=str(uuid.uuid4()),
        additional_kwargs={
            "citations": persistable_citations,
            "language": language,
            "blocked": blocked,
        },
    )


# ── Nodes ────────────────────────────────────────────────────────────────────

def _build_nodes(config):
    """
    Builds and returns the node callables. All AWS clients are instantiated
    once here and closed over by the node closures so repeated graph.invoke
    calls reuse warm clients.
    """
    comprehend = ComprehendAdapter(make_comprehend(config), config.is_local)
    bedrock_runtime = make_bedrock_runtime(config)
    bedrock_agent_runtime = make_bedrock_agent_runtime(config)
    s3 = make_s3(config)
    sagemaker_runtime = (
        make_sagemaker_runtime(config)
        if config.use_finetuned_model and config.sagemaker_endpoint_name
        else None
    )
    rag_adapter = _make_rag_adapter(
        config,
        s3_client=s3,
        bedrock_runtime=bedrock_runtime,
        bedrock_agent_runtime=bedrock_agent_runtime,
        sagemaker_runtime=sagemaker_runtime,
    )

    def detect_language_node(state: AgentState) -> dict:
        user_msg = _latest_user_message(state.get("messages", []))
        return {"language": _detect_language(user_msg, comprehend)}

    def supervisor_node(state: AgentState) -> dict:
        selected, reason = route_supervisor(
            _as_role_content(state.get("messages", [])),
            bedrock_runtime,
            config.bedrock_model_id,
        )
        return {
            "selected_agents": selected,
            "routing_reason": reason,
            # Also set needs_rag for back-compat metadata + handler paths.
            "needs_rag": selected != ["chat"],
            # Per-turn reset sentinel. Consumed by
            # ``_specialist_outputs_reducer`` to clear stale outputs from
            # the previous turn that the checkpointer would otherwise
            # carry forward into this turn's synthesis.
            "specialist_outputs": None,
        }

    specialist_nodes = {
        agent: build_specialist(
            agent,
            rag_adapter=rag_adapter,
            bedrock_runtime=bedrock_runtime,
            model_id=config.bedrock_model_id,
            s3_client=s3,
            s3_bucket=config.s3_bucket,
        )
        for agent in AGENT_REGISTRY
    }

    def synthesizer_node(state: AgentState) -> dict:
        outputs = state.get("specialist_outputs") or []
        language = state.get("language", "english")
        merged = synthesize(outputs, language, bedrock_runtime, config.bedrock_model_id)
        # Re-presign citation URLs so the frontend gets fresh links each turn.
        fresh_citations = refresh_presigned_urls(
            merged["citations"], s3_client=s3, bucket=config.s3_bucket,
        )
        return {
            "response_text": merged["text"],
            "citations": fresh_citations,
            "blocked": merged["blocked"],
            "messages": [_assistant_message(
                content=merged["text"],
                citations=fresh_citations,
                language=language,
                blocked=merged["blocked"],
            )],
        }

    return {
        "detect_language": detect_language_node,
        "supervisor": supervisor_node,
        "synthesizer": synthesizer_node,
        **specialist_nodes,
    }


# ── Graph wiring ─────────────────────────────────────────────────────────────

def _dispatch_to_specialists(state: AgentState) -> list:
    """
    Fan-out from supervisor \u2192 selected specialists. Uses LangGraph\u2019s `Send`
    so only the chosen specialists actually execute (skipped agents never
    enter the runtime). The value passed in each Send is the full state so
    the specialist has access to messages + language.
    """
    agents = state.get("selected_agents") or []
    if not agents:
        agents = ["chat"]
    return [Send(a, dict(state)) for a in agents]


def build_graph(config, checkpointer):
    """
    Constructs and compiles the two-tier multi-agent graph. Broken out from
    `get_compiled_graph` so tests can inject custom checkpointers and verify
    fan-out / synthesizer behaviour without touching DynamoDB.
    """
    nodes = _build_nodes(config)
    builder = StateGraph(AgentState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "supervisor")

    # Fan-out: supervisor dispatches to selected specialists via Send().
    builder.add_conditional_edges(
        "supervisor",
        _dispatch_to_specialists,
        list(AGENT_REGISTRY),
    )

    # Fan-in: every specialist feeds the synthesizer. LangGraph waits for
    # all dispatched branches to complete before executing synthesizer.
    for agent in AGENT_REGISTRY:
        builder.add_edge(agent, "synthesizer")

    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer)


_compiled_graph = None


def get_compiled_graph(config):
    global _compiled_graph
    if _compiled_graph is None:
        table = make_dynamodb_table(config, config.checkpoints_table)
        checkpointer = DynamoDBSaver(table)
        _compiled_graph = build_graph(config, checkpointer)
    return _compiled_graph


# ── History hydration ────────────────────────────────────────────────────────

def load_history(config, session_id: str) -> list[dict]:
    """
    Reconstructs the persisted conversation for a given sessionId. See the
    original implementation for the response shape; that contract is
    unchanged by the Phase 2 graph rewrite.
    """
    graph = get_compiled_graph(config)
    state = graph.get_state({"configurable": {"thread_id": session_id}})
    if not state or not state.values:
        return []
    messages = state.values.get("messages", []) or []

    s3 = make_s3(config)
    out: list[dict] = []
    for m in messages:
        normalized = _normalize_message(m)
        content = normalized["content"]
        if not content:
            continue
        msg: dict = {
            "id": _message_id(m) or str(uuid.uuid4()),
            "role": normalized["role"],
            "content": content,
        }
        meta = _message_metadata(m)
        if normalized["role"] == "assistant":
            stored = meta.get("citations") or []
            msg["citations"] = refresh_presigned_urls(
                stored, s3_client=s3, bucket=config.s3_bucket,
            )
            if meta.get("language"):
                msg["language"] = meta["language"]
            if meta.get("blocked"):
                msg["blocked"] = True
        out.append(msg)
    return out
