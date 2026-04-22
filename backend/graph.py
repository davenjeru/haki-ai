"""
LangGraph orchestration for Haki AI.

Replaces the linear step-by-step handler flow with a compiled StateGraph:

    START
      → detect_language
      → classify_intent         (Haiku: needs_rag? true/false)
      → [ rag_node | chat_node ]
      → END

The compiled graph is cached at module level and reused across invocations
within the same Lambda execution context (warm container).

Memory:
  Checkpointer = DynamoDBSaver. Keyed by thread_id = sessionId from the
  frontend. Persists across cold starts and process restarts.

Routing:
  classify_intent reads the full conversation from state (LangGraph's
  add_messages reducer keeps it growing turn-by-turn), calls Haiku, and
  writes needs_rag into state. A single conditional edge routes execution.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from adapters import BedrockRAGAdapter, ComprehendAdapter, LocalRAGAdapter
from chat_node import invoke_chat
from checkpointer import DynamoDBSaver
from citations import extract_citations, refresh_presigned_urls
from classifier import classify_intent
from clients import (
    make_bedrock_agent_runtime,
    make_bedrock_runtime,
    make_comprehend,
    make_dynamodb_table,
    make_s3,
)
from prompts import CLASSIFIER_PROMPT, build_system_prompt
from rag import check_guardrail_block, retrieve_and_generate


_VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), ".local-vectorstore")

# Match handler.detect_language thresholds so behaviour is identical.
_DOMINANCE_THRESHOLD = 0.85


# ── State ────────────────────────────────────────────────────────────────────
# `messages` uses the add_messages reducer so new turns append to history.
# Everything else is a per-turn value overwritten each run.

class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], add_messages]
    language: str
    needs_rag: bool
    citations: list[dict]
    blocked: bool
    response_text: str
    kb_session_id: str | None


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
    """
    LangGraph's `add_messages` reducer converts dicts into LangChain
    BaseMessage objects (HumanMessage, AIMessage, ...). Nodes that read from
    state need a consistent {role, content} dict view so the downstream
    Bedrock calls can serialize to JSON.
    """
    if isinstance(m, dict):
        return {"role": m.get("role", ""), "content": m.get("content", "") or ""}
    msg_type = getattr(m, "type", None)
    role = {"human": "user", "ai": "assistant"}.get(msg_type, msg_type or "")
    content = getattr(m, "content", "") or ""
    return {"role": role, "content": content}


def _message_metadata(m) -> dict:
    """
    Returns the additional_kwargs dict attached to a LangChain message, or
    an empty dict for plain message dicts. Used by load_history() to
    reconstruct per-turn citations / language / blocked from state.
    """
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


def _make_rag_adapter(config):
    """
    Same selection logic as handler._make_rag_adapter, duplicated here so the
    graph module is self-contained. in_process=False because Lambda and
    server_local.py both supply CHROMA_HOST appropriately.
    """
    if config.is_local:
        return LocalRAGAdapter(
            make_bedrock_runtime(config),
            config.embedding_model_id,
            _VECTORSTORE_PATH,
            chroma_host=config.chroma_host,
            chroma_port=config.chroma_port,
        )
    return BedrockRAGAdapter(make_bedrock_agent_runtime(config), config)


# ── Nodes ────────────────────────────────────────────────────────────────────
# Each node receives AgentState and returns a partial state dict that
# LangGraph merges into the checkpointed state.


def _build_nodes(config):
    """
    Returns a dict of node_name -> callable. All AWS clients are instantiated
    once here and closed over by the node closures so repeated graph.invoke
    calls don't rebuild them.
    """
    comprehend = ComprehendAdapter(make_comprehend(config), config.is_local)
    bedrock_runtime = make_bedrock_runtime(config)
    rag_adapter = _make_rag_adapter(config)
    s3 = make_s3(config)

    def detect_language_node(state: AgentState) -> dict:
        user_msg = _latest_user_message(state.get("messages", []))
        return {"language": _detect_language(user_msg, comprehend)}

    def classify_intent_node(state: AgentState) -> dict:
        needs_rag = classify_intent(
            _as_role_content(state.get("messages", [])),
            bedrock_runtime,
            config.bedrock_model_id,
            CLASSIFIER_PROMPT,
        )
        return {"needs_rag": needs_rag}

    def _assistant_message(
        *, content: str, citations: list[dict], language: str, blocked: bool
    ) -> AIMessage:
        """
        Builds an AIMessage whose additional_kwargs carry everything the
        frontend needs to render a historical turn: citations (with
        pageImageKey for re-presigning), language, blocked flag, and a
        stable turn_id used as the React key. Presigned pageImageUrls
        are intentionally stripped before persistence because they
        expire after ~1 hour; load_history() re-presigns on read.
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

    def rag_node(state: AgentState) -> dict:
        user_msg = _latest_user_message(state.get("messages", []))
        language = state.get("language", "english")
        system_prompt = build_system_prompt(language)
        result = retrieve_and_generate(
            user_msg,
            system_prompt,
            config.bedrock_model_id,
            rag_adapter,
            kb_session_id=state.get("kb_session_id"),
        )
        blocked = check_guardrail_block(result)
        if blocked:
            from prompts import BILINGUAL_REFUSAL

            return {
                "response_text": BILINGUAL_REFUSAL,
                "citations": [],
                "blocked": True,
                "messages": [_assistant_message(
                    content=BILINGUAL_REFUSAL,
                    citations=[],
                    language=language,
                    blocked=True,
                )],
            }
        citations = extract_citations(result, s3_client=s3, bucket=config.s3_bucket)
        response_text = result.get("output", {}).get("text", "")
        return {
            "response_text": response_text,
            "citations": citations,
            "blocked": False,
            "kb_session_id": result.get("sessionId"),
            "messages": [_assistant_message(
                content=response_text,
                citations=citations,
                language=language,
                blocked=False,
            )],
        }

    def chat_node(state: AgentState) -> dict:
        language = state.get("language", "english")
        response_text = invoke_chat(
            _as_role_content(state.get("messages", [])),
            language,
            bedrock_runtime,
            config.bedrock_model_id,
        )
        return {
            "response_text": response_text,
            "citations": [],
            "blocked": False,
            "messages": [_assistant_message(
                content=response_text,
                citations=[],
                language=language,
                blocked=False,
            )],
        }

    return {
        "detect_language": detect_language_node,
        "classify_intent": classify_intent_node,
        "rag_node": rag_node,
        "chat_node": chat_node,
    }


# ── Compiled graph factory ───────────────────────────────────────────────────

_compiled_graph = None


def _route_after_classify(state: AgentState) -> str:
    return "rag_node" if state.get("needs_rag") else "chat_node"


def build_graph(config, checkpointer):
    """
    Constructs and compiles the graph. Separated from get_compiled_graph so
    unit tests can inject a fake checkpointer (e.g. InMemorySaver) and verify
    routing without touching DynamoDB.
    """
    nodes = _build_nodes(config)
    builder = StateGraph(AgentState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"rag_node": "rag_node", "chat_node": "chat_node"},
    )
    builder.add_edge("rag_node", END)
    builder.add_edge("chat_node", END)
    return builder.compile(checkpointer=checkpointer)


def get_compiled_graph(config):
    """
    Returns a process-cached compiled graph backed by DynamoDBSaver.
    Safe to call on every Lambda invocation; warm containers reuse the
    compiled graph and the boto3 clients inside it.
    """
    global _compiled_graph
    if _compiled_graph is None:
        table = make_dynamodb_table(config, config.checkpoints_table)
        checkpointer = DynamoDBSaver(table)
        _compiled_graph = build_graph(config, checkpointer)
    return _compiled_graph


# ── History hydration ────────────────────────────────────────────────────────

def load_history(config, session_id: str) -> list[dict]:
    """
    Returns the persisted conversation for a given sessionId as a list of
    ChatMessage-shaped dicts ready for the frontend:

      [{ id, role: "user", content }, ...,
       { id, role: "assistant", content, citations, language, blocked }]

    Citations are re-presigned here so pageImageUrls are always fresh. If
    the session has no checkpoint yet (brand-new sessionId), an empty list
    is returned — this is the normal path for a first-visit user.
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
