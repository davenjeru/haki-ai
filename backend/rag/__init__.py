"""
Advanced RAG package.

Phase 1 split of the single `retrieve_and_generate` call into a five-stage
pipeline:

    query_expansion
        \u2192 hybrid_retrieve (dense + BM25 \u2192 RRF)
            \u2192 filters (drop TOC, dedup by section)
                \u2192 rerank (Cohere Rerank v3.5 on Bedrock)
                    \u2192 generate (Claude InvokeModel with context)

The pipeline is orchestrated by `rag.pipeline.run_rag`. Business logic in
`graph.py` stays almost identical \u2014 it receives the same response shape
(`output.text`, `citations`, `stopReason`) that `retrieve_and_generate`
used to return, so most of the graph code is untouched.

Legacy imports (`from rag import check_guardrail_block, retrieve_and_generate,
blocked_response`) continue to work; they\u2019re re-exported below.
"""

from __future__ import annotations

from prompts import BILINGUAL_REFUSAL

from .citations import extract_citations, refresh_presigned_urls
from .pipeline import run_rag


def check_guardrail_block(rag_result: dict) -> bool:
    """
    Returns True if Bedrock Guardrails blocked the response.

    Our advanced-RAG pipeline runs generate via `invoke_model`, which sets
    `stopReason == "guardrail_intervened"` when the guardrail fires. The
    legacy `retrieve_and_generate` path also set `guardrailAction=INTERVENED`;
    we honour both shapes so the same helper works across call sites.
    """
    if rag_result.get("guardrailAction") == "INTERVENED":
        return True
    return rag_result.get("stopReason") == "guardrail_intervened"


def blocked_response(language: str) -> str:
    """Bilingual refusal returned when a guardrail fires. Language agnostic."""
    return BILINGUAL_REFUSAL


def retrieve_and_generate(
    query: str,
    system_prompt: str,
    model_id: str,
    rag_adapter,
    kb_session_id: str | None = None,  # noqa: ARG001 \u2014 legacy signature kept for compatibility
) -> dict:
    """
    Back-compat shim over the advanced-RAG pipeline.

    Preserves the old signature (`query, system_prompt, model_id, rag_adapter`)
    so callers in `graph.py` don\u2019t change. The `kb_session_id` argument is
    accepted but ignored \u2014 LangGraph\u2019s DynamoDB checkpointer already persists
    the full conversation, so we no longer rely on Bedrock KB\u2019s server-side
    session context.
    """
    return run_rag(
        query=query,
        system_prompt=system_prompt,
        model_id=model_id,
        rag_adapter=rag_adapter,
    )


__all__ = [
    "run_rag",
    "retrieve_and_generate",
    "check_guardrail_block",
    "blocked_response",
    "extract_citations",
    "refresh_presigned_urls",
]
