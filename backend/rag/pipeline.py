"""
RAG pipeline orchestrator.

Runs the five-stage advanced RAG flow:

    expand_query        \u2192 [original, hypothetical, decomposed]
        retrieve        \u2192 dense + BM25 per variant
            fuse        \u2192 RRF across all 2\u00d7N result lists
                filter  \u2192 drop TOC, dedup by (source, section)
                    rerank  \u2192 Cohere Rerank v3.5 on Bedrock
                        generate \u2192 Claude InvokeModel with context

Returns a Bedrock-KB-compatible response dict so `citations.extract_citations`
and the handler\u2019s downstream code can treat advanced-RAG output identically
to the legacy retrieve_and_generate response:

    {
      "output":     {"text": "\u2026"},
      "citations":  [ { "retrievedReferences": [ {content, metadata, location} ] } ],
      "stopReason": "end_turn" | "guardrail_intervened" | \u2026,
    }

The adapter abstraction (`rag_adapter`) owns the dense-retrieval side of
things \u2014 it may talk to Bedrock KB (prod) or ChromaDB (local). BM25,
query expansion, rerank, and generation are environment-agnostic because
they all talk to real Bedrock, which is available in both environments.
"""

from __future__ import annotations

from typing import Protocol

from . import bm25 as bm25_module
from . import filters as filter_module
from . import rrf
from .catalog import get_catalog
from .generator import build_context, generate
from .query_expansion import expand_query
from .reranker import rerank


_DEFAULT_DENSE_TOP_K = 30
_DEFAULT_BM25_TOP_K = 30
_DEFAULT_FUSE_TOP_N = 20
_DEFAULT_RERANK_TOP_N = 5


class RAGAdapterLike(Protocol):
    """
    Duck-typed RAG adapter. Both `LocalRAGAdapter` and `BedrockRAGAdapter`
    implement this interface after the Phase 1 refactor.
    """

    def retrieve(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        ...

    def generate(
        self,
        query: str,
        system_prompt: str,
        context: str,
        model_id: str,
    ) -> tuple[str, str]:
        ...

    @property
    def catalog_bucket(self) -> str: ...
    @property
    def catalog_s3_client(self): ...
    @property
    def catalog_list_client(self): ...
    @property
    def bedrock_agent_runtime(self): ...
    @property
    def bedrock_runtime(self): ...
    @property
    def aws_region(self) -> str: ...


def run_rag(
    query: str,
    system_prompt: str,
    model_id: str,
    rag_adapter: RAGAdapterLike,
    *,
    metadata_filter: dict | None = None,
    dense_top_k: int = _DEFAULT_DENSE_TOP_K,
    bm25_top_k: int = _DEFAULT_BM25_TOP_K,
    fuse_top_n: int = _DEFAULT_FUSE_TOP_N,
    rerank_top_n: int = _DEFAULT_RERANK_TOP_N,
) -> dict:
    """
    Executes the full pipeline and returns a retrieve_and_generate-shaped
    response dict.

    metadata_filter is applied to BOTH dense and BM25 retrieval so each
    specialist agent (Phase 2) can constrain its corpus to e.g.
    `{"source": "Employment Act 2007"}`.
    """
    query = (query or "").strip()

    # 1. Query expansion (original + up to 2 rewrites).
    variants = expand_query(query, rag_adapter.bedrock_runtime, model_id)
    if not variants:
        variants = [query]

    # 2. Dense + BM25 retrieval per variant.
    catalog = get_catalog(
        rag_adapter.catalog_list_client,
        rag_adapter.catalog_bucket,
    )

    dense_lists: list[list[dict]] = []
    sparse_lists: list[list[dict]] = []
    for variant in variants:
        try:
            dense = rag_adapter.retrieve(
                variant, top_k=dense_top_k, metadata_filter=metadata_filter
            )
        except Exception as err:
            print(f"[rag.pipeline] dense retrieve failed ({variant!r}): {err}")
            dense = []
        dense_lists.append([{**r, "retriever": "dense"} for r in dense])

        try:
            sparse = bm25_module.retrieve(
                variant,
                catalog,
                top_k=bm25_top_k,
                metadata_filter=metadata_filter,
            )
        except Exception as err:
            print(f"[rag.pipeline] BM25 retrieve failed ({variant!r}): {err}")
            sparse = []
        sparse_lists.append([{**r, "retriever": "bm25"} for r in sparse])

    # 3. RRF fusion across all 2\u00d7len(variants) result lists.
    fused = rrf.fuse(dense_lists + sparse_lists)[:fuse_top_n]

    # 4. Filter: drop TOC + boilerplate (preamble, short-title,
    # definitions), then dedup by (source, section). Boilerplate is
    # dropped unconditionally for v1 because none of the current golden
    # questions are definitional; if we later add a "/define X" flow the
    # filter should be gated by a query-intent classifier.
    filtered = filter_module.drop_toc(fused)
    filtered = filter_module.drop_boilerplate(filtered)
    deduped = filter_module.dedup_by_section(filtered)

    # 5. Rerank with Cohere Rerank v3.5 (returns top_n).
    reranked = rerank(
        query,
        deduped,
        rag_adapter.bedrock_agent_runtime,
        top_n=rerank_top_n,
        region=rag_adapter.aws_region,
    )

    # 6. Generate answer grounded in the reranked context.
    context = build_context(reranked)
    answer_text, stop_reason = rag_adapter.generate(
        query, system_prompt, context, model_id
    )

    return {
        "output": {"text": answer_text},
        "citations": [
            {
                "retrievedReferences": [
                    {
                        "content":  r.get("content", {}),
                        "metadata": r.get("metadata", {}),
                        "location": r.get("location", {}),
                    }
                    for r in reranked
                ]
            }
        ],
        "stopReason": stop_reason,
    }
