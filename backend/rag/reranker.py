"""
Reranker \u2014 Cohere Rerank v3.5 via Amazon Bedrock.

Takes the fused + filtered retrieval candidates and re-scores them against
the original user query with a cross-encoder, which consistently improves
top-K precision by 10\u201330 percentage points on legal / statutory retrieval
benchmarks.

API surface: `bedrock-agent-runtime.rerank`. The rerank model runs in the
same AWS region as our KB so latency is low (80\u2013200 ms for 20 documents).

The reranker accepts up to 100 documents per call. If the caller passes
more, we truncate to the top-`_RERANK_MAX` preserving RRF order so we
still rerank the strongest candidates.

Failure mode: if the Bedrock Rerank call fails (throttling, model not
enabled, etc.), we log and return the original list unchanged. The
pipeline keeps working on degraded-but-correct retrievals rather than
hard-failing the request.
"""

from __future__ import annotations

_RERANK_MAX = 100
_DEFAULT_TOP_N = 5
_DEFAULT_MODEL = "cohere.rerank-v3-5:0"


def _model_arn(region: str, model_id: str) -> str:
    """
    Bedrock\u2019s rerank API takes a foundation-model ARN, not the short id.
    All rerank models live at `arn:aws:bedrock:<region>::foundation-model/<id>`.
    """
    return f"arn:aws:bedrock:{region}::foundation-model/{model_id}"


def rerank(
    query: str,
    results: list[dict],
    bedrock_agent_runtime,
    *,
    top_n: int = _DEFAULT_TOP_N,
    model_id: str = _DEFAULT_MODEL,
    region: str = "us-east-1",
) -> list[dict]:
    """
    Returns the top_n rerank-scored results in descending order.

    Each returned dict keeps the input metadata and adds:
      - score: the rerank relevance score (float, typically 0\u20131)
      - rerankScore: same value, kept separately so downstream code can
        reason about the component scores individually
    """
    query = (query or "").strip()
    if not query or not results:
        return results[:top_n]

    candidates = results[:_RERANK_MAX]
    sources = [
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {"text": (r.get("content") or {}).get("text", "")[:8000]},
            },
        }
        for r in candidates
    ]

    try:
        response = bedrock_agent_runtime.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": min(top_n, len(candidates)),
                    "modelConfiguration": {
                        "modelArn": _model_arn(region, model_id),
                    },
                },
            },
        )
    except Exception as err:
        print(f"[rerank] Bedrock Rerank failed, returning un-reranked list: {err}")
        return candidates[:top_n]

    ranked = response.get("results", []) or []
    out: list[dict] = []
    for hit in ranked:
        idx = hit.get("index")
        if idx is None or idx >= len(candidates):
            continue
        item = dict(candidates[idx])
        score = float(hit.get("relevanceScore", 0.0))
        item["score"] = score
        item["rerankScore"] = score
        out.append(item)
    return out[:top_n]
