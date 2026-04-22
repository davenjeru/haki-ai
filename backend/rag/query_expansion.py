"""
Query expansion \u2014 first stage of the advanced RAG pipeline.

Given a user query, returns a list of 3 rewritten variants used to broaden
dense + sparse retrieval recall. Each variant targets a different failure
mode of naive embedding retrieval:

  1. original       \u2014 the user\u2019s exact words (keeps BM25 precision high
                      on statutory jargon like \u201cSection 40\u201d).
  2. hypothetical   \u2014 a HyDE-style one-sentence hypothetical answer. Moves
                      the query into the same vector neighbourhood as the
                      statutory text we indexed, which shortens the
                      embedding-space distance for real matches.
  3. decomposed     \u2014 the single most-important sub-question, useful when
                      a user asks a compound query like \u201cWhat rights do I
                      have and what\u2019s the notice period?\u201d.

A single Haiku call returns all three variants as JSON. If the call or the
parse fails we return `[original]` so retrieval still proceeds \u2014 query
expansion is a precision/recall multiplier, not a correctness gate.
"""

from __future__ import annotations

import json
import re


_EXPANSION_PROMPT = """You expand a user's legal question into alternative \
phrasings so a retriever can find the right statutes.

Return ONLY a JSON object with exactly two string fields and nothing else:
  {
    "hypothetical": "<a plausible one-sentence answer that cites a law>",
    "decomposed":   "<the single most important sub-question, or the same question rephrased>"
  }

Rules:
  - Keep every field short (\u2264 25 words).
  - Use Kenyan legal vocabulary (Act, Chapter, Part, Section, Article).
  - Do not wrap the JSON in markdown fences.
  - If the user's input is already narrow, make 'decomposed' a rephrasing \
using different keywords."""


_MAX_VARIANTS = 3


def _parse_variants(raw: str) -> tuple[str | None, str | None]:
    """
    Pulls the first JSON object out of the model\u2019s reply and reads the
    `hypothetical` and `decomposed` fields. Returns (None, None) on any
    parse failure so the caller falls back to the original query only.
    """
    # Haiku sometimes wraps in markdown fences despite instructions.
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None

    hypothetical = obj.get("hypothetical")
    decomposed = obj.get("decomposed")
    return (
        hypothetical if isinstance(hypothetical, str) and hypothetical.strip() else None,
        decomposed if isinstance(decomposed, str) and decomposed.strip() else None,
    )


def expand_query(
    query: str,
    bedrock_runtime,
    model_id: str,
) -> list[str]:
    """
    Returns a list of up to 3 query strings: the original plus up to two
    rewrites. Variants are deduplicated case-insensitively so an identical
    rewrite doesn\u2019t waste a retrieval round-trip.

    The Bedrock call uses a small max_tokens budget and a single system
    message \u2014 the overhead is ~200ms per turn.
    """
    query = (query or "").strip()
    if not query:
        return []

    variants: list[str] = [query]

    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "system": _EXPANSION_PROMPT,
            "messages": [{"role": "user", "content": query}],
        }
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        text = (result.get("content", [{}]) or [{}])[0].get("text", "")
        hypothetical, decomposed = _parse_variants(text)
        for v in (hypothetical, decomposed):
            if v:
                variants.append(v.strip())
    except Exception as err:
        # Fall back to original only \u2014 expansion is best-effort.
        print(f"[query_expansion] falling back to original query: {err}")

    return _dedup_preserving_order(variants)[:_MAX_VARIANTS]


def _dedup_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        key = s.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
