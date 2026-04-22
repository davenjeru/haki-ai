"""
Citation extraction \u2014 final stage of the RAG pipeline.

extract_citations() takes the pipeline\u2019s intermediate retrieval result (after
rerank + filters) and produces the citation list the frontend renders.

Dedup key is `(source, section)` rather than the raw `chunkId` so that two
chunks emitted by `splitToSegments` (same section, multi-paragraph split)
collapse into one citation \u2014 which is what users expect when the UI says
\u201cSource: Employment Act 2007 \u2022 Section 40\u201d. Falls back to `chunkId` when
either field is missing (defensive; shouldn\u2019t happen in prod).

Output format (one entry per unique section):
  {
    "source":       "Employment Act 2007",
    "chapter":      "Part III \u2014 Termination of Contract",
    "section":      "Section 40",
    "title":        "Termination of employment",
    "chunkId":      "employment-act-2007-part-iii-section-40",
    "pageImageUrl": "https://.../page-40.pdf?X-Amz-Signature=...",  # may be absent
  }
"""

from __future__ import annotations

_PAGE_IMAGES_PREFIX = "page-images/"
_PRESIGN_TTL_SECONDS = 3600


def _presign(s3_client, bucket: str, key: str) -> str | None:
    """
    Returns a presigned GET URL for an S3 object under page-images/.

    Returns None when:
      - no S3 client was provided
      - the key is outside the page-images/ prefix (defensive)
      - boto3 raises (network blip, bad creds) \u2014 we don\u2019t want a single
        bad page to break the whole response
    """
    if s3_client is None or not bucket or not key:
        return None
    if not key.startswith(_PAGE_IMAGES_PREFIX):
        return None
    try:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=_PRESIGN_TTL_SECONDS,
        )
    except Exception as err:
        print(f"Failed to presign {key}: {err}")
        return None


def refresh_presigned_urls(
    citations: list[dict],
    *,
    s3_client,
    bucket: str,
) -> list[dict]:
    """
    Returns a new list of citations with `pageImageUrl` re-presigned from
    the persisted `pageImageKey`. Citations without a `pageImageKey` or
    with a key outside the allowed prefix pass through unchanged (minus
    any stale pageImageUrl).

    Used by the history hydration path where stored citations may carry a
    presigned URL that has since expired.
    """
    refreshed: list[dict] = []
    for c in citations:
        out = dict(c)
        out.pop("pageImageUrl", None)
        key = out.get("pageImageKey")
        if key:
            url = _presign(s3_client, bucket, key)
            if url:
                out["pageImageUrl"] = url
        refreshed.append(out)
    return refreshed


def _dedup_key(meta: dict) -> str:
    """
    Canonical key for deduplicating citations. Uses (source, section) so
    splitToSegments-produced sibling chunks (same section, different paragraph
    segment) collapse into one UI citation. Falls back to chunkId if either
    field is absent (preserves legacy behaviour on malformed metadata).
    """
    source = (meta.get("source") or "").strip()
    section = (meta.get("section") or "").strip()
    if source and section:
        return f"{source}|{section}"
    return meta.get("chunkId") or ""


def extract_citations(
    rag_result: dict,
    *,
    s3_client=None,
    bucket: str = "",
) -> list[dict]:
    """
    Extracts and deduplicates citations from a RAG pipeline response.

    Accepts both the new pipeline shape and the legacy Bedrock KB shape so
    we don\u2019t break existing tests. Both shapes store retrieved chunks under
    `citations[*].retrievedReferences[*]`.

    Args:
        rag_result: The dict returned by `run_rag` (or legacy adapter).
        s3_client:  boto3 S3 client used to presign page-image URLs. When
                    None (e.g. in unit tests), pageImageUrl is omitted.
        bucket:     S3 bucket holding page-images/. Required together with
                    s3_client to produce URLs.

    Returns:
        Ordered list of unique citation dicts. Order matches the order the
        references first appeared in the response (most relevant first).
    """
    seen: set[str] = set()
    citations: list[dict] = []

    for citation_group in rag_result.get("citations", []):
        for ref in citation_group.get("retrievedReferences", []):
            meta: dict = ref.get("metadata", {}) or {}
            chunk_id: str = meta.get("chunkId") or ""

            if not chunk_id:
                location = ref.get("location", {}) or {}
                chunk_id = (location.get("s3Location", {}) or {}).get("uri", "")

            dedup_key = _dedup_key({**meta, "chunkId": chunk_id})
            if not dedup_key or dedup_key in seen:
                continue
            seen.add(dedup_key)

            citation = {
                "source":  meta.get("source", ""),
                "chapter": meta.get("chapter", ""),
                "section": meta.get("section", ""),
                "title":   meta.get("title", ""),
                "chunkId": chunk_id,
            }

            page_image_key = meta.get("pageImageKey")
            if page_image_key:
                # Persisted alongside the citation so history hydration can
                # re-presign a fresh URL each time (presigned URLs expire
                # after _PRESIGN_TTL_SECONDS \u2248 1 hour).
                citation["pageImageKey"] = page_image_key
                url = _presign(s3_client, bucket, page_image_key)
                if url:
                    citation["pageImageUrl"] = url

            citations.append(citation)

    return citations
