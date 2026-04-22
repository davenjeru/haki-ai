"""
Citation extraction — step 5.

extract_citations() parses the RAG response from either LocalRAGAdapter
or BedrockRAGAdapter and returns a list of citation dicts for the frontend.

Each citation identifies a unique law section that was retrieved and used
to generate the answer. Duplicates (same chunkId) are collapsed.

Output format (one entry per unique chunk):
  {
    "source":       "Employment Act 2007",
    "chapter":      "Part III — Termination of Contract",
    "section":      "Section 40",
    "title":        "Termination of employment",
    "chunkId":      "employment-act-2007-part-iii-section-40",
    "pageImageUrl": "https://.../page-40.pdf?X-Amz-Signature=...",  # may be absent
  }

pageImageUrl is a presigned S3 GET URL with a 1-hour TTL. The frontend
fetches the single-page PDF directly from S3 (LocalStack locally, real S3
in prod) and renders it onto a canvas with PDF.js. Same code path everywhere.
"""

_PAGE_IMAGES_PREFIX = "page-images/"
_PRESIGN_TTL_SECONDS = 3600


def _presign(s3_client, bucket: str, key: str) -> str | None:
    """
    Returns a presigned GET URL for an S3 object under page-images/.

    Returns None when:
      - no S3 client was provided
      - the key is outside the page-images/ prefix (defensive)
      - boto3 raises (network blip, bad creds) — we don't want a single
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


def extract_citations(
    rag_result: dict,
    *,
    s3_client=None,
    bucket: str = "",
) -> list[dict]:
    """
    Extracts and deduplicates citations from a retrieve_and_generate response.

    Args:
        rag_result: The dict returned by LocalRAGAdapter or BedrockRAGAdapter.
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
            meta: dict = ref.get("metadata", {})
            chunk_id: str = meta.get("chunkId", "")

            if not chunk_id:
                location = ref.get("location", {})
                chunk_id = location.get("s3Location", {}).get("uri", "")

            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            citation = {
                "source":   meta.get("source", ""),
                "chapter":  meta.get("chapter", ""),
                "section":  meta.get("section", ""),
                "title":    meta.get("title", ""),
                "chunkId":  chunk_id,
            }

            page_image_key = meta.get("pageImageKey")
            if page_image_key:
                url = _presign(s3_client, bucket, page_image_key)
                if url:
                    citation["pageImageUrl"] = url

            citations.append(citation)

    return citations
