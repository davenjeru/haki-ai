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
    "pageImageKey": "page-images/employment-act-2007/page-40.pdf",  # may be absent
  }
"""


def extract_citations(rag_result: dict) -> list[dict]:
    """
    Extracts and deduplicates citations from a retrieve_and_generate response.

    Args:
        rag_result: The dict returned by LocalRAGAdapter or BedrockRAGAdapter.

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

            # Use S3 URI as fallback key when chunkId is missing
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

            # pageImageKey is optional — only present when the pipeline
            # successfully extracted a single-page PDF for this chunk
            if meta.get("pageImageKey"):
                citation["pageImageKey"] = meta["pageImageKey"]

            citations.append(citation)

    return citations
