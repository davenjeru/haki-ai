"""
Adapters for AWS service clients.

Each adapter wraps a boto3 client and handles two things:
  1. Presents a simplified interface (only the methods handler.py needs).
  2. Papers over LocalStack limitations so business logic stays clean.

Phase 1 advanced-RAG split:
  The RAG adapter now exposes TWO methods instead of one monolithic
  `retrieve_and_generate`:

    - `retrieve(query, top_k, metadata_filter) -> list[dict]`
      Returns raw ranked chunks. Downstream rerank/fuse/filter code treats
      both adapters symmetrically.

    - `generate(query, system_prompt, context, model_id) -> (text, stop_reason)`
      Calls Claude InvokeModel with the already-assembled context from the
      rerank stage. Guardrails attach here (prod only).

  The pipeline orchestrator in `rag/pipeline.py` owns the stages between
  `retrieve` and `generate` (query expansion, RRF, rerank, filters).

LocalRAGAdapter is a local-only composite used by `server_local.py` and
`test_e2e_local.py`. BedrockRAGAdapter targets real AWS Bedrock KB + Cohere
Rerank for production.

When chroma_host is set, LocalRAGAdapter queries the ChromaDB HTTP server
over the network. When empty, it uses ChromaDB\u2019s PersistentClient in-process.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from botocore.exceptions import ClientError


class ComprehendAdapter:
    """
    Wraps a Comprehend boto3 client.

    When LocalStack does not support DetectDominantLanguage, returns a
    default English response so the Lambda can complete successfully.
    The real operation is always attempted first.
    """

    def __init__(self, client, is_local: bool):
        self._client = client
        self._is_local = is_local

    def detect_dominant_language(self, text: str) -> list[dict]:
        """
        Returns a list of detected languages sorted by confidence score:
          [{"LanguageCode": "en", "Score": 0.98}, ...]
        """
        try:
            response = self._client.detect_dominant_language(Text=text)
            return response.get("Languages", [])
        except ClientError as err:
            if self._is_local and "not currently supported" in str(err):
                print("[local] Comprehend DetectDominantLanguage unavailable — defaulting to English")
                return [{"LanguageCode": "en", "Score": 1.0}]
            raise


# ── RAG adapters ─────────────────────────────────────────────────────────────

_CHUNKS_PREFIX = "processed-chunks/"
_CHROMA_COLLECTION = "haki_chunks"

_CHROMA_TENANT = "default_tenant"
_CHROMA_DB = "default_database"


class _ChromaHttpClient:
    """
    Minimal ChromaDB v2 HTTP client using only stdlib urllib.

    No chromadb package required — works inside the Lambda zip without
    native dependencies. Mirrors the query interface used by ChromaDB's
    own PersistentClient so LocalRAGAdapter can swap between the two.
    """

    def __init__(self, host: str, port: int):
        self._base = f"http://{host}:{port}/api/v2/tenants/{_CHROMA_TENANT}/databases/{_CHROMA_DB}"
        self._collection_id: str | None = None

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self._base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"ChromaDB {method} {path} → HTTP {err.code}: {err.read().decode()}") from err

    def _get_collection_id(self) -> str:
        if self._collection_id is None:
            collections = self._request("GET", "/collections")
            for col in collections:
                if col.get("name") == _CHROMA_COLLECTION:
                    self._collection_id = col["id"]
                    break
            if self._collection_id is None:
                raise RuntimeError(f"ChromaDB collection '{_CHROMA_COLLECTION}' not found. Run ingest_local.py first.")
        return self._collection_id

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        include: list[str] | None = None,
        where: dict | None = None,
    ) -> dict:
        col_id = self._get_collection_id()
        payload: dict = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        if where:
            payload["where"] = where
        return self._request("POST", f"/collections/{col_id}/query", payload)


class LocalRAGAdapter:
    """
    Local-dev RAG adapter that mirrors the BedrockRAGAdapter interface.

    Retrieval runs over a ChromaDB collection ingested from LocalStack S3;
    embeddings + generation always hit real AWS Bedrock. Supports both
    in-process (ChromaDB PersistentClient) and HTTP (ChromaDB server)
    backends so the local Lambda-in-Docker and server_local.py paths both
    work.
    """

    def __init__(
        self,
        bedrock_runtime,
        bedrock_agent_runtime,
        embed_model: str,
        vectorstore_path: str,
        *,
        s3_client,
        s3_bucket: str,
        aws_region: str,
        chroma_host: str = "",
        chroma_port: int = 8000,
        guardrail_id: str = "",
        guardrail_version: str = "",
    ):
        self._bedrock_runtime = bedrock_runtime
        self._bedrock_agent_runtime = bedrock_agent_runtime
        self._embed_model = embed_model
        self._s3_client = s3_client
        self._s3_bucket = s3_bucket
        self._aws_region = aws_region
        self._guardrail_id = guardrail_id
        self._guardrail_version = guardrail_version

        if chroma_host:
            self._collection = _ChromaHttpClient(chroma_host, chroma_port)
        else:
            import chromadb  # local-only — not available in Lambda runtime
            chroma = chromadb.PersistentClient(path=vectorstore_path)
            self._collection = chroma.get_or_create_collection(
                name=_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    # ── Catalog accessors used by the RAG pipeline ────────────────────────────

    @property
    def catalog_s3_client(self):
        return self._s3_client

    @property
    def catalog_bucket(self) -> str:
        return self._s3_bucket

    @property
    def bedrock_runtime(self):
        return self._bedrock_runtime

    @property
    def bedrock_agent_runtime(self):
        return self._bedrock_agent_runtime

    @property
    def aws_region(self) -> str:
        return self._aws_region

    # ── Public RAG interface ─────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Embeds `query` with Titan v2 and queries ChromaDB. Returns results
        in the Bedrock-KB-compatible shape so the pipeline treats both
        adapters identically.
        """
        query = (query or "").strip()
        if not query:
            return []
        vector = self._embed(query)
        where = self._chroma_where(metadata_filter)

        if where is not None:
            results = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
        else:
            results = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        documents: list[str] = (results.get("documents") or [[]])[0]
        metadatas: list[dict] = (results.get("metadatas") or [[]])[0]
        distances: list[float] = (results.get("distances") or [[]])[0] or []

        out: list[dict] = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            chunk_id = (meta or {}).get("chunkId") or ""
            score = 1.0 - float(distances[i]) if i < len(distances) else 0.0
            out.append({
                "content": {"text": doc},
                "metadata": meta or {},
                "location": {
                    "type": "S3",
                    "s3Location": {
                        "uri": f"s3://{self._s3_bucket}/{_CHUNKS_PREFIX}{chunk_id}.txt"
                    },
                },
                "score": score,
            })
        return out

    def generate(
        self,
        query: str,
        system_prompt: str,
        context: str,
        model_id: str,
    ) -> tuple[str, str]:
        """Calls Claude InvokeModel."""
        from rag.generator import generate as _generate
        return _generate(
            query=query,
            system_prompt=system_prompt,
            context=context,
            model_id=model_id,
            bedrock_runtime=self._bedrock_runtime,
            guardrail_id=self._guardrail_id,
            guardrail_version=self._guardrail_version,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        response = self._bedrock_runtime.invoke_model(
            modelId=self._embed_model,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
            accept="application/json",
        )
        raw = response["body"].read()
        try:
            return json.loads(raw)["embedding"]
        except (json.JSONDecodeError, KeyError) as err:
            raise RuntimeError(
                f"Bedrock embed call returned unexpected response "
                f"(model={self._embed_model}): {raw[:200]!r}"
            ) from err

    @staticmethod
    def _chroma_where(metadata_filter: dict | None) -> dict | None:
        """
        Translates our simple AND-of-equals filter into ChromaDB\u2019s query
        syntax. Returns None when no filter is provided so we don\u2019t send
        an empty `where` clause (ChromaDB rejects it).
        """
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            key, value = next(iter(metadata_filter.items()))
            return {key: {"$eq": value}}
        return {"$and": [{k: {"$eq": v}} for k, v in metadata_filter.items()]}


class BedrockRAGAdapter:
    """
    Production RAG adapter backed by Bedrock KB (retrieve) + Bedrock
    InvokeModel (generate). Cohere Rerank is handled by the pipeline, not
    the adapter.

    The old `retrieve_and_generate` path is retired; the advanced-RAG
    pipeline runs retrieval and generation as separate steps so query
    expansion, BM25 fusion, rerank, and TOC filtering can slot between
    them. See `rag/pipeline.py` for the orchestrator.
    """

    def __init__(
        self,
        bedrock_agent_runtime,
        bedrock_runtime,
        config,
        *,
        s3_client,
    ):
        self._bedrock_agent_runtime = bedrock_agent_runtime
        self._bedrock_runtime = bedrock_runtime
        self._config = config
        self._s3_client = s3_client

    @property
    def catalog_s3_client(self):
        return self._s3_client

    @property
    def catalog_bucket(self) -> str:
        return self._config.s3_bucket

    @property
    def bedrock_runtime(self):
        return self._bedrock_runtime

    @property
    def bedrock_agent_runtime(self):
        return self._bedrock_agent_runtime

    @property
    def aws_region(self) -> str:
        return self._config.aws_region

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """
        Calls `bedrock-agent-runtime.retrieve` against the configured KB
        and normalises the response to the standard adapter shape.
        """
        query = (query or "").strip()
        if not query:
            return []

        kwargs: dict = {
            "knowledgeBaseId": self._config.knowledge_base_id,
            "retrievalQuery": {"text": query},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                }
            },
        }
        kb_filter = self._kb_filter(metadata_filter)
        if kb_filter is not None:
            kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = kb_filter

        response = self._bedrock_agent_runtime.retrieve(**kwargs)

        out: list[dict] = []
        for item in response.get("retrievalResults", []) or []:
            content = item.get("content") or {}
            metadata = item.get("metadata") or {}
            location = item.get("location") or {}
            score = float(item.get("score", 0.0))
            out.append({
                "content": {"text": content.get("text", "")},
                "metadata": metadata,
                "location": location,
                "score": score,
            })
        return out

    def generate(
        self,
        query: str,
        system_prompt: str,
        context: str,
        model_id: str,
    ) -> tuple[str, str]:
        """Claude InvokeModel call with guardrail attached."""
        from rag.generator import generate as _generate
        return _generate(
            query=query,
            system_prompt=system_prompt,
            context=context,
            model_id=model_id,
            bedrock_runtime=self._bedrock_runtime,
            guardrail_id=self._config.guardrail_id,
            guardrail_version=self._config.guardrail_version,
        )

    @staticmethod
    def _kb_filter(metadata_filter: dict | None) -> dict | None:
        """
        Translates our simple AND-of-equals filter dict to Bedrock KB's
        `retrievalFilter` JSON:
          single key  \u2192 {"equals": {"key": "...", "value": "..."}}
          multi keys  \u2192 {"andAll": [ { "equals": ... }, { "equals": ... } ]}
        """
        if not metadata_filter:
            return None
        terms = [
            {"equals": {"key": k, "value": v}}
            for k, v in metadata_filter.items()
        ]
        if len(terms) == 1:
            return terms[0]
        return {"andAll": terms}
