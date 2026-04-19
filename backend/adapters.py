"""
Adapters for AWS service clients.

Each adapter wraps a boto3 client and handles two things:
  1. Presents a simplified interface (only the methods handler.py needs).
  2. Papers over LocalStack limitations so business logic stays clean.

Business logic functions receive adapters, not raw boto3 clients.

LocalRAGAdapter is a local-only composite that mimics the Bedrock KB
retrieve_and_generate interface using ChromaDB + Bedrock InvokeModel.
It is never instantiated in the Lambda environment.

When chroma_host is set, LocalRAGAdapter uses _ChromaHttpClient to query
the ChromaDB HTTP server over the network (same pattern as Bedrock KB →
S3 Vectors in prod). This lets the Lambda container reach ChromaDB running
on the host machine via host.docker.internal without needing the chromadb
package (and its native hnswlib binary) inside the Lambda zip.

When chroma_host is empty, LocalRAGAdapter falls back to ChromaDB
PersistentClient for in-process access (test_e2e_local.py path).
"""

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

_S3_BUCKET = "haki-ai-data"
_CHUNKS_PREFIX = "processed-chunks/"
_CHROMA_COLLECTION = "haki_chunks"
_TOP_K = 5

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
    ) -> dict:
        col_id = self._get_collection_id()
        payload: dict = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        return self._request("POST", f"/collections/{col_id}/query", payload)


class LocalRAGAdapter:
    """
    Mimics the Bedrock KB retrieve_and_generate interface for local testing.

    Uses ChromaDB as the vector store and Bedrock InvokeModel for both Titan
    embeddings and Claude generation. Embeddings always hit real AWS.

    Two storage backends (selected at construction time):
      - HTTP  (chroma_host set): queries ChromaDB HTTP server via _ChromaHttpClient.
               Used by Lambda-in-Docker — no chromadb package needed in the zip.
               host.docker.internal:8000 reaches the host from the container.
      - In-process (chroma_host empty): uses chromadb.PersistentClient directly.
               Used by test_e2e_local.py running outside Docker.

    Returns a dict that matches the Bedrock KB response shape so that
    citations.py and handler.py can treat local and prod identically.
    """

    def __init__(
        self,
        bedrock_runtime,
        embed_model: str,
        vectorstore_path: str,
        chroma_host: str = "",
        chroma_port: int = 8000,
    ):
        self._bedrock = bedrock_runtime
        self._embed_model = embed_model

        if chroma_host:
            self._collection = _ChromaHttpClient(chroma_host, chroma_port)
        else:
            import chromadb  # local-only — not available in Lambda runtime
            chroma = chromadb.PersistentClient(path=vectorstore_path)
            self._collection = chroma.get_or_create_collection(
                name=_CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def retrieve_and_generate(
        self, query: str, system_prompt: str, model_id: str
    ) -> dict:
        """
        Retrieve relevant chunks from ChromaDB, then generate a response via
        Claude InvokeModel. Returns the same shape as Bedrock KB:

          {
            "output": { "text": str },
            "citations": [ { "retrievedReferences": [ { "content", "metadata", "location" } ] } ],
            "stopReason": str,
          }
        """
        query_vector = self._embed(query)

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=_TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        documents: list[str] = results["documents"][0]
        metadatas: list[dict] = results["metadatas"][0]

        context = self._build_context(documents, metadatas)
        user_content = f"<context>\n{context}\n</context>\n\n{query}"

        answer_text, stop_reason = self._invoke_claude(model_id, system_prompt, user_content)

        citations = [
            {
                "retrievedReferences": [
                    {
                        "content": {"text": doc},
                        "metadata": meta,
                        "location": {
                            "type": "S3",
                            "s3Location": {
                                "uri": (
                                    f"s3://{_S3_BUCKET}/{_CHUNKS_PREFIX}"
                                    f"{meta.get('chunkId', '')}.txt"
                                )
                            },
                        },
                    }
                ]
            }
            for doc, meta in zip(documents, metadatas)
        ]

        return {
            "output": {"text": answer_text},
            "citations": citations,
            "stopReason": stop_reason,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        response = self._bedrock.invoke_model(
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

    def _invoke_claude(
        self, model_id: str, system_prompt: str, user_content: str
    ) -> tuple[str, str]:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }
        response = self._bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        text = result.get("content", [{}])[0].get("text", "")
        stop_reason = result.get("stop_reason", "end_turn")
        return text, stop_reason

    @staticmethod
    def _build_context(documents: list[str], metadatas: list[dict]) -> str:
        parts = []
        for doc, meta in zip(documents, metadatas):
            header = " — ".join(
                filter(None, [
                    meta.get("source"),
                    meta.get("chapter"),
                    meta.get("section"),
                    meta.get("title"),
                ])
            )
            parts.append(f"[{header}]\n{doc}")
        return "\n\n---\n\n".join(parts)



class BedrockRAGAdapter:
    """
    Wraps the bedrock-agent-runtime boto3 client so it shares the same
    retrieve_and_generate interface as LocalRAGAdapter.

    rag.py calls retrieve_and_generate() on whichever adapter it receives —
    it never needs to know which environment it's in.
    """

    def __init__(self, client, config):
        self._client = client
        self._config = config

    def retrieve_and_generate(
        self, query: str, system_prompt: str, model_id: str
    ) -> dict:
        """
        Calls Bedrock KB retrieve_and_generate and returns the raw response.
        The response already matches the expected shape:
          { "output": { "text" }, "citations": [...], "stopReason": str }
        """
        response = self._client.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": self._config.knowledge_base_id,
                    "modelArn": f"arn:aws:bedrock:{self._config.aws_region}::foundation-model/{model_id}",
                    "generationConfiguration": {
                        "promptTemplate": {"textPromptTemplate": system_prompt},
                        "guardrailConfiguration": {
                            "guardrailId": self._config.guardrail_id,
                            "guardrailVersion": self._config.guardrail_version,
                        },
                    },
                },
            },
        )
        return response
