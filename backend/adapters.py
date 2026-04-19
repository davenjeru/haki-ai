"""
Adapters for AWS service clients.

Each adapter wraps a boto3 client and handles two things:
  1. Presents a simplified interface (only the methods handler.py needs).
  2. Papers over LocalStack limitations so business logic stays clean.

Business logic functions receive adapters, not raw boto3 clients.

LocalRAGAdapter is a local-only composite that mimics the Bedrock KB
retrieve_and_generate interface using ChromaDB + Bedrock InvokeModel.
It is never instantiated in the Lambda environment.
"""

import json

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


class LocalRAGAdapter:
    """
    Mimics the Bedrock KB retrieve_and_generate interface for local testing.

    Uses ChromaDB (persistent, cosine similarity) as the vector store and
    Bedrock InvokeModel for both Titan embeddings and Claude generation.
    Embeddings always hit real AWS — Bedrock is never available in LocalStack.

    Returns a dict that matches the Bedrock KB response shape so that
    citations.py and handler.py can treat local and prod identically.
    """

    def __init__(self, bedrock_runtime, embed_model: str, vectorstore_path: str):
        import chromadb  # local-only — not available in Lambda runtime

        self._bedrock = bedrock_runtime
        self._embed_model = embed_model

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
        return json.loads(response["body"].read())["embedding"]

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


class StubRAGAdapter:
    """
    Minimal RAG adapter for LocalStack Lambda invocations.

    LocalStack Lambda cannot access the host filesystem (no ChromaDB store)
    and chromadb is not installed in the Lambda zip. This stub lets the full
    handler pipeline run end-to-end inside Docker so the Lambda wiring,
    env vars, CloudWatch metrics, and response shape can all be verified
    without real retrieval.

    For actual RAG quality testing, use LocalRAGAdapter directly via
    test_e2e_local.py (runs in-process, not inside Docker).
    """

    def retrieve_and_generate(
        self, query: str, system_prompt: str, model_id: str
    ) -> dict:
        return {
            "output": {
                "text": (
                    "[LocalStack stub] RAG not available inside Docker. "
                    "Run ENV=local uv run test_e2e_local.py for real RAG testing."
                )
            },
            "citations": [],
            "stopReason": "end_turn",
        }


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
