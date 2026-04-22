"""
End-to-end local RAG test runner.

Invokes lambda_handler directly in-process (not via LocalStack Docker). Runs
with CHROMA_HOST="" so LocalRAGAdapter uses chromadb.PersistentClient to
access .local-vectorstore/ directly — no ChromaDB HTTP server required.

This tests the full 6-step pipeline:
  1. Language detection (Comprehend → LocalStack fallback)
  2. System prompt construction
  3. RAG via ChromaDB + Titan embed + Claude InvokeModel (real AWS)
  4. Guardrail block check (always False locally — no guardrail layer)
  5. Citation extraction
  6. CloudWatch metrics (LocalStack)

Prerequisites:
  - LocalStack running (for Comprehend + CloudWatch)
  - .local-vectorstore/ populated (ENV=local uv run ingest_local.py)
  - AWS credentials configured for real Bedrock

Usage:
  ENV=local uv run test_e2e_local.py
"""

import json
import os
import sys

# Tell handler to use LocalRAGAdapter with in-process PersistentClient.
# CHROMA_HOST="" (default) → chromadb.PersistentClient on .local-vectorstore/
os.environ["ENV"] = "local"
os.environ.setdefault("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
os.environ.setdefault("CHROMA_HOST", "")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.handler import lambda_handler

# ── Test cases ────────────────────────────────────────────────────────────────

CASES = [
    {
        "message": "What are my rights if I am unfairly dismissed from my job?",
        "label": "Unfair dismissal (English)",
    },
    {
        "message": "Haki zangu ni zipi nikiachwa kazi bila sababu?",
        "label": "Unfair dismissal (Swahili)",
    },
    {
        "message": "What are the rules for land ownership in Kenya?",
        "label": "Land ownership (English)",
    },
    {
        "message": "What is the capital of France?",
        "label": "Out-of-scope question",
    },
]

# ── Runner ────────────────────────────────────────────────────────────────────

def run():
    failures = 0

    for case in CASES:
        print(f"── {case['label']} {'─' * (50 - len(case['label']))}")
        event = {"body": json.dumps({"message": case["message"]})}

        try:
            result = lambda_handler(event, None)
        except Exception as err:
            print(f"  [ERROR] {err}\n")
            failures += 1
            continue

        status = result["statusCode"]
        body = json.loads(result["body"])

        print(f"  Status:    {status}")
        print(f"  Language:  {body.get('language')}")
        print(f"  Blocked:   {body.get('blocked')}")
        print(f"  Citations: {len(body.get('citations', []))}")

        response = body.get("response", "")
        print(f"  Response:  {response[:200]}{'...' if len(response) > 200 else ''}")

        if body.get("citations"):
            first = body["citations"][0]
            print(f"  First citation: {first.get('source')} — {first.get('section')}")

        if status != 200:
            failures += 1

        print()

    return failures


def main():
    vectorstore = os.path.join(os.path.dirname(__file__), ".local-vectorstore")
    if not os.path.exists(vectorstore):
        print("Error: .local-vectorstore/ not found.")
        print("Run: ENV=local uv run ingest_local.py")
        sys.exit(1)

    print(f"Running {len(CASES)} end-to-end tests against local RAG...\n")
    failures = run()
    total = len(CASES)
    passed = total - failures
    print(f"Result: {passed}/{total} passed", end="")
    if failures:
        print(f"  ({failures} failed)")
        sys.exit(1)
    else:
        print(" ✓")


if __name__ == "__main__":
    main()
