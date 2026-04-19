"""
Local development HTTP server.

Serves POST /chat by calling lambda_handler directly in-process. Runs
outside Docker so LocalRAGAdapter can use chromadb.PersistentClient to
access the host ChromaDB store (.local-vectorstore/) directly.

This is the recommended way to test the full stack locally with the frontend.
The production code path (Lambda + API Gateway) is unchanged.

Usage:
  ENV=local uv run server_local.py

Then in frontend/.env.local:
  LOCAL_API_URL=http://localhost:8080
  VITE_API_BASE_URL=

Run both together:
  Terminal 1: cd backend && ENV=local uv run server_local.py
  Terminal 2: cd frontend && npm run dev

ChromaDB HTTP server (for Lambda-in-Docker testing) runs on port 8000:
  cd backend && uv run chroma run --path .local-vectorstore --port 8000 --host 0.0.0.0

Note: The LocalStack Lambda path uses CHROMA_HOST=host.docker.internal so
the container can reach ChromaDB via HTTP. This server runs in-process and
uses PersistentClient directly (no HTTP server needed) — set CHROMA_HOST=""
(default) so config.py routes to PersistentClient.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

# Must be set before importing handler so load_config() picks it up
os.environ.setdefault("ENV", "local")
os.environ.setdefault("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
# In-process: use PersistentClient, not the HTTP server
os.environ.setdefault("CHROMA_HOST", "")

sys.path.insert(0, os.path.dirname(__file__))

from handler import lambda_handler

PORT = int(os.environ.get("PORT", 8080))


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/chat":
            self._send(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")

        result = lambda_handler({"body": body}, None)

        self.send_response(result["statusCode"])
        for key, val in result.get("headers", {}).items():
            self.send_header(key, val)
        self.end_headers()
        self.wfile.write(result["body"].encode("utf-8"))

    def do_OPTIONS(self):
        # Handle CORS preflight so browser fetch works without the Vite proxy
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send(self, status: int, body: dict):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        # Replace default noisy logging with a single clean line per request
        print(f"  {self.command} {self.path} → {args[1]}")


if __name__ == "__main__":
    vectorstore = os.path.join(os.path.dirname(__file__), ".local-vectorstore")
    if not os.path.exists(vectorstore):
        print("Error: .local-vectorstore/ not found.")
        print("Run: ENV=local uv run ingest_local.py")
        sys.exit(1)

    print(f"Local API server → http://localhost:{PORT}/chat")
    print("(Ctrl-C to stop)\n")
    HTTPServer(("", PORT), _Handler).serve_forever()
