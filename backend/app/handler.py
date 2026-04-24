"""
Haki AI — Lambda handler.

Thin entry point that:
  1. Parses the request (message, sessionId).
  2. Invokes the compiled LangGraph with thread_id = sessionId.
  3. Maps the final graph state back to the existing JSON response shape.

All orchestration (language detection, classify_intent, rag_node, chat_node)
now lives in graph.py. Memory is persisted via DynamoDBSaver, keyed by the
frontend-provided sessionId.

Signed-in routes (Clerk):
  - POST /chat accepts an optional ``Authorization: Bearer <clerk_jwt>``
    header. When present and valid we record the thread under the user's
    id in the chat-threads table and generate a short title on the first
    turn. Anonymous turns behave identically to before.
  - GET  /chat/threads            — list the user's threads
  - PATCH /chat/threads           — rename a thread  { threadId, title }
  - POST /chat/threads/claim      — attach the current sessionId to the
                                     signed-in user (post-sign-in hook).
"""

import json
import uuid

from clients import make_bedrock_runtime, make_cloudwatch, make_dynamodb_table
from app.config import load_config

# NOTE: observability MUST be imported and bootstrapped before `graph` so
# LANGSMITH_API_KEY lands in the environment before LangChain initialises
# its tracer. Importing graph triggers the langchain import chain.
from observability.tracing import bootstrap_langsmith, run_traced_turn

bootstrap_langsmith(load_config())

from app.auth import extract_bearer, verify_clerk_jwt  # noqa: E402
from app.graph import get_compiled_graph, load_history  # noqa: E402
from agents.title import generate_title  # noqa: E402
from memory.threads import ThreadsRepo  # noqa: E402
from observability.metrics import elapsed_ms, emit_metrics, now_ms  # noqa: E402


def lambda_handler(event, context):
    """
    Main Lambda entry point. Called by API Gateway HTTP API.

    Routes:
      POST  /chat                         → chat turn (optional Bearer auth)
      GET   /chat/history?sessionId=      → full persisted conversation
      GET   /chat/threads                 → list signed-in user's threads
      PATCH /chat/threads                 → rename { threadId, title }
      POST  /chat/threads/claim           → claim { threadId } for signed-in user
    """
    method, path = _method_and_path(event)
    clean_path = path.rstrip("/") or "/"

    if method == "GET" and clean_path == "/chat/history":
        return _handle_history(event)
    if clean_path == "/chat/threads":
        if method == "GET":
            return _handle_list_threads(event)
        if method == "PATCH":
            return _handle_rename_thread(event)
    if method == "POST" and clean_path == "/chat/threads/claim":
        return _handle_claim_thread(event)
    return _handle_chat(event)


# ── Chat turn ────────────────────────────────────────────────────────────────


def _handle_chat(event):
    config = load_config()
    cloudwatch = make_cloudwatch(config)
    start = now_ms()
    language = "english"  # default for metrics if we fail before detection

    try:
        body = json.loads(event.get("body") or "{}")
        user_message = body.get("message", "").strip()
        if not user_message:
            return _response(400, {"error": "message is required"})

        session_id = (body.get("sessionId") or "").strip() or str(uuid.uuid4())
        user_id = _verified_user_id(event, config)

        # Thread ownership gate. If the thread_id is already claimed by a
        # signed-in user, only that owner may continue the conversation —
        # neither anonymous callers nor a different signed-in user can
        # write into or read the LangGraph checkpoint for it. Unowned
        # threads fall through to the existing behaviour so brand-new
        # sessions and anonymous-only flows stay unchanged.
        owner = _thread_owner(config, session_id)
        if owner is not None and owner != user_id:
            return _response(403, {"error": "thread does not belong to this user"})

        graph = get_compiled_graph(config)
        final_state = run_traced_turn(
            graph,
            {"messages": [{"role": "user", "content": user_message}]},
            {"configurable": {"thread_id": session_id}},
            session_id=session_id,
            env=config.environment,
            message_length=len(user_message),
        )

        language = final_state.get("language", "english")
        blocked = bool(final_state.get("blocked", False))
        citations = final_state.get("citations", [])
        response_text = final_state.get("response_text", "")

        emit_metrics(
            cloudwatch,
            language=language,
            latency_ms=elapsed_ms(start),
            blocked=blocked,
            citations=citations,
        )

        # Index the thread and mint a title on the first signed-in turn.
        if user_id:
            try:
                _record_signed_in_turn(
                    config=config,
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_text=response_text,
                )
            except Exception as err:  # noqa: BLE001
                # Failing the thread index must never fail the chat reply.
                print(f"Thread index update failed: {err}")

        return _response(200, {
            "response": response_text,
            "citations": citations,
            "language": language,
            "blocked": blocked,
            "sessionId": session_id,
        })

    except Exception as err:
        print(f"Unhandled error (chat): {err}")
        emit_metrics(
            cloudwatch,
            language=language,
            latency_ms=elapsed_ms(start),
            blocked=False,
            citations=[],
            failed=True,
        )
        return _response(500, {"error": "Internal server error"})


def _record_signed_in_turn(
    *,
    config,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_text: str,
) -> None:
    """
    Upserts the thread row, generating a title the first time we see the
    ``(user_id, session_id)`` pair. On subsequent turns we only bump
    ``updated_at`` so the sidebar stays sorted by recency without
    overwriting a user-edited title.
    """
    repo = _threads_repo(config)
    existing = repo.get(user_id, session_id)
    if existing is None:
        title = generate_title(
            user_message,
            assistant_text,
            make_bedrock_runtime(config),
            config.bedrock_model_id,
        )
        repo.upsert(user_id, session_id, title=title)
    else:
        repo.upsert(user_id, session_id)


# ── History ──────────────────────────────────────────────────────────────────


def _handle_history(event):
    """
    Returns { messages: [{id, role, content, citations?, language?, blocked?}], sessionId }.
    Safe for brand-new sessions — returns an empty list.

    Ownership is enforced exactly as on ``POST /chat``: if the thread is
    already claimed by a signed-in user, only that owner may read its
    messages. Unowned threads remain readable by anyone with the UUID,
    which preserves the anonymous flow's behaviour (clients persist the
    sessionId in localStorage on their own device, so in practice this is
    still private to the device that minted it).
    """
    try:
        config = load_config()
        params = (event.get("queryStringParameters") or {})
        session_id = (params.get("sessionId") or "").strip()
        if not session_id:
            return _response(400, {"error": "sessionId is required"})
        user_id = _verified_user_id(event, config)
        owner = _thread_owner(config, session_id)
        if owner is not None and owner != user_id:
            return _response(403, {"error": "thread does not belong to this user"})
        messages = load_history(config, session_id)
        return _response(200, {"messages": messages, "sessionId": session_id})
    except Exception as err:
        print(f"Unhandled error (history): {err}")
        return _response(500, {"error": "Internal server error"})


# ── Threads (signed-in only) ─────────────────────────────────────────────────


def _handle_list_threads(event):
    config = load_config()
    user_id = _verified_user_id(event, config)
    if not user_id:
        return _response(401, {"error": "Authentication required"})
    try:
        repo = _threads_repo(config)
        threads = [row.to_api() for row in repo.list_for_user(user_id)]
        return _response(200, {"threads": threads})
    except Exception as err:
        print(f"Unhandled error (list threads): {err}")
        return _response(500, {"error": "Internal server error"})


def _handle_rename_thread(event):
    config = load_config()
    user_id = _verified_user_id(event, config)
    if not user_id:
        return _response(401, {"error": "Authentication required"})
    try:
        body = json.loads(event.get("body") or "{}")
        thread_id = (body.get("threadId") or "").strip()
        title = (body.get("title") or "").strip()
        if not thread_id or not title:
            return _response(400, {"error": "threadId and title are required"})
        repo = _threads_repo(config)
        row = repo.update_title(user_id, thread_id, title)
        if row is None:
            return _response(404, {"error": "thread not found"})
        return _response(200, {"thread": row.to_api()})
    except Exception as err:
        print(f"Unhandled error (rename thread): {err}")
        return _response(500, {"error": "Internal server error"})


def _handle_claim_thread(event):
    """
    Attaches an anonymous sessionId to the now-signed-in user so it shows
    up in their thread list. Idempotent when the caller already owns the
    thread, and rejected with 403 when the thread is owned by a different
    user — prevents hijacking someone else's history by sending their
    threadId to the claim endpoint post-sign-in.
    """
    config = load_config()
    user_id = _verified_user_id(event, config)
    if not user_id:
        return _response(401, {"error": "Authentication required"})
    try:
        body = json.loads(event.get("body") or "{}")
        thread_id = (body.get("threadId") or "").strip()
        if not thread_id:
            return _response(400, {"error": "threadId is required"})
        repo = _threads_repo(config)
        owner = repo.find_owner(thread_id)
        if owner is not None and owner != user_id:
            return _response(403, {"error": "thread does not belong to this user"})
        row = repo.upsert(user_id, thread_id)
        return _response(200, {"thread": row.to_api()})
    except Exception as err:
        print(f"Unhandled error (claim thread): {err}")
        return _response(500, {"error": "Internal server error"})


# ── Helpers ──────────────────────────────────────────────────────────────────


def _verified_user_id(event, config) -> str | None:
    token = extract_bearer(event)
    if not token:
        return None
    return verify_clerk_jwt(token, config.clerk_publishable_key)


def _threads_repo(config) -> ThreadsRepo:
    table = make_dynamodb_table(config, config.chat_threads_table)
    return ThreadsRepo(table)


def _thread_owner(config, session_id: str) -> str | None:
    """
    Thin wrapper around ``ThreadsRepo.find_owner`` that swallows transient
    DynamoDB errors so an ownership-index outage cannot harden the chat
    path into a full brownout. We fail open (treat as unowned) and log —
    the worst-case regression is temporary exposure of anonymous threads,
    which already had no owner, so the user-visible impact is nil.
    """
    if not session_id:
        return None
    try:
        return _threads_repo(config).find_owner(session_id)
    except Exception as err:  # noqa: BLE001
        print(f"Thread ownership lookup failed: {err}")
        return None


def _method_and_path(event) -> tuple[str, str]:
    """
    Extracts (method, path) from an API Gateway v2 HTTP event. Falls back to
    top-level keys when called from server_local.py which hand-rolls the event.
    """
    http = (event.get("requestContext") or {}).get("http") or {}
    method = http.get("method") or event.get("httpMethod") or event.get("method") or "POST"
    path = http.get("path") or event.get("rawPath") or event.get("path") or "/chat"
    return method.upper(), path


def _response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
        },
        "body": json.dumps(body),
    }
