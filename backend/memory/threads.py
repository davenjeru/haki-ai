"""
Thread index (signed-in only).

Thin DynamoDB-backed catalogue of ``(user_id, thread_id) -> title`` rows that
power the "your chats" sidebar for authenticated users. The actual message
history still lives in the existing LangGraph checkpointer table — this
module deliberately knows nothing about messages.

Table schema (single-table, composite key):

  user_id     (S, HASH)   — Clerk user id (``sub`` from the session JWT)
  thread_id   (S, RANGE)  — same value as LangGraph's ``thread_id``
  title                   — human-readable chat title (LLM-generated + editable)
  created_at  (N)         — unix seconds
  updated_at  (N)         — unix seconds, bumped on rename and on each turn
  expires_at  (N)         — unix seconds; DynamoDB TTL purges abandoned rows

Rows are tiny (< 256 B), so we intentionally skip pagination in
:func:`list_threads` — the per-user row count is bounded by how many
chats a single human starts, which realistically fits inside one
``Query`` response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


_DEFAULT_TTL_SECONDS = 180 * 24 * 3600  # 180 days

_DEFAULT_TITLE = "New chat"


def _now() -> int:
    return int(time.time())


def _expires_at() -> int:
    return _now() + _DEFAULT_TTL_SECONDS


@dataclass(frozen=True)
class ThreadRow:
    user_id: str
    thread_id: str
    title: str
    created_at: int
    updated_at: int

    def to_api(self) -> dict:
        """Shape consumed by the frontend — omits the owning user_id."""
        return {
            "threadId": self.thread_id,
            "title": self.title,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


def _row_from_item(item: dict) -> ThreadRow:
    return ThreadRow(
        user_id=item["user_id"],
        thread_id=item["thread_id"],
        title=item.get("title") or _DEFAULT_TITLE,
        created_at=int(item.get("created_at") or 0),
        updated_at=int(item.get("updated_at") or 0),
    )


class ThreadsRepo:
    """
    DynamoDB-backed repository for the per-user thread index.

    Constructed once per handler invocation (clients are cheap) from a
    boto3 Table resource produced by :func:`clients.make_dynamodb_table`.
    """

    def __init__(self, table, *, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._table = table
        self._ttl_seconds = ttl_seconds

    # ── Reads ────────────────────────────────────────────────────────────

    def get(self, user_id: str, thread_id: str) -> ThreadRow | None:
        if not user_id or not thread_id:
            return None
        resp = self._table.get_item(
            Key={"user_id": user_id, "thread_id": thread_id},
        )
        item = resp.get("Item")
        return _row_from_item(item) if item else None

    def list_for_user(self, user_id: str) -> list[ThreadRow]:
        if not user_id:
            return []
        from boto3.dynamodb.conditions import Key

        resp = self._table.query(
            KeyConditionExpression=Key("user_id").eq(user_id),
        )
        rows = [_row_from_item(it) for it in resp.get("Items", [])]
        rows.sort(key=lambda r: r.updated_at, reverse=True)
        return rows

    def find_owner(self, thread_id: str) -> str | None:
        """
        Returns the ``user_id`` that owns a given thread, or ``None`` when
        no user has claimed it. Backed by the ``thread_id_index`` GSI so
        the call is a single DynamoDB Query with ``KEYS_ONLY`` projection
        — no payload bytes, one RCU at most.

        The API Gateway handler uses this to reject cross-user access to
        ``/chat``, ``/chat/history``, and ``/chat/threads/claim``.
        """
        if not thread_id:
            return None
        from boto3.dynamodb.conditions import Key

        resp = self._table.query(
            IndexName="thread_id_index",
            KeyConditionExpression=Key("thread_id").eq(thread_id),
            Limit=1,
        )
        items = resp.get("Items", [])
        if not items:
            return None
        owner = items[0].get("user_id")
        return owner if isinstance(owner, str) and owner else None

    # ── Writes ───────────────────────────────────────────────────────────

    def upsert(
        self,
        user_id: str,
        thread_id: str,
        *,
        title: str | None = None,
    ) -> ThreadRow:
        """
        Creates the row if missing, otherwise bumps ``updated_at``.

        When ``title`` is ``None`` we leave an existing title alone; a
        caller that wants to force-update the title should pass an explicit
        string. This lets the handler cheaply touch the row on every turn
        (to refresh sort order) without stomping on a user-edited title.
        """
        now = _now()
        existing = self.get(user_id, thread_id)
        if existing is None:
            item = {
                "user_id": user_id,
                "thread_id": thread_id,
                "title": (title or _DEFAULT_TITLE).strip() or _DEFAULT_TITLE,
                "created_at": now,
                "updated_at": now,
                "expires_at": now + self._ttl_seconds,
            }
            self._table.put_item(Item=item)
            return _row_from_item(item)

        updates: dict[str, dict] = {
            "updated_at": {"Value": now, "Action": "PUT"},
            "expires_at": {"Value": now + self._ttl_seconds, "Action": "PUT"},
        }
        if title is not None:
            clean = title.strip()
            if clean:
                updates["title"] = {"Value": clean, "Action": "PUT"}
        self._table.update_item(
            Key={"user_id": user_id, "thread_id": thread_id},
            AttributeUpdates=updates,
        )
        return ThreadRow(
            user_id=user_id,
            thread_id=thread_id,
            title=(title.strip() if title and title.strip() else existing.title),
            created_at=existing.created_at,
            updated_at=now,
        )

    def update_title(self, user_id: str, thread_id: str, title: str) -> ThreadRow | None:
        """Renames an existing thread. Returns ``None`` if the row is missing."""
        clean = (title or "").strip()
        if not clean:
            return None
        existing = self.get(user_id, thread_id)
        if existing is None:
            return None
        now = _now()
        self._table.update_item(
            Key={"user_id": user_id, "thread_id": thread_id},
            AttributeUpdates={
                "title": {"Value": clean, "Action": "PUT"},
                "updated_at": {"Value": now, "Action": "PUT"},
                "expires_at": {"Value": now + self._ttl_seconds, "Action": "PUT"},
            },
        )
        return ThreadRow(
            user_id=user_id,
            thread_id=thread_id,
            title=clean,
            created_at=existing.created_at,
            updated_at=now,
        )
