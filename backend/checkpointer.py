"""
DynamoDB-backed LangGraph checkpointer.

Persists conversation state keyed by thread_id (= the frontend's sessionId)
so multi-turn memory survives Lambda cold starts and server_local.py restarts.

Table schema (single-table, composite key):
  thread_id    (S, HASH)   = LangGraph thread id
  sort_key     (S, RANGE)  = one of three prefixes:
                               "ckpt#{ns}#{checkpoint_id}"
                               "blob#{ns}#{channel}#{version}"
                               "write#{ns}#{checkpoint_id}#{task_id}#{idx}"

Binary payloads (checkpoint, metadata, blob, write value) are produced by
LangGraph's built-in serializer (`self.serde.dumps_typed`), which returns
a `(type_name, bytes)` tuple — we store both fields for faithful round-trip.

A TTL attribute `expires_at` is set on every item so abandoned conversations
auto-purge after 30 days.

The item layout intentionally mirrors `InMemorySaver`'s internal
`storage / blobs / writes` tripartite structure so `get_tuple`, `list`,
`put`, and `put_writes` can be a fairly direct translation of the
in-memory reference implementation.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


_DEFAULT_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _ckpt_sk(ns: str, checkpoint_id: str) -> str:
    return f"ckpt#{ns}#{checkpoint_id}"


def _blob_sk(ns: str, channel: str, version: str | int | float) -> str:
    return f"blob#{ns}#{channel}#{version}"


def _write_sk(ns: str, checkpoint_id: str, task_id: str, idx: int) -> str:
    # zero-pad with sign so lexicographic order matches numeric order
    return f"write#{ns}#{checkpoint_id}#{task_id}#{idx:+07d}"


class DynamoDBSaver(BaseCheckpointSaver[str]):
    """
    Checkpoint saver backed by a single DynamoDB table.

    Args:
        table: boto3 DynamoDB Table resource (`boto3.resource("dynamodb").Table(name)`).
        ttl_seconds: How long before items expire via DynamoDB TTL. Default 30 days.
        serde: Optional custom serializer; defaults to LangGraph's JsonPlusSerializer.
    """

    def __init__(
        self,
        table,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self._table = table
        self._ttl_seconds = ttl_seconds

    def _expires_at(self) -> int:
        return int(time.time()) + self._ttl_seconds

    # ── Blob loader (shared by get_tuple + list) ──────────────────────────────

    def _load_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        versions: ChannelVersions,
    ) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        for channel, version in versions.items():
            resp = self._table.get_item(
                Key={
                    "thread_id": thread_id,
                    "sort_key": _blob_sk(checkpoint_ns, channel, version),
                }
            )
            item = resp.get("Item")
            if not item:
                continue
            type_name = item["blob_type"]
            if type_name == "empty":
                continue
            data = bytes(item["blob_data"])
            channel_values[channel] = self.serde.loads_typed((type_name, data))
        return channel_values

    def _load_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        # All write items for this checkpoint share the prefix
        # `write#{ns}#{checkpoint_id}#`, so we can Query by BEGINS_WITH on sort_key.
        from boto3.dynamodb.conditions import Key

        prefix = f"write#{checkpoint_ns}#{checkpoint_id}#"
        resp = self._table.query(
            KeyConditionExpression=Key("thread_id").eq(thread_id)
            & Key("sort_key").begins_with(prefix),
        )
        out: list[tuple[str, str, Any]] = []
        for item in resp.get("Items", []):
            task_id = item["task_id"]
            channel = item["channel"]
            value = self.serde.loads_typed((item["value_type"], bytes(item["value_data"])))
            out.append((task_id, channel, value))
        return out

    # ── BaseCheckpointSaver implementation ────────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id:
            resp = self._table.get_item(
                Key={
                    "thread_id": thread_id,
                    "sort_key": _ckpt_sk(checkpoint_ns, checkpoint_id),
                }
            )
            item = resp.get("Item")
            if not item:
                return None
        else:
            # Latest checkpoint for this thread+ns: Query by prefix, descending, Limit=1.
            from boto3.dynamodb.conditions import Key

            prefix = f"ckpt#{checkpoint_ns}#"
            resp = self._table.query(
                KeyConditionExpression=Key("thread_id").eq(thread_id)
                & Key("sort_key").begins_with(prefix),
                ScanIndexForward=False,
                Limit=1,
            )
            items = resp.get("Items", [])
            if not items:
                return None
            item = items[0]
            checkpoint_id = item["checkpoint_id"]

        checkpoint_bytes = bytes(item["checkpoint_data"])
        metadata_bytes = bytes(item["metadata_data"])
        parent_checkpoint_id = item.get("parent_id") or None

        checkpoint_: Checkpoint = self.serde.loads_typed(
            (item["checkpoint_type"], checkpoint_bytes)
        )
        metadata: CheckpointMetadata = self.serde.loads_typed(
            (item["metadata_type"], metadata_bytes)
        )

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={
                **checkpoint_,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                ),
            },
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=self._load_writes(thread_id, checkpoint_ns, checkpoint_id),
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        # Simple implementation: only supports the common case of a specific
        # thread_id. Cross-thread listing isn't used by our handler.
        if not config:
            return
        from boto3.dynamodb.conditions import Key

        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        before_id = get_checkpoint_id(before) if before else None
        config_checkpoint_id = get_checkpoint_id(config)

        prefix = f"ckpt#{checkpoint_ns}#"
        resp = self._table.query(
            KeyConditionExpression=Key("thread_id").eq(thread_id)
            & Key("sort_key").begins_with(prefix),
            ScanIndexForward=False,
        )
        count = 0
        for item in resp.get("Items", []):
            checkpoint_id = item["checkpoint_id"]
            if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                continue
            if before_id and checkpoint_id >= before_id:
                continue
            metadata = self.serde.loads_typed(
                (item["metadata_type"], bytes(item["metadata_data"]))
            )
            if filter and not all(
                query_value == metadata.get(query_key)
                for query_key, query_value in filter.items()
            ):
                continue
            if limit is not None and count >= limit:
                break
            count += 1

            checkpoint_: Checkpoint = self.serde.loads_typed(
                (item["checkpoint_type"], bytes(item["checkpoint_data"]))
            )
            parent_checkpoint_id = item.get("parent_id") or None

            yield CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint={
                    **checkpoint_,
                    "channel_values": self._load_blobs(
                        thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                    ),
                },
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                pending_writes=self._load_writes(thread_id, checkpoint_ns, checkpoint_id),
            )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_id = config["configurable"].get("checkpoint_id")
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]

        # Blob items, one per new channel version.
        for channel, version in new_versions.items():
            if channel in values:
                type_name, data = self.serde.dumps_typed(values[channel])
            else:
                type_name, data = "empty", b""
            self._table.put_item(
                Item={
                    "thread_id": thread_id,
                    "sort_key": _blob_sk(checkpoint_ns, channel, version),
                    "blob_type": type_name,
                    "blob_data": data,
                    "expires_at": self._expires_at(),
                }
            )

        # Main checkpoint item.
        ckpt_type, ckpt_bytes = self.serde.dumps_typed(c)
        meta_type, meta_bytes = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )
        self._table.put_item(
            Item={
                "thread_id": thread_id,
                "sort_key": _ckpt_sk(checkpoint_ns, checkpoint_id),
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": ckpt_type,
                "checkpoint_data": ckpt_bytes,
                "metadata_type": meta_type,
                "metadata_data": meta_bytes,
                "parent_id": parent_id or "",
                "expires_at": self._expires_at(),
            }
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            type_name, data = self.serde.dumps_typed(value)
            # Skip duplicates for non-negative indices (matches InMemorySaver).
            sort_key = _write_sk(checkpoint_ns, checkpoint_id, task_id, write_idx)
            if write_idx >= 0:
                existing = self._table.get_item(
                    Key={"thread_id": thread_id, "sort_key": sort_key}
                ).get("Item")
                if existing:
                    continue
            self._table.put_item(
                Item={
                    "thread_id": thread_id,
                    "sort_key": sort_key,
                    "task_id": task_id,
                    "channel": channel,
                    "value_type": type_name,
                    "value_data": data,
                    "task_path": task_path,
                    "expires_at": self._expires_at(),
                }
            )
