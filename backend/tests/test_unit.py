"""
Unit tests for backend business logic.

No AWS or LocalStack required — all external dependencies are stubbed.
Tests cover edge cases not exercised by test_language_detection.py or
test_e2e_local.py.

Usage:
  uv run test_unit.py
"""

import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from adapters import ComprehendAdapter
from chat_node import invoke_chat
from checkpointer import DynamoDBSaver
from citations import extract_citations, refresh_presigned_urls
from classifier import classify_intent, _parse_needs_rag
from graph import _detect_language
import observability as obs
from observability import _trace_metadata, bootstrap_langsmith
from handler import lambda_handler
from prompts import (
    BILINGUAL_REFUSAL,
    CLASSIFIER_PROMPT,
    build_chat_system_prompt,
    build_system_prompt,
)
from rag import blocked_response, check_guardrail_block


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _FixedComprehendClient:
    """Returns a fixed language list."""
    def __init__(self, languages: list[dict]):
        self._languages = languages

    def detect_dominant_language(self, Text: str) -> dict:
        return {"Languages": self._languages}


def _adapter(languages: list[dict]) -> ComprehendAdapter:
    return ComprehendAdapter(_FixedComprehendClient(languages), is_local=False)


class _FakeBedrockRuntime:
    """
    Stub for bedrock-runtime invoke_model. Returns a canned response body
    shaped like Claude's InvokeModel output: {"content": [{"text": ...}], ...}.
    """
    def __init__(self, text: str = "(stubbed)", stop_reason: str = "end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls: list[dict] = []  # captured request payloads

    def invoke_model(self, *, modelId, body, contentType, accept):
        self.calls.append({
            "modelId": modelId,
            "body": json.loads(body),
            "contentType": contentType,
            "accept": accept,
        })
        payload = json.dumps({
            "content": [{"text": self.text}],
            "stop_reason": self.stop_reason,
        }).encode()
        return {"body": io.BytesIO(payload)}


# ── detect_language ───────────────────────────────────────────────────────────

class TestDetectLanguage(unittest.TestCase):

    def test_high_confidence_english(self):
        adapter = _adapter([{"LanguageCode": "en", "Score": 0.97}])
        self.assertEqual(_detect_language("hello", adapter), "english")

    def test_high_confidence_swahili(self):
        adapter = _adapter([{"LanguageCode": "sw", "Score": 0.91}])
        self.assertEqual(_detect_language("habari", adapter), "swahili")

    def test_mixed_when_both_languages_present(self):
        adapter = _adapter([
            {"LanguageCode": "en", "Score": 0.72},
            {"LanguageCode": "sw", "Score": 0.26},
        ])
        self.assertEqual(_detect_language("hello habari", adapter), "mixed")

    def test_english_below_threshold_no_swahili_defaults_english(self):
        adapter = _adapter([{"LanguageCode": "en", "Score": 0.70}])
        self.assertEqual(_detect_language("bonjour", adapter), "english")

    def test_swahili_below_threshold_no_english_still_swahili(self):
        adapter = _adapter([{"LanguageCode": "sw", "Score": 0.70}])
        self.assertEqual(_detect_language("habari", adapter), "swahili")

    def test_unknown_language_defaults_english(self):
        adapter = _adapter([{"LanguageCode": "fr", "Score": 0.95}])
        self.assertEqual(_detect_language("bonjour", adapter), "english")

    def test_empty_language_list_defaults_english(self):
        adapter = _adapter([])
        self.assertEqual(_detect_language("", adapter), "english")


# ── build_system_prompt ───────────────────────────────────────────────────────

class TestBuildSystemPrompt(unittest.TestCase):

    def test_english_prompt_contains_english_instruction(self):
        self.assertIn("Respond in English.", build_system_prompt("english"))

    def test_swahili_prompt_contains_swahili_instruction(self):
        self.assertIn("Kiswahili", build_system_prompt("swahili"))

    def test_mixed_prompt_contains_both_language_instructions(self):
        prompt = build_system_prompt("mixed")
        self.assertIn("English", prompt)
        self.assertIn("Swahili", prompt)

    def test_unknown_language_falls_back_to_english(self):
        self.assertIn("Respond in English.", build_system_prompt("klingon"))

    def test_all_prompts_contain_citation_rule(self):
        for lang in ("english", "swahili", "mixed"):
            with self.subTest(lang=lang):
                self.assertIn("citation", build_system_prompt(lang).lower())

    def test_all_prompts_contain_scope_rule(self):
        for lang in ("english", "swahili", "mixed"):
            with self.subTest(lang=lang):
                self.assertIn("Kenyan law", build_system_prompt(lang))

    def test_all_prompts_contain_bilingual_refusal(self):
        for lang in ("english", "swahili", "mixed"):
            with self.subTest(lang=lang):
                self.assertIn(BILINGUAL_REFUSAL, build_system_prompt(lang))


# ── build_chat_system_prompt ──────────────────────────────────────────────────

class TestBuildChatSystemPrompt(unittest.TestCase):

    def test_chat_prompt_is_not_rag_prompt(self):
        self.assertNotEqual(build_chat_system_prompt("english"), build_system_prompt("english"))

    def test_chat_prompt_contains_scope_and_language(self):
        prompt = build_chat_system_prompt("english")
        self.assertIn("Haki AI", prompt)
        self.assertIn("Respond in English.", prompt)

    def test_chat_prompt_refuses_non_legal_substantive(self):
        prompt = build_chat_system_prompt("english")
        self.assertIn(BILINGUAL_REFUSAL, prompt)

    def test_chat_prompt_does_not_demand_citations(self):
        # The RAG prompt requires citations; the chat prompt explicitly does not.
        prompt = build_chat_system_prompt("english")
        self.assertNotIn("MUST include a citation", prompt)


# ── check_guardrail_block ─────────────────────────────────────────────────────

class TestCheckGuardrailBlock(unittest.TestCase):

    def test_guardrail_intervened_returns_true(self):
        self.assertTrue(check_guardrail_block({"stopReason": "guardrail_intervened"}))

    def test_end_turn_returns_false(self):
        self.assertFalse(check_guardrail_block({"stopReason": "end_turn"}))

    def test_missing_stop_reason_returns_false(self):
        self.assertFalse(check_guardrail_block({}))

    def test_blocked_response_contains_bilingual_refusal(self):
        self.assertEqual(blocked_response("english"), BILINGUAL_REFUSAL)


# ── extract_citations ─────────────────────────────────────────────────────────

class TestExtractCitations(unittest.TestCase):

    def _ref(self, chunk_id, **meta_overrides):
        meta = {
            "source": "Employment Act 2007",
            "chapter": "Part III",
            "section": "Section 45",
            "title": "Unfair termination",
            "chunkId": chunk_id,
            "pageImageKey": f"page-images/employment-act-2007/page-45.pdf",
        }
        meta.update(meta_overrides)
        return {
            "content": {"text": "some text"},
            "metadata": meta,
            "location": {"type": "S3", "s3Location": {"uri": f"s3://haki-ai-data/processed-chunks/{chunk_id}.txt"}},
        }

    def _rag_result(self, refs: list) -> dict:
        return {
            "output": {"text": "answer"},
            "citations": [{"retrievedReferences": refs}],
            "stopReason": "end_turn",
        }

    def test_returns_citations_in_order(self):
        refs = [self._ref("chunk-1"), self._ref("chunk-2")]
        result = extract_citations(self._rag_result(refs))
        self.assertEqual([c["chunkId"] for c in result], ["chunk-1", "chunk-2"])

    def test_deduplicates_by_chunk_id(self):
        refs = [self._ref("chunk-1"), self._ref("chunk-1"), self._ref("chunk-2")]
        result = extract_citations(self._rag_result(refs))
        self.assertEqual(len(result), 2)

    def test_page_image_url_included_when_key_and_s3_client_present(self):
        class _FakeS3:
            def generate_presigned_url(self, op, Params, ExpiresIn):
                return f"https://example.test/{Params['Bucket']}/{Params['Key']}?sig=x"

        result = extract_citations(
            self._rag_result([self._ref("chunk-1")]),
            s3_client=_FakeS3(),
            bucket="haki-ai-data",
        )
        self.assertIn("pageImageUrl", result[0])

    def test_page_image_key_is_persisted_alongside_url(self):
        # The key must be stored so history hydration can re-presign later.
        class _FakeS3:
            def generate_presigned_url(self, op, Params, ExpiresIn):
                return "https://example.test/signed"

        result = extract_citations(
            self._rag_result([self._ref("chunk-1")]),
            s3_client=_FakeS3(),
            bucket="haki-ai-data",
        )
        self.assertEqual(result[0]["pageImageKey"], "page-images/employment-act-2007/page-45.pdf")


    def test_page_image_url_absent_when_no_s3_client(self):
        result = extract_citations(self._rag_result([self._ref("chunk-1")]))
        self.assertNotIn("pageImageUrl", result[0])

    def test_page_image_url_rejects_keys_outside_page_images_prefix(self):
        ref = self._ref("chunk-1", pageImageKey="processed-chunks/evil.pdf")
        class _FakeS3:
            def generate_presigned_url(self, **_):
                raise AssertionError("should not sign keys outside page-images/")
        result = extract_citations(
            self._rag_result([ref]), s3_client=_FakeS3(), bucket="haki-ai-data",
        )
        self.assertNotIn("pageImageUrl", result[0])

    def test_empty_citations_returns_empty_list(self):
        self.assertEqual(
            extract_citations({"output": {"text": ""}, "citations": [], "stopReason": "end_turn"}),
            [],
        )

    def test_falls_back_to_s3_uri_when_chunk_id_missing(self):
        ref = self._ref("chunk-1")
        del ref["metadata"]["chunkId"]
        result = extract_citations(self._rag_result([ref]))
        self.assertEqual(result[0]["chunkId"], "s3://haki-ai-data/processed-chunks/chunk-1.txt")

    def test_missing_metadata_fields_return_empty_strings(self):
        ref = {"content": {"text": "text"}, "metadata": {"chunkId": "chunk-x"}, "location": {}}
        result = extract_citations(self._rag_result([ref]))
        self.assertEqual(result[0]["source"], "")


class TestRefreshPresignedUrls(unittest.TestCase):
    """Re-presigns persisted citations on history hydration."""

    class _FakeS3:
        def __init__(self):
            self.calls = 0
        def generate_presigned_url(self, op, Params, ExpiresIn):
            self.calls += 1
            return f"https://fresh.test/{Params['Key']}?nonce={self.calls}"

    def test_strips_stale_url_and_signs_fresh_one_from_key(self):
        s3 = self._FakeS3()
        citations = [{
            "source": "Employment Act 2007",
            "chunkId": "chunk-1",
            "pageImageKey": "page-images/employment-act-2007/page-45.pdf",
            "pageImageUrl": "https://stale.test/expired",
        }]
        refreshed = refresh_presigned_urls(citations, s3_client=s3, bucket="haki-ai-data")
        self.assertIn("nonce=1", refreshed[0]["pageImageUrl"])
        self.assertEqual(refreshed[0]["pageImageKey"], citations[0]["pageImageKey"])
        self.assertEqual(citations[0]["pageImageUrl"], "https://stale.test/expired")

    def test_passes_through_citations_without_key(self):
        s3 = self._FakeS3()
        citations = [{"source": "X", "chunkId": "c"}]
        refreshed = refresh_presigned_urls(citations, s3_client=s3, bucket="haki-ai-data")
        self.assertNotIn("pageImageUrl", refreshed[0])
        self.assertEqual(s3.calls, 0)


# ── classifier ────────────────────────────────────────────────────────────────

class TestClassifierParsing(unittest.TestCase):
    """Parses the model's raw reply into a bool."""

    def test_parses_true(self):
        self.assertTrue(_parse_needs_rag('{"needs_rag": true}'))

    def test_parses_false(self):
        self.assertFalse(_parse_needs_rag('{"needs_rag": false}'))

    def test_extracts_json_from_surrounding_text(self):
        self.assertTrue(_parse_needs_rag('Here you go: {"needs_rag": true} cheers.'))

    def test_defaults_to_true_on_malformed_json(self):
        self.assertTrue(_parse_needs_rag("not json at all"))

    def test_defaults_to_true_when_key_missing(self):
        self.assertTrue(_parse_needs_rag('{"other_key": false}'))

    def test_defaults_to_true_when_value_not_bool(self):
        self.assertTrue(_parse_needs_rag('{"needs_rag": "yes"}'))


class TestClassifyIntent(unittest.TestCase):
    """End-to-end classifier flow with a stubbed Bedrock client."""

    def test_returns_true_when_model_says_true(self):
        br = _FakeBedrockRuntime(text='{"needs_rag": true}')
        self.assertTrue(classify_intent(
            [{"role": "user", "content": "What does section 40 say?"}],
            br,
            "claude-haiku",
            CLASSIFIER_PROMPT,
        ))

    def test_returns_false_when_model_says_false(self):
        br = _FakeBedrockRuntime(text='{"needs_rag": false}')
        self.assertFalse(classify_intent(
            [{"role": "user", "content": "My name is Dave"}],
            br,
            "claude-haiku",
            CLASSIFIER_PROMPT,
        ))

    def test_sends_classifier_prompt_as_system(self):
        br = _FakeBedrockRuntime(text='{"needs_rag": false}')
        classify_intent(
            [{"role": "user", "content": "hi"}],
            br,
            "claude-haiku",
            CLASSIFIER_PROMPT,
        )
        self.assertEqual(br.calls[0]["body"]["system"], CLASSIFIER_PROMPT)

    def test_includes_recent_turns_in_transcript(self):
        br = _FakeBedrockRuntime(text='{"needs_rag": false}')
        history = [
            {"role": "user", "content": "My name is Dave"},
            {"role": "assistant", "content": "Nice to meet you, Dave."},
            {"role": "user", "content": "What is my name?"},
        ]
        classify_intent(history, br, "claude-haiku", CLASSIFIER_PROMPT)
        transcript = br.calls[0]["body"]["messages"][0]["content"]
        self.assertIn("Dave", transcript)
        self.assertIn("What is my name?", transcript)


# ── chat_node.invoke_chat ─────────────────────────────────────────────────────

class TestInvokeChat(unittest.TestCase):

    def test_returns_model_text(self):
        br = _FakeBedrockRuntime(text="Your name is Dave.")
        reply = invoke_chat(
            [{"role": "user", "content": "What is my name?"}],
            "english",
            br,
            "claude-haiku",
        )
        self.assertEqual(reply, "Your name is Dave.")

    def test_sends_chat_system_prompt(self):
        br = _FakeBedrockRuntime(text="ok")
        invoke_chat(
            [{"role": "user", "content": "Hi"}],
            "english",
            br,
            "claude-haiku",
        )
        self.assertEqual(br.calls[0]["body"]["system"], build_chat_system_prompt("english"))

    def test_passes_full_history(self):
        br = _FakeBedrockRuntime(text="ok")
        history = [
            {"role": "user", "content": "My name is Dave"},
            {"role": "assistant", "content": "Nice to meet you, Dave."},
            {"role": "user", "content": "What is my name?"},
        ]
        invoke_chat(history, "english", br, "claude-haiku")
        sent = br.calls[0]["body"]["messages"]
        self.assertEqual(len(sent), 3)
        self.assertEqual(sent[-1]["content"], "What is my name?")

    def test_skips_empty_or_invalid_messages(self):
        br = _FakeBedrockRuntime(text="ok")
        history = [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "real question"},
        ]
        invoke_chat(history, "english", br, "claude-haiku")
        sent = br.calls[0]["body"]["messages"]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["content"], "real question")


# ── DynamoDBSaver ─────────────────────────────────────────────────────────────

class _InMemoryDynamoTable:
    """
    Minimal stand-in for boto3.resource('dynamodb').Table. Supports the
    operations DynamoDBSaver uses: put_item, get_item, query. Items are
    stored keyed by (thread_id, sort_key) in a dict; query does a
    begins_with filter on sort_key and honours ScanIndexForward + Limit.
    """

    def __init__(self):
        self.items: dict[tuple[str, str], dict] = {}

    def put_item(self, *, Item):
        self.items[(Item["thread_id"], Item["sort_key"])] = dict(Item)

    def get_item(self, *, Key):
        item = self.items.get((Key["thread_id"], Key["sort_key"]))
        return {"Item": item} if item else {}

    def query(self, *, KeyConditionExpression, ScanIndexForward=True, Limit=None):
        # Dig the thread_id + prefix out of the Boto3 condition object.
        expr = KeyConditionExpression.get_expression()
        values = expr["values"]
        thread_cond = values[0].get_expression()
        sort_cond = values[1].get_expression()
        thread_id = thread_cond["values"][1]
        prefix = sort_cond["values"][1]

        matches = [
            item for (tid, sk), item in self.items.items()
            if tid == thread_id and sk.startswith(prefix)
        ]
        matches.sort(key=lambda it: it["sort_key"], reverse=not ScanIndexForward)
        if Limit is not None:
            matches = matches[:Limit]
        return {"Items": matches}


class TestDynamoDBSaver(unittest.TestCase):
    """
    Verifies round-trip of a checkpoint via the in-memory DynamoDB stub.
    Uses a minimal Checkpoint dict constructed by hand rather than running
    a full graph — keeps the test hermetic and focused on storage.
    """

    def _saver(self) -> tuple[DynamoDBSaver, _InMemoryDynamoTable]:
        table = _InMemoryDynamoTable()
        return DynamoDBSaver(table), table

    def _config(self, thread_id: str = "t1") -> dict:
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}

    def _checkpoint(self, ckpt_id: str = "c1") -> dict:
        return {
            "v": 1,
            "id": ckpt_id,
            "ts": "2026-01-01T00:00:00+00:00",
            "channel_values": {"messages": [{"role": "user", "content": "hi"}]},
            "channel_versions": {"messages": "1.0"},
            "versions_seen": {},
        }

    def test_put_then_get_tuple_returns_same_values(self):
        saver, _ = self._saver()
        cfg = self._config()
        ckpt = self._checkpoint()
        saver.put(cfg, ckpt, {"source": "input"}, {"messages": "1.0"})

        tup = saver.get_tuple(cfg)
        self.assertIsNotNone(tup)
        self.assertEqual(tup.checkpoint["id"], "c1")
        self.assertEqual(
            tup.checkpoint["channel_values"]["messages"],
            [{"role": "user", "content": "hi"}],
        )

    def test_get_tuple_returns_latest_when_multiple_checkpoints(self):
        saver, _ = self._saver()
        cfg = self._config()
        saver.put(cfg, self._checkpoint("c1"), {"source": "input"}, {"messages": "1.0"})
        saver.put(cfg, self._checkpoint("c2"), {"source": "input"}, {"messages": "2.0"})
        tup = saver.get_tuple(cfg)
        self.assertEqual(tup.checkpoint["id"], "c2")

    def test_get_tuple_returns_none_for_unknown_thread(self):
        saver, _ = self._saver()
        self.assertIsNone(saver.get_tuple(self._config("unknown")))

    def test_put_writes_stores_and_load_via_get_tuple(self):
        saver, _ = self._saver()
        cfg = self._config()
        saver.put(cfg, self._checkpoint(), {"source": "input"}, {"messages": "1.0"})
        write_cfg = {"configurable": {"thread_id": "t1", "checkpoint_ns": "", "checkpoint_id": "c1"}}
        saver.put_writes(write_cfg, [("messages", "a-write")], task_id="task-1")
        tup = saver.get_tuple(cfg)
        self.assertEqual(len(tup.pending_writes), 1)
        self.assertEqual(tup.pending_writes[0][2], "a-write")


# ── Graph routing ─────────────────────────────────────────────────────────────

class TestGraphRouting(unittest.TestCase):
    """
    Verifies classify_intent's return value selects the correct downstream
    node. We build a real StateGraph with stubbed nodes so the conditional
    edge is exercised against the actual LangGraph runtime.
    """

    def _run(self, needs_rag: bool) -> dict:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import add_messages
        from typing import Annotated, TypedDict

        class S(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            needs_rag: bool
            branch: str

        def classify(_state):
            return {"needs_rag": needs_rag}

        def rag(_state):
            return {"branch": "rag", "messages": [{"role": "assistant", "content": "rag reply"}]}

        def chat(_state):
            return {"branch": "chat", "messages": [{"role": "assistant", "content": "chat reply"}]}

        b = StateGraph(S)
        b.add_node("classify", classify)
        b.add_node("rag", rag)
        b.add_node("chat", chat)
        b.add_edge(START, "classify")
        b.add_conditional_edges(
            "classify",
            lambda s: "rag" if s.get("needs_rag") else "chat",
            {"rag": "rag", "chat": "chat"},
        )
        b.add_edge("rag", END)
        b.add_edge("chat", END)
        graph = b.compile(checkpointer=InMemorySaver())
        return graph.invoke(
            {"messages": [{"role": "user", "content": "hello"}]},
            config={"configurable": {"thread_id": "t-graph"}},
        )

    def test_needs_rag_true_routes_to_rag(self):
        final = self._run(True)
        self.assertEqual(final["branch"], "rag")

    def test_needs_rag_false_routes_to_chat(self):
        final = self._run(False)
        self.assertEqual(final["branch"], "chat")


# ── observability.bootstrap_langsmith ─────────────────────────────────────────

class _FakeConfig:
    def __init__(self, *, is_local=False, region="us-east-1", endpoint="", param=""):
        self.is_local = is_local
        self.aws_region = region
        self.localstack_endpoint = endpoint
        self.langsmith_ssm_parameter = param


class _FakeSSM:
    def __init__(self, *, value: str | None = "secret-key", raise_on_get: bool = False):
        self._value = value
        self._raise = raise_on_get
        self.calls: list[dict] = []

    def get_parameter(self, *, Name, WithDecryption):
        self.calls.append({"Name": Name, "WithDecryption": WithDecryption})
        if self._raise:
            raise RuntimeError("SSM is down")
        if self._value is None:
            return {"Parameter": {}}
        return {"Parameter": {"Name": Name, "Value": self._value}}


class TestBootstrapLangsmith(unittest.TestCase):
    """
    Verifies the cold-start SSM-fetch path. Each test runs with a clean
    _BOOTSTRAPPED flag + cleared LANGSMITH_* env vars so results are
    deterministic regardless of dev-machine state.
    """

    def setUp(self):
        obs._BOOTSTRAPPED = False
        for k in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            os.environ.pop(k, None)

        # Patch make_ssm so the bootstrap never reaches real boto3/LocalStack.
        import clients
        self._ssm = _FakeSSM()
        self._orig_make_ssm = clients.make_ssm
        clients.make_ssm = lambda config: self._ssm

    def tearDown(self):
        import clients
        clients.make_ssm = self._orig_make_ssm
        obs._BOOTSTRAPPED = False
        for k in ("LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
            os.environ.pop(k, None)

    def test_noop_when_no_ssm_parameter_configured(self):
        bootstrap_langsmith(_FakeConfig(param=""))
        self.assertNotIn("LANGSMITH_API_KEY", os.environ)
        self.assertEqual(self._ssm.calls, [])

    def test_fetches_key_and_sets_env_when_parameter_configured(self):
        bootstrap_langsmith(_FakeConfig(param="/haki-ai/langsmith/api-key"))
        self.assertEqual(os.environ["LANGSMITH_API_KEY"], "secret-key")
        self.assertEqual(self._ssm.calls[0]["Name"], "/haki-ai/langsmith/api-key")
        self.assertTrue(self._ssm.calls[0]["WithDecryption"])

    def test_disables_tracing_when_ssm_fetch_fails(self):
        import clients
        clients.make_ssm = lambda config: _FakeSSM(raise_on_get=True)
        bootstrap_langsmith(_FakeConfig(param="/haki-ai/langsmith/api-key"))
        self.assertNotIn("LANGSMITH_API_KEY", os.environ)
        self.assertEqual(os.environ.get("LANGSMITH_TRACING"), "false")

    def test_idempotent_second_call_does_not_refetch(self):
        cfg = _FakeConfig(param="/haki-ai/langsmith/api-key")
        bootstrap_langsmith(cfg)
        bootstrap_langsmith(cfg)
        self.assertEqual(len(self._ssm.calls), 1)

    def test_respects_existing_env_var(self):
        os.environ["LANGSMITH_API_KEY"] = "already-set-locally"
        bootstrap_langsmith(_FakeConfig(param="/haki-ai/langsmith/api-key"))
        self.assertEqual(os.environ["LANGSMITH_API_KEY"], "already-set-locally")
        self.assertEqual(self._ssm.calls, [])


class TestTraceMetadata(unittest.TestCase):
    """Shape of the metadata attached to the haki_turn root span."""

    def test_rag_turn_metadata(self):
        state = {
            "language": "english",
            "needs_rag": True,
            "blocked": False,
            "citations": [{"chunkId": "a"}, {"chunkId": "b"}],
        }
        meta = _trace_metadata(state, session_id="s", env="local", message_length=42)
        self.assertEqual(meta["session_id"], "s")
        self.assertEqual(meta["env"], "local")
        self.assertEqual(meta["language"], "english")
        self.assertTrue(meta["needs_rag"])
        self.assertFalse(meta["blocked"])
        self.assertEqual(meta["citations_count"], 2)
        self.assertEqual(meta["message_length"], 42)

    def test_chat_turn_has_zero_citations(self):
        state = {"language": "swahili", "needs_rag": False, "blocked": False}
        meta = _trace_metadata(state, session_id="s", env="prod", message_length=5)
        self.assertEqual(meta["citations_count"], 0)
        self.assertFalse(meta["needs_rag"])


# ── lambda_handler ────────────────────────────────────────────────────────────

class _FakeCompiledGraph:
    """Returns a canned final state from .invoke() and records the call."""
    def __init__(self, final_state: dict):
        self._final_state = final_state
        self.invocations: list[dict] = []

    def invoke(self, state, config):
        self.invocations.append({"state": state, "config": config})
        return self._final_state


class TestLambdaHandler(unittest.TestCase):
    """
    Exercises the thin handler against a stubbed compiled graph.
    Neither AWS nor LocalStack is needed.
    """

    def setUp(self):
        import handler as h

        self._orig_get_graph = h.get_compiled_graph
        self._orig_make_cw = h.make_cloudwatch
        self._orig_load_history = h.load_history

        self.graph = _FakeCompiledGraph({
            "messages": [],
            "language": "english",
            "needs_rag": True,
            "citations": [{"source": "Employment Act 2007"}],
            "blocked": False,
            "response_text": "Here is your answer.",
            "kb_session_id": "kb-123",
        })
        h.get_compiled_graph = lambda config: self.graph

        class _FakeCloudWatch:
            def put_metric_data(self, **_): pass

        h.make_cloudwatch = lambda config: _FakeCloudWatch()

        self.history_payload: list[dict] = []
        h.load_history = lambda config, session_id: self.history_payload

    def tearDown(self):
        import handler as h
        h.get_compiled_graph = self._orig_get_graph
        h.make_cloudwatch = self._orig_make_cw
        h.load_history = self._orig_load_history

    def _invoke(self, body: dict) -> dict:
        event = {"body": json.dumps(body)}
        result = lambda_handler(event, None)
        result["body"] = json.loads(result["body"])
        return result

    def test_valid_message_returns_200(self):
        result = self._invoke({"message": "What are my rights?", "sessionId": "sess-1"})
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["body"]["response"], "Here is your answer.")

    def test_response_includes_session_id_from_request(self):
        result = self._invoke({"message": "hi", "sessionId": "sess-abc"})
        self.assertEqual(result["body"]["sessionId"], "sess-abc")

    def test_generates_session_id_when_missing(self):
        result = self._invoke({"message": "hi"})
        self.assertTrue(result["body"]["sessionId"])

    def test_graph_receives_thread_id_equal_to_session_id(self):
        self._invoke({"message": "hi", "sessionId": "sess-xyz"})
        self.assertEqual(
            self.graph.invocations[-1]["config"]["configurable"]["thread_id"],
            "sess-xyz",
        )

    def test_empty_message_returns_400(self):
        result = self._invoke({"message": ""})
        self.assertEqual(result["statusCode"], 400)

    def test_whitespace_only_message_returns_400(self):
        result = self._invoke({"message": "   "})
        self.assertEqual(result["statusCode"], 400)

    def test_missing_body_returns_400(self):
        result = lambda_handler({"body": None}, None)
        self.assertEqual(result["statusCode"], 400)

    def test_response_shape_is_complete(self):
        result = self._invoke({"message": "hello", "sessionId": "s"})
        body = result["body"]
        for key in ("response", "citations", "language", "blocked", "sessionId"):
            self.assertIn(key, body)

    def test_cors_header_present(self):
        result = self._invoke({"message": "hi"})
        self.assertEqual(result["headers"]["Access-Control-Allow-Origin"], "*")

    def test_history_get_returns_empty_for_new_session(self):
        result = lambda_handler(
            {
                "requestContext": {"http": {"method": "GET", "path": "/chat/history"}},
                "queryStringParameters": {"sessionId": "unknown"},
            },
            None,
        )
        self.assertEqual(result["statusCode"], 200)
        body = json.loads(result["body"])
        self.assertEqual(body["messages"], [])
        self.assertEqual(body["sessionId"], "unknown")

    def test_history_get_forwards_session_id_and_returns_messages(self):
        self.history_payload = [
            {"id": "m1", "role": "user", "content": "My name is Dave"},
            {"id": "m2", "role": "assistant", "content": "Nice to meet you.", "language": "english"},
        ]
        result = lambda_handler(
            {
                "requestContext": {"http": {"method": "GET", "path": "/chat/history"}},
                "queryStringParameters": {"sessionId": "sess-1"},
            },
            None,
        )
        body = json.loads(result["body"])
        self.assertEqual(len(body["messages"]), 2)
        self.assertEqual(body["messages"][0]["content"], "My name is Dave")

    def test_history_get_400_when_session_id_missing(self):
        result = lambda_handler(
            {
                "requestContext": {"http": {"method": "GET", "path": "/chat/history"}},
                "queryStringParameters": {},
            },
            None,
        )
        self.assertEqual(result["statusCode"], 400)

    def test_blocked_response_passes_through_from_graph(self):
        self.graph._final_state = {
            "messages": [],
            "language": "english",
            "citations": [],
            "blocked": True,
            "response_text": BILINGUAL_REFUSAL,
        }
        result = self._invoke({"message": "what's the weather?"})
        self.assertTrue(result["body"]["blocked"])
        self.assertEqual(result["body"]["response"], BILINGUAL_REFUSAL)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestDetectLanguage,
        TestBuildSystemPrompt,
        TestBuildChatSystemPrompt,
        TestCheckGuardrailBlock,
        TestExtractCitations,
        TestRefreshPresignedUrls,
        TestClassifierParsing,
        TestClassifyIntent,
        TestInvokeChat,
        TestDynamoDBSaver,
        TestGraphRouting,
        TestBootstrapLangsmith,
        TestTraceMetadata,
        TestLambdaHandler,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
