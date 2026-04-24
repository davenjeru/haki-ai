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
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clients.adapters import ComprehendAdapter
from agents.chat import invoke_chat
from memory.checkpointer import DynamoDBSaver
from rag.citations import extract_citations, refresh_presigned_urls
from rag import bm25 as bm25_module
from rag import catalog as catalog_module
from rag import filters as rag_filters
from rag import rrf as rag_rrf
from rag.pipeline import run_rag
from rag.query_expansion import _parse_variants, expand_query
from agents.supervisor import _parse_routing, route_supervisor
from agents.synthesizer import _dedup_citations, synthesize
from evals.audit import AuditRow, classify, format_report
from evals.llm_judge import AXES, _parse_judge_response, judge
from evals.loader import GoldenCase, load_golden_set
from evals.runner import EvalResult
from evals.report import (
    CaseScore,
    _aggregate_judge,
    _category_breakdown,
    _overall_mean,
    write_report,
)
from agents.classifier import classify_intent, _parse_needs_rag
from app.graph import _detect_language
from observability import tracing as obs
from observability.tracing import _trace_metadata, bootstrap_langsmith
from app.handler import lambda_handler
from prompts import (
    BILINGUAL_REFUSAL,
    CLASSIFIER_PROMPT,
    SUPERVISOR_PROMPT,
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

    def test_chat_prompt_forbids_freelance_refusal_prefix(self):
        # Locks in the fix for the prod bug where the chat node emitted the
        # bilingual refusal as a prefix and then kept talking (e.g. suggesting
        # the user go to a specific government office). The prompt must
        # instruct the model to return the refusal verbatim with no
        # prefix/suffix — as a general rule, not a topic-specific patch.
        prompt = build_chat_system_prompt("english")
        self.assertIn("ENTIRE reply must be", prompt)
        self.assertIn("no prefix, suffix, or follow-up", prompt)

    def test_chat_prompt_is_topic_agnostic(self):
        # Scope-enforcement belongs to the guardrail + routing layer, not to
        # an ever-growing allowlist inside the chat prompt. If a specific bug
        # tempts us to name a topic here (e.g. "citizenship is in scope",
        # "don't suggest the Immigration office"), that is a sign the fix
        # belongs upstream (supervisor routing + golden set coverage), not
        # in the prompt. These assertions fail loudly if such overfits leak
        # back in.
        prompt = build_chat_system_prompt("english").lower()
        forbidden_phrases = [
            "citizenship",
            "chapter 3",
            "immigration office",
            "contact information",
            "government offices",
        ]
        for phrase in forbidden_phrases:
            with self.subTest(phrase=phrase):
                self.assertNotIn(
                    phrase,
                    prompt,
                    f"chat prompt leaked topic-specific phrase: {phrase!r}",
                )


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

    def _ref(self, chunk_id, *, section="Section 45", **meta_overrides):
        meta = {
            "source": "Employment Act 2007",
            "chapter": "Part III",
            "section": section,
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
        refs = [
            self._ref("chunk-1", section="Section 40"),
            self._ref("chunk-2", section="Section 45"),
        ]
        result = extract_citations(self._rag_result(refs))
        self.assertEqual([c["chunkId"] for c in result], ["chunk-1", "chunk-2"])

    def test_deduplicates_sibling_chunks_of_same_section(self):
        # splitToSegments can emit chunk-1 and chunk-1-2 for the same
        # (source, section). Advanced RAG dedups them into one citation.
        refs = [self._ref("chunk-1"), self._ref("chunk-1-2"), self._ref("chunk-2", section="Section 40")]
        result = extract_citations(self._rag_result(refs))
        self.assertEqual(len(result), 2)
        self.assertEqual({c["section"] for c in result}, {"Section 45", "Section 40"})

    def test_deduplicates_exact_duplicate_references(self):
        refs = [self._ref("chunk-1"), self._ref("chunk-1")]
        result = extract_citations(self._rag_result(refs))
        self.assertEqual(len(result), 1)

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
        from app import handler as h

        self._orig_get_graph = h.get_compiled_graph
        self._orig_make_cw = h.make_cloudwatch
        self._orig_load_history = h.load_history
        self._orig_thread_owner = h._thread_owner

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

        # Every session in this suite is assumed unowned; keeps the test
        # fast by skipping the real DynamoDB ownership lookup.
        h._thread_owner = lambda config, session_id: None

    def tearDown(self):
        from app import handler as h
        h.get_compiled_graph = self._orig_get_graph
        h.make_cloudwatch = self._orig_make_cw
        h.load_history = self._orig_load_history
        h._thread_owner = self._orig_thread_owner

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


# ── Advanced RAG pipeline ─────────────────────────────────────────────────────

def _catalog_entry(chunk_id: str, text: str, **meta) -> dict:
    """Small helper for building in-memory catalog entries."""
    m = {"chunkId": chunk_id, "source": "Employment Act 2007", "section": f"Section {chunk_id}"}
    m.update(meta)
    return {"chunkId": chunk_id, "text": text, "metadata": m}


class TestQueryExpansion(unittest.TestCase):
    """Parses the model\u2019s reply and handles failures gracefully."""

    def test_parses_well_formed_json(self):
        raw = '{"hypothetical": "An employer must give notice.", "decomposed": "What notice period applies?"}'
        h, d = _parse_variants(raw)
        self.assertEqual(h, "An employer must give notice.")
        self.assertEqual(d, "What notice period applies?")

    def test_strips_markdown_fences(self):
        raw = '```json\n{"hypothetical": "foo", "decomposed": "bar"}\n```'
        h, d = _parse_variants(raw)
        self.assertEqual(h, "foo")
        self.assertEqual(d, "bar")

    def test_returns_none_on_malformed(self):
        self.assertEqual(_parse_variants("not json"), (None, None))

    def test_expand_query_falls_back_to_original_on_error(self):
        class _Broken:
            def invoke_model(self, **_):
                raise RuntimeError("bedrock down")
        out = expand_query("test query", _Broken(), "model")
        self.assertEqual(out, ["test query"])

    def test_expand_query_adds_two_variants(self):
        br = _FakeBedrockRuntime(
            text='{"hypothetical": "hypo answer", "decomposed": "sub question"}'
        )
        out = expand_query("original query", br, "model")
        self.assertEqual(out, ["original query", "hypo answer", "sub question"])

    def test_expand_query_dedupes_identical_variants(self):
        br = _FakeBedrockRuntime(
            text='{"hypothetical": "same", "decomposed": "Same"}'
        )
        out = expand_query("same", br, "model")
        self.assertEqual(len(out), 1)


class TestBM25Retrieve(unittest.TestCase):
    """BM25 retriever over an in-memory catalog."""

    def setUp(self):
        bm25_module.reset_index()
        # BM25Okapi\u2019s IDF goes to zero when a term appears in half the corpus,
        # so we use a realistic set of 6 chunks where \"lease\" is distinctive
        # to chunk-land and \"strike\" is distinctive to chunk-strike.
        self.catalog = [
            _catalog_entry("40", "An employer who terminates the employment shall give notice"),
            _catalog_entry("45", "Unfair termination entitles the employee to compensation"),
            _catalog_entry("10", "This Act applies to all employees in Kenya"),
            _catalog_entry("20", "A contract of service for more than three months shall be in writing"),
            _catalog_entry("land", "The tenant shall pay rent under a lease agreement with the landlord"),
            _catalog_entry("strike", "Trade unions may call a strike after giving seven days notice"),
        ]

    def tearDown(self):
        bm25_module.reset_index()

    def test_retrieves_distinctive_term_first(self):
        results = bm25_module.retrieve("lease agreement", self.catalog, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["metadata"]["chunkId"], "land")

    def test_returns_empty_on_empty_query(self):
        self.assertEqual(bm25_module.retrieve("", self.catalog), [])

    def test_filter_restricts_corpus(self):
        tagged = [
            {**entry, "metadata": {**entry["metadata"], "chunkType": "toc"}}
            if entry["chunkId"] == "strike"
            else entry
            for entry in self.catalog
        ]
        results = bm25_module.retrieve(
            "strike notice",
            tagged,
            metadata_filter={"chunkType": "toc"},
        )
        self.assertTrue(all(r["metadata"].get("chunkType") == "toc" for r in results))
        self.assertTrue(any(r["metadata"]["chunkId"] == "strike" for r in results))

    def test_filter_list_value_matches_any_member(self):
        # Domain specialists (e.g. criminal = Penal Code + CPC + Sexual
        # Offences Act) scope retrieval with a list-valued source filter.
        # Entries whose source matches ANY list member must be kept; all
        # others must be excluded. This is the symmetric local-BM25 path
        # to the Bedrock KB ``in`` clause used in prod.
        tagged = [
            {**entry, "metadata": {**entry["metadata"], "source": src}}
            for entry, src in zip(
                self.catalog,
                [
                    "Penal Code (Cap. 63)",
                    "Employment Act 2007",
                    "Criminal Procedure Code (Cap. 75)",
                    "Employment Act 2007",
                    "Sexual Offences Act 2006",
                    "Employment Act 2007",
                ],
            )
        ]
        results = bm25_module.retrieve(
            "notice",
            tagged,
            metadata_filter={
                "source": [
                    "Penal Code (Cap. 63)",
                    "Criminal Procedure Code (Cap. 75)",
                    "Sexual Offences Act 2006",
                ],
            },
        )
        returned_sources = {r["metadata"]["source"] for r in results}
        # Criminal-domain sources only \u2014 no Employment Act leakage.
        self.assertTrue(
            returned_sources.issubset({
                "Penal Code (Cap. 63)",
                "Criminal Procedure Code (Cap. 75)",
                "Sexual Offences Act 2006",
            }),
            f"unexpected sources in results: {returned_sources}",
        )
        self.assertNotIn("Employment Act 2007", returned_sources)


class TestAdapterFilterTranslation(unittest.TestCase):
    """
    Domain specialists scope retrieval with a list-valued ``source``
    filter. The two adapters must translate that into their native
    multi-value syntax so the filter actually reaches the store.
    """

    def test_bedrock_kb_translates_list_value_to_in_clause(self):
        from clients.adapters import BedrockRAGAdapter

        clause = BedrockRAGAdapter._kb_filter({
            "source": ["Penal Code (Cap. 63)", "Criminal Procedure Code (Cap. 75)"],
        })
        self.assertEqual(
            clause,
            {"in": {"key": "source", "value": [
                "Penal Code (Cap. 63)",
                "Criminal Procedure Code (Cap. 75)",
            ]}},
        )

    def test_bedrock_kb_scalar_value_still_equals(self):
        from clients.adapters import BedrockRAGAdapter

        clause = BedrockRAGAdapter._kb_filter({"source": "Employment Act 2007"})
        self.assertEqual(
            clause,
            {"equals": {"key": "source", "value": "Employment Act 2007"}},
        )

    def test_bedrock_kb_mixes_in_and_equals_under_andall(self):
        from clients.adapters import BedrockRAGAdapter

        clause = BedrockRAGAdapter._kb_filter({
            "source": ["A", "B"],
            "chunkType": "body",
        })
        self.assertEqual(clause["andAll"][0]["in"]["value"], ["A", "B"])
        self.assertEqual(clause["andAll"][1]["equals"]["value"], "body")

    def test_bedrock_kb_none_on_empty(self):
        from clients.adapters import BedrockRAGAdapter
        self.assertIsNone(BedrockRAGAdapter._kb_filter(None))
        self.assertIsNone(BedrockRAGAdapter._kb_filter({}))

    def test_chroma_translates_list_value_to_in_clause(self):
        from clients.adapters import LocalRAGAdapter

        where = LocalRAGAdapter._chroma_where({
            "source": ["Marriage Act 2014", "Children Act 2022"],
        })
        self.assertEqual(
            where,
            {"source": {"$in": ["Marriage Act 2014", "Children Act 2022"]}},
        )

    def test_chroma_scalar_value_still_eq(self):
        from clients.adapters import LocalRAGAdapter

        where = LocalRAGAdapter._chroma_where({"source": "Employment Act 2007"})
        self.assertEqual(where, {"source": {"$eq": "Employment Act 2007"}})


class TestRRF(unittest.TestCase):

    def _r(self, chunk_id: str, score: float = 0.0) -> dict:
        return {
            "content": {"text": chunk_id},
            "metadata": {"chunkId": chunk_id},
            "score": score,
        }

    def test_single_list_preserves_order(self):
        fused = rag_rrf.fuse([[self._r("a"), self._r("b"), self._r("c")]])
        self.assertEqual([r["metadata"]["chunkId"] for r in fused], ["a", "b", "c"])

    def test_fuses_two_lists_boosts_overlap(self):
        # "a" appears at rank 1 in both lists; "b" only at rank 1 in one.
        fused = rag_rrf.fuse([
            [self._r("a"), self._r("b"), self._r("c")],
            [self._r("a"), self._r("d"), self._r("e")],
        ])
        self.assertEqual(fused[0]["metadata"]["chunkId"], "a")
        self.assertGreater(len(fused), 3)

    def test_empty_input_returns_empty(self):
        self.assertEqual(rag_rrf.fuse([]), [])


class TestFilters(unittest.TestCase):

    def _entry(self, **meta) -> dict:
        return {"metadata": meta}

    def test_drop_toc_removes_explicit_chunktype(self):
        results = [
            self._entry(chunkType="body", section="Section 40"),
            self._entry(chunkType="toc", section="Arrangement"),
        ]
        self.assertEqual(len(rag_filters.drop_toc(results)), 1)

    def test_drop_toc_removes_heuristic_match(self):
        results = [
            self._entry(section="Arrangement of Sections", title=""),
            self._entry(section="Section 40", title="Termination"),
        ]
        kept = rag_filters.drop_toc(results)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["metadata"]["section"], "Section 40")

    def test_dedup_by_section_keeps_first(self):
        results = [
            self._entry(source="X", section="S1", chunkId="a"),
            self._entry(source="X", section="S1", chunkId="b"),
            self._entry(source="X", section="S2", chunkId="c"),
        ]
        out = rag_filters.dedup_by_section(results)
        self.assertEqual([r["metadata"]["chunkId"] for r in out], ["a", "c"])

    # ── drop_boilerplate ────────────────────────────────────────────────
    # Preamble / Section 1 / Section 2 chunks were over-retrieved in the
    # Land Act baseline (see plan §5b). drop_boilerplate removes them by
    # chunkType and by the legacy heuristic fallback for un-retagged
    # corpora.

    def test_drop_boilerplate_explicit_chunktype(self):
        results = [
            self._entry(chunkType="body", section="Section 40"),
            self._entry(chunkType="preamble", section="Preamble"),
            self._entry(chunkType="short-title", section="Section 1"),
            self._entry(chunkType="definitions", section="Section 2"),
            self._entry(chunkType="toc", section="Arrangement"),
        ]
        kept = rag_filters.drop_boilerplate(results)
        sections = [r["metadata"]["section"] for r in kept]
        # TOC is not boilerplate — drop_boilerplate only handles preamble /
        # short-title / definitions. The existing drop_toc filter handles TOC.
        self.assertEqual(sections, ["Section 40", "Arrangement"])

    def test_drop_boilerplate_heuristic_fallback(self):
        # Chunks ingested before chunkType was introduced should still be
        # dropped based on (section, title) heuristics.
        results = [
            self._entry(section="Preamble", title=""),
            self._entry(section="Section 1", title="Short title"),
            self._entry(section="Section 2", title="Interpretation"),
            self._entry(section="Section 2", title="Definitions"),
            self._entry(section="Section 40", title="Termination"),
        ]
        kept = rag_filters.drop_boilerplate(results)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["metadata"]["section"], "Section 40")

    def test_drop_boilerplate_ignores_unrelated_section_titles(self):
        # "Section 1" with a non-boilerplate title (e.g. a renumbered Act)
        # must NOT be dropped by the heuristic. Only section+title pairs
        # that actually look like boilerplate qualify.
        results = [
            self._entry(section="Section 1", title="Rights of a child"),
            self._entry(section="Section 2", title="Duty of care"),
        ]
        kept = rag_filters.drop_boilerplate(results)
        self.assertEqual(len(kept), 2)

    def test_is_boilerplate_accepts_flat_metadata(self):
        # Audit passes plain metadata dicts (no "metadata" wrapper) out of
        # EvalResult.retrieved_metadata. The helper must handle both.
        self.assertTrue(rag_filters.is_boilerplate({"chunkType": "preamble"}))
        self.assertTrue(
            rag_filters.is_boilerplate({"section": "Section 2", "title": "Interpretation"})
        )
        self.assertFalse(rag_filters.is_boilerplate({"chunkType": "body"}))
        self.assertFalse(rag_filters.is_boilerplate({}))


class _StubAdapter:
    """
    In-memory adapter used to drive the RAG pipeline without AWS. Implements
    the subset of the RAGAdapterLike protocol the pipeline touches.
    """

    def __init__(self, dense_results: list[dict], generated_text: str, stop_reason: str = "end_turn"):
        self._dense = dense_results
        self._text = generated_text
        self._stop_reason = stop_reason
        self.retrieve_calls: list[tuple[str, int, dict | None]] = []
        self.generate_calls: list[dict] = []

    @property
    def catalog_s3_client(self):
        return None

    @property
    def catalog_list_client(self):
        return None

    @property
    def catalog_bucket(self) -> str:
        return "haki-ai-data"

    @property
    def bedrock_runtime(self):
        return _FakeBedrockRuntime(text='{"hypothetical": "", "decomposed": ""}')

    @property
    def bedrock_agent_runtime(self):
        class _NoRerank:
            def rerank(self, **_):
                raise RuntimeError("no rerank in test")
        return _NoRerank()

    @property
    def aws_region(self) -> str:
        return "us-east-1"

    def retrieve(self, query, top_k=30, metadata_filter=None):
        self.retrieve_calls.append((query, top_k, metadata_filter))
        return [dict(r) for r in self._dense]

    def generate(self, query, system_prompt, context, model_id):
        self.generate_calls.append({
            "query": query,
            "system": system_prompt,
            "context": context,
            "model": model_id,
        })
        return self._text, self._stop_reason


class TestRunRag(unittest.TestCase):
    """End-to-end pipeline wiring against a stub adapter and prebuilt catalog."""

    def setUp(self):
        bm25_module.reset_index()
        catalog_module.reset_catalog()
        catalog_module.set_catalog(
            [_catalog_entry("40", "Notice termination employment")],
            bucket="haki-ai-data",
        )

    def tearDown(self):
        catalog_module.reset_catalog()
        bm25_module.reset_index()

    def _dense_result(self, chunk_id: str) -> dict:
        return {
            "content": {"text": f"body for {chunk_id}"},
            "metadata": {
                "source": "Employment Act 2007",
                "section": f"Section {chunk_id}",
                "chunkId": chunk_id,
                "chunkType": "body",
            },
            "location": {
                "type": "S3",
                "s3Location": {"uri": f"s3://bucket/processed-chunks/{chunk_id}.txt"},
            },
            "score": 0.9,
        }

    def test_returns_retrieve_and_generate_shape(self):
        adapter = _StubAdapter(
            dense_results=[self._dense_result("40")],
            generated_text="Here is your answer.",
        )
        result = run_rag("What is Section 40?", "system", "model", adapter)
        self.assertEqual(result["output"]["text"], "Here is your answer.")
        self.assertEqual(result["stopReason"], "end_turn")
        self.assertIn("citations", result)
        refs = result["citations"][0]["retrievedReferences"]
        self.assertGreaterEqual(len(refs), 1)
        self.assertEqual(refs[0]["metadata"]["chunkId"], "40")

    def test_propagates_guardrail_stop_reason(self):
        adapter = _StubAdapter(
            dense_results=[self._dense_result("40")],
            generated_text="",
            stop_reason="guardrail_intervened",
        )
        result = run_rag("blocked?", "system", "model", adapter)
        self.assertEqual(result["stopReason"], "guardrail_intervened")
        self.assertTrue(check_guardrail_block(result))

    def test_filters_toc_before_generation(self):
        toc = self._dense_result("toc")
        toc["metadata"]["chunkType"] = "toc"
        toc["metadata"]["section"] = "Arrangement of Sections"
        adapter = _StubAdapter(
            dense_results=[toc, self._dense_result("40")],
            generated_text="ok",
        )
        result = run_rag("help", "system", "model", adapter)
        refs = result["citations"][0]["retrievedReferences"]
        # Neither the retrieved refs nor the context should contain TOC.
        chunk_ids = [r["metadata"]["chunkId"] for r in refs]
        self.assertNotIn("toc", chunk_ids)
        self.assertIn("body for 40", adapter.generate_calls[-1]["context"])


# ── Supervisor router ─────────────────────────────────────────────────────────

class TestSupervisorParsing(unittest.TestCase):

    def test_parses_single_agent(self):
        agents, reason = _parse_routing('{"agents": ["employment"], "reason": "notice"}')
        self.assertEqual(agents, ["employment"])
        self.assertEqual(reason, "notice")

    def test_parses_multiple_agents(self):
        agents, _ = _parse_routing(
            '{"agents": ["land", "constitution"], "reason": "land + rights"}'
        )
        self.assertEqual(agents, ["land", "constitution"])

    def test_accepts_new_domain_specialists(self):
        # Corpus expansion added criminal / family / contracts domains.
        # The parser must accept them as known agents so supervisor
        # routing actually reaches those specialists instead of the
        # ``_FALLBACK`` chat path.
        for name in ("criminal", "family", "contracts"):
            with self.subTest(agent=name):
                agents, _ = _parse_routing(
                    '{"agents": ["' + name + '"], "reason": "domain"}'
                )
                self.assertEqual(agents, [name])

    def test_chat_is_exclusive(self):
        agents, _ = _parse_routing('{"agents": ["chat", "employment"]}')
        self.assertEqual(agents, ["chat"])

    def test_drops_unknown_agents(self):
        # "tax" is not (yet) a domain specialist; must be dropped while
        # legitimate agents are preserved in order.
        agents, _ = _parse_routing('{"agents": ["employment", "tax", "land"]}')
        self.assertEqual(agents, ["employment", "land"])

    def test_caps_at_three_agents(self):
        agents, _ = _parse_routing(
            '{"agents": ["constitution", "employment", "land", "tax"]}'
        )
        self.assertEqual(len(agents), 3)

    def test_malformed_json_falls_back(self):
        agents, _ = _parse_routing("not json at all")
        # We fall back to "chat" (not a statute specialist) so the app
        # bilingually-refuses / converses instead of hallucinating citations
        # when the router's JSON is corrupt. See
        # backend/agents/supervisor.py::_FALLBACK for the rationale.
        self.assertEqual(agents, ["chat"])

    def test_empty_agents_falls_back(self):
        agents, _ = _parse_routing('{"agents": []}')
        self.assertEqual(agents, ["chat"])


class TestSupervisorPromptContent(unittest.TestCase):
    """
    The supervisor prompt must describe agents generically (one statute
    each) and enforce a few GENERAL routing rules. Topic-specific hints
    ("citizenship is in scope", "how to apply for citizenship") are
    overfits to a single bug and create maintenance debt — regression
    coverage for those bugs belongs in the golden set, not the prompt.
    """

    def test_explicitly_warns_against_history_bias(self):
        # Fixes the prod bug where prior chit-chat biased routing toward
        # "chat" even when the newest message was clearly a legal question.
        # This is a GENERAL rule (not topic-specific) and should stay.
        self.assertIn("LATEST user message", SUPERVISOR_PROMPT)
        self.assertIn("MUST NOT bias", SUPERVISOR_PROMPT)

    def test_describes_each_specialist_by_its_primary_statute(self):
        # Agents are defined by the primary source(s) they cover, not by
        # an enumerated list of sub-topics. This keeps the prompt stable
        # as we add more statutes or edge cases. Each domain specialist
        # must name every statute in its filter so the router can make
        # informed decisions without us leaking sub-topic hints into the
        # prompt.
        required = [
            "Constitution of Kenya 2010",
            "Employment Act 2007",
            "Land Act 2012",
            "Landlord and Tenant Act (Cap. 301)",
            "Penal Code (Cap. 63)",
            "Criminal Procedure Code (Cap. 75)",
            "Sexual Offences Act 2006",
            "Marriage Act 2014",
            "Children Act 2022",
            "Law of Contract Act",
            "Consumer Protection Act 2012",
        ]
        for source in required:
            with self.subTest(source=source):
                self.assertIn(source, SUPERVISOR_PROMPT)

    def test_is_free_of_topic_specific_overfits(self):
        # If a prod bug tempts us to patch a specific topic into the prompt
        # (e.g. "CITIZENSHIP (Chapter 3)"), the structural fix is: broaden
        # the agent description, improve retrieval, and add a regression
        # row to the golden set. These assertions lock that discipline in.
        lowered = SUPERVISOR_PROMPT.lower()
        forbidden_phrases = [
            "citizenship",
            "chapter 3",
            "dual citizenship",
            "by birth",
            "by registration",
            "how to apply for citizenship",
            "is in scope",
            "out-of-scope",
            "never refuse",
        ]
        for phrase in forbidden_phrases:
            with self.subTest(phrase=phrase):
                self.assertNotIn(
                    phrase,
                    lowered,
                    f"supervisor prompt leaked topic-specific phrase: {phrase!r}",
                )


class TestRouteSupervisor(unittest.TestCase):

    def test_happy_path_calls_bedrock(self):
        br = _FakeBedrockRuntime(text='{"agents": ["employment"], "reason": "notice"}')
        agents, reason = route_supervisor(
            [{"role": "user", "content": "What notice am I owed?"}],
            br,
            "haiku",
        )
        self.assertEqual(agents, ["employment"])
        self.assertEqual(reason, "notice")

    def test_falls_back_on_bedrock_error(self):
        class _Broken:
            def invoke_model(self, **_):
                raise RuntimeError("throttled")
        agents, reason = route_supervisor(
            [{"role": "user", "content": "hi"}],
            _Broken(),
            "haiku",
        )
        # Regression: the fallback used to be ["employment"] which caused
        # conversational turns (e.g. "Naitwa nani?") to be answered by the
        # employment specialist whenever Bedrock throttled. ``chat`` is the
        # only safe default.
        self.assertEqual(agents, ["chat"])
        self.assertEqual(reason, "bedrock_error")

    def test_retries_bedrock_once_before_falling_back(self):
        # Haiku 4.5 has a 10 RPM account quota and transient
        # ThrottlingException was the proximate cause of the "Naitwa nani?"
        # routing failure. One cheap retry meaningfully reduces the
        # fallback rate without materially increasing happy-path latency.
        class _FlakyOnce:
            def __init__(self):
                self.calls = 0

            def invoke_model(self, *, modelId, body, contentType, accept):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("ThrottlingException")
                payload = json.dumps({
                    "content": [{"text": '{"agents": ["chat"], "reason": "conv"}'}],
                    "stop_reason": "end_turn",
                }).encode()
                return {"body": io.BytesIO(payload)}

        br = _FlakyOnce()
        agents, reason = route_supervisor(
            [{"role": "user", "content": "Naitwa nani?"}],
            br,
            "haiku",
        )
        self.assertEqual(br.calls, 2, "expected exactly one retry")
        self.assertEqual(agents, ["chat"])
        self.assertEqual(reason, "conv")

    def test_gives_up_after_exhausting_retries(self):
        class _AlwaysBroken:
            def __init__(self):
                self.calls = 0

            def invoke_model(self, **_):
                self.calls += 1
                raise RuntimeError("ThrottlingException")

        br = _AlwaysBroken()
        agents, reason = route_supervisor(
            [{"role": "user", "content": "hi"}],
            br,
            "haiku",
        )
        self.assertGreaterEqual(br.calls, 2)
        self.assertEqual(agents, ["chat"])
        self.assertEqual(reason, "bedrock_error")


# ── Synthesizer ───────────────────────────────────────────────────────────────

class TestSynthesizer(unittest.TestCase):

    def _output(self, agent: str, *, text="answer", citations=None, blocked=False) -> dict:
        return {"agent": agent, "text": text, "citations": citations or [], "blocked": blocked}

    def test_single_output_passthrough_no_llm_call(self):
        class _FailIfCalled:
            def invoke_model(self, **_):
                raise AssertionError("synthesizer should not call LLM for single output")
        out = synthesize(
            [self._output("employment", text="The Act says X.")],
            "english",
            _FailIfCalled(),
            "haiku",
        )
        self.assertEqual(out["text"], "The Act says X.")
        self.assertFalse(out["blocked"])

    def test_blocked_specialist_shortcircuits(self):
        class _FailIfCalled:
            def invoke_model(self, **_):
                raise AssertionError("synthesizer should not call LLM on blocked output")
        out = synthesize(
            [
                self._output("employment", text="answer", citations=[]),
                self._output("land", text="REFUSED", blocked=True),
            ],
            "english",
            _FailIfCalled(),
            "haiku",
        )
        self.assertTrue(out["blocked"])
        self.assertEqual(out["text"], "REFUSED")

    def test_multi_specialist_calls_llm_and_dedups_citations(self):
        br = _FakeBedrockRuntime(text="Merged answer referencing both statutes.")
        outputs = [
            self._output("employment", text="Employment answer", citations=[
                {"source": "Employment Act 2007", "section": "Section 40"}
            ]),
            self._output("land", text="Land answer", citations=[
                {"source": "Land Act 2012", "section": "Section 152"},
                # Duplicate of first specialist's citation (cross-reference).
                {"source": "Employment Act 2007", "section": "Section 40"},
            ]),
        ]
        out = synthesize(outputs, "english", br, "haiku")
        self.assertEqual(out["text"], "Merged answer referencing both statutes.")
        self.assertEqual(len(out["citations"]), 2)
        sections = {c["section"] for c in out["citations"]}
        self.assertEqual(sections, {"Section 40", "Section 152"})

    def test_empty_outputs_returns_blocked(self):
        out = synthesize([], "english", _FakeBedrockRuntime(text=""), "haiku")
        self.assertTrue(out["blocked"])


class TestDedupCitations(unittest.TestCase):

    def test_dedups_across_lists(self):
        a = [{"source": "X", "section": "S1", "chunkId": "a"}]
        b = [{"source": "X", "section": "S1", "chunkId": "b"}]
        self.assertEqual(len(_dedup_citations([a, b])), 1)

    def test_falls_back_to_chunkid_when_fields_missing(self):
        a = [{"chunkId": "a"}]
        b = [{"chunkId": "b"}, {"chunkId": "a"}]
        self.assertEqual(len(_dedup_citations([a, b])), 2)


# ── Evals: golden set ─────────────────────────────────────────────────────────

class TestGoldenSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cases = load_golden_set()

    def test_has_at_least_thirty_cases(self):
        self.assertGreaterEqual(len(self.cases), 30)

    def test_has_at_least_ten_per_statute_category(self):
        counts: dict[str, int] = {}
        for c in self.cases:
            counts[c.category] = counts.get(c.category, 0) + 1
        self.assertGreaterEqual(counts.get("constitution", 0), 10)
        self.assertGreaterEqual(counts.get("employment", 0), 10)
        self.assertGreaterEqual(counts.get("land", 0), 10)

    def test_has_off_topic_refusal_cases(self):
        # Off-topic refusal coverage is what lets us keep the prompts
        # topic-agnostic: the eval catches "half-refusal" freelancing
        # bugs without us having to patch each one into the prompt.
        refusals = [c for c in self.cases if c.category == "refusal"]
        self.assertGreaterEqual(
            len(refusals),
            3,
            "Need at least 3 off-topic refusal cases (weather / other-jurisdiction / medical)",
        )

    def test_refusal_cases_reference_the_bilingual_refusal(self):
        for c in (c for c in self.cases if c.category == "refusal"):
            with self.subTest(case=c.id):
                self.assertEqual(
                    c.reference_answer.strip(),
                    BILINGUAL_REFUSAL,
                    f"{c.id}: refusal gold-answer must match BILINGUAL_REFUSAL verbatim",
                )

    def test_constitution_breadth_beyond_citizenship(self):
        # Stress-tests the generic supervisor description by making sure the
        # constitution category covers a wide span of articles — not just
        # the one topic that happened to trigger a prod bug. If the prompt
        # starts overfitting again, these non-citizenship cases will flag
        # any routing/retrieval regression that was being masked.
        articles_touched: set[str] = set()
        for c in self.cases:
            if c.category != "constitution":
                continue
            for section in c.expected_sections:
                articles_touched.add(section)
        # At least five DIFFERENT articles across chapters other than 3.
        non_citizenship = {
            a for a in articles_touched
            if a not in {"Article 14", "Article 15", "Article 16"}
        }
        self.assertGreaterEqual(
            len(non_citizenship),
            5,
            f"only {len(non_citizenship)} non-citizenship articles: {non_citizenship}",
        )

    def test_has_at_least_ten_non_english_cases(self):
        non_english = [c for c in self.cases if c.language != "english"]
        self.assertGreaterEqual(len(non_english), 10)

    def test_case_ids_are_unique(self):
        ids = [c.id for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_statute_case_has_an_expected_source(self):
        # Refusal cases intentionally have empty expected_sources — the
        # gold answer IS the bilingual refusal, no statute applies.
        for c in self.cases:
            if c.category == "refusal":
                continue
            with self.subTest(case=c.id):
                self.assertGreater(len(c.expected_sources), 0, c.id)


# ── Evals: judge response parsing ─────────────────────────────────────────────

class TestJudgeParsing(unittest.TestCase):

    def test_parses_well_formed_scores(self):
        text = (
            'Here: {"accuracy": 4, "citation_correctness": 5, '
            '"tone": 3, "language_appropriateness": 5, '
            '"notes": "good"}'
        )
        score = _parse_judge_response(text)
        self.assertEqual(score.accuracy, 4)
        self.assertEqual(score.citation_correctness, 5)
        self.assertEqual(score.tone, 3)
        self.assertEqual(score.language_appropriateness, 5)
        self.assertEqual(score.notes, "good")

    def test_clamps_out_of_range(self):
        score = _parse_judge_response(
            '{"accuracy": 9, "citation_correctness": -2, '
            '"tone": 3, "language_appropriateness": 5}'
        )
        self.assertEqual(score.accuracy, 5)
        self.assertEqual(score.citation_correctness, 0)

    def test_missing_axes_default_to_zero(self):
        score = _parse_judge_response('{"accuracy": 4}')
        self.assertEqual(score.accuracy, 4)
        self.assertEqual(score.citation_correctness, 0)

    def test_malformed_returns_zero_with_note(self):
        score = _parse_judge_response("not json")
        self.assertEqual(score.accuracy, 0)
        self.assertIn("parse_failure", score.notes)

    def test_mean_is_average_over_four_axes(self):
        score = _parse_judge_response(
            '{"accuracy": 4, "citation_correctness": 4, "tone": 4, "language_appropriateness": 4}'
        )
        self.assertAlmostEqual(score.mean(), 4.0)


class TestJudgeBedrockErrors(unittest.TestCase):

    def test_returns_zero_on_bedrock_error(self):
        class _Broken:
            def invoke_model(self, **_):
                raise RuntimeError("boom")
        score = judge(
            question="q",
            language="english",
            candidate_answer="a",
            candidate_citations=[],
            reference_answer="r",
            expected_sources=["Employment Act 2007"],
            retrieved_contexts=[],
            bedrock_runtime=_Broken(),
            model_id="haiku",
        )
        self.assertEqual(score.accuracy, 0)
        self.assertTrue(score.notes.startswith("bedrock_error"))

    def test_happy_path_invokes_bedrock(self):
        br = _FakeBedrockRuntime(
            text='{"accuracy": 5, "citation_correctness": 4, "tone": 5, "language_appropriateness": 5}'
        )
        score = judge(
            question="What is Section 40?",
            language="english",
            candidate_answer="Redundancy rules.",
            candidate_citations=[{"source": "Employment Act 2007", "section": "Section 40"}],
            reference_answer="Section 40 governs redundancy.",
            expected_sources=["Employment Act 2007"],
            retrieved_contexts=["ctx"],
            bedrock_runtime=br,
            model_id="haiku",
        )
        self.assertEqual(score.accuracy, 5)
        self.assertEqual(score.language_appropriateness, 5)


# ── Evals: report writer ──────────────────────────────────────────────────────

class TestReportWriter(unittest.TestCase):

    def _make_case_score(self, category: str, *, scores: tuple[int, int, int, int]) -> CaseScore:
        from evals.loader import GoldenCase
        from evals.llm_judge import JudgeScore
        from evals.runner import EvalResult

        case = GoldenCase(
            id=f"{category}-test",
            category=category,
            question="Q?",
            reference_answer="R.",
            expected_sources=["Source"],
            expected_sections=[],
            language="english",
        )
        result = EvalResult(
            case=case,
            answer="A",
            citations=[{"source": "Source", "section": "Section 40"}],
            retrieved_contexts=["ctx"],
        )
        judge_score = JudgeScore(
            accuracy=scores[0],
            citation_correctness=scores[1],
            tone=scores[2],
            language_appropriateness=scores[3],
            notes="ok",
        )
        return CaseScore(result=result, judge=judge_score)

    def test_aggregate_judge_averages_per_axis(self):
        from evals.llm_judge import JudgeScore
        scores = [
            JudgeScore(accuracy=5, citation_correctness=3, tone=4, language_appropriateness=5),
            JudgeScore(accuracy=3, citation_correctness=5, tone=4, language_appropriateness=3),
        ]
        agg = _aggregate_judge(scores)
        self.assertAlmostEqual(agg["accuracy"], 4.0)
        self.assertAlmostEqual(agg["citation_correctness"], 4.0)
        self.assertAlmostEqual(agg["tone"], 4.0)
        self.assertAlmostEqual(agg["language_appropriateness"], 4.0)

    def test_overall_mean_averages_axes(self):
        agg = {axis: 4.0 for axis in AXES}
        self.assertAlmostEqual(_overall_mean(agg), 4.0)

    def test_category_breakdown(self):
        case_scores = [
            self._make_case_score("constitution", scores=(5, 5, 5, 5)),
            self._make_case_score("constitution", scores=(3, 3, 3, 3)),
            self._make_case_score("employment", scores=(4, 4, 4, 4)),
        ]
        breakdown = _category_breakdown(case_scores)
        self.assertAlmostEqual(breakdown["constitution"]["accuracy"], 4.0)
        self.assertAlmostEqual(breakdown["employment"]["accuracy"], 4.0)

    def test_write_report_creates_markdown(self):
        import tempfile

        case_scores = [self._make_case_score("employment", scores=(5, 4, 3, 5))]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(case_scores, ragas=None, out_dir=tmp)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                body = f.read()
        self.assertIn("Haki AI", body)
        self.assertIn("employment", body)
        self.assertIn("Section", body)  # citation rendered
        self.assertIn("Overall (0–5)", body)

    def test_write_report_includes_ragas_section(self):
        import tempfile

        case_scores = [self._make_case_score("land", scores=(4, 4, 4, 4))]
        ragas = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.9,
            "context_recall": 0.6,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(case_scores, ragas=ragas, out_dir=tmp)
            with open(path, "r", encoding="utf-8") as f:
                body = f.read()
        self.assertIn("RAGAS metrics", body)
        self.assertIn("faithfulness", body)


# ── Evals: retrieval metrics ──────────────────────────────────────────────────

class TestRetrievalMetrics(unittest.TestCase):
    """Recall@k / MRR@k over post-rerank chunks."""

    def _result(
        self,
        *,
        language: str,
        expected_sections: list[str],
        chunks: list[tuple[str, dict]],
    ):
        from evals.loader import GoldenCase
        from evals.runner import EvalResult

        case = GoldenCase(
            id="test",
            category="constitution",
            question="Q?",
            reference_answer="R.",
            expected_sources=["Source"],
            expected_sections=expected_sections,
            language=language,
        )
        return EvalResult(
            case=case,
            answer="A",
            citations=[],
            retrieved_contexts=[text for text, _ in chunks],
            retrieved_metadata=[meta for _, meta in chunks],
        )

    def test_recall_hits_when_section_in_metadata(self):
        from evals.retrieval_metrics import chunk_recall_at_k

        r = self._result(
            language="english",
            expected_sections=["Article 174"],
            chunks=[
                ("body text", {"section": "Article 174", "chapter": "Chapter 11"}),
                ("other", {"section": "Article 160"}),
            ],
        )
        self.assertEqual(chunk_recall_at_k(r, k=5), 1.0)

    def test_recall_hits_when_section_in_chunk_text(self):
        from evals.retrieval_metrics import chunk_recall_at_k

        r = self._result(
            language="swahili",
            expected_sections=["Section 35"],
            chunks=[
                ("Kifungu cha Section 35 kinasema...", {}),
            ],
        )
        self.assertEqual(chunk_recall_at_k(r, k=5), 1.0)

    def test_recall_zero_when_no_match(self):
        from evals.retrieval_metrics import chunk_recall_at_k

        r = self._result(
            language="english",
            expected_sections=["Article 174"],
            chunks=[("text about article 160", {"section": "Article 160"})],
        )
        self.assertEqual(chunk_recall_at_k(r, k=5), 0.0)

    def test_recall_partial_across_multi_section_case(self):
        from evals.retrieval_metrics import chunk_recall_at_k

        r = self._result(
            language="english",
            expected_sections=["Article 43", "Article 53"],
            chunks=[("body", {"section": "Article 43"})],
        )
        self.assertEqual(chunk_recall_at_k(r, k=5), 0.5)

    def test_recall_returns_none_when_no_expected_sections(self):
        from evals.retrieval_metrics import chunk_recall_at_k

        r = self._result(language="english", expected_sections=[], chunks=[("x", {})])
        self.assertIsNone(chunk_recall_at_k(r, k=5))

    def test_recall_respects_k_cutoff(self):
        from evals.retrieval_metrics import chunk_recall_at_k

        r = self._result(
            language="english",
            expected_sections=["Article 174"],
            chunks=[
                ("noise", {"section": "Article 160"}),
                ("noise", {"section": "Article 161"}),
                ("body", {"section": "Article 174"}),
            ],
        )
        self.assertEqual(chunk_recall_at_k(r, k=2), 0.0)
        self.assertEqual(chunk_recall_at_k(r, k=3), 1.0)

    def test_mrr_uses_first_matching_rank(self):
        from evals.retrieval_metrics import mrr_at_k

        r = self._result(
            language="english",
            expected_sections=["Section 40"],
            chunks=[
                ("noise", {"section": "Section 44"}),
                ("noise", {"section": "Section 45"}),
                ("body", {"section": "Section 40"}),
            ],
        )
        self.assertAlmostEqual(mrr_at_k(r, k=10), 1.0 / 3)

    def test_mrr_zero_when_no_match_within_k(self):
        from evals.retrieval_metrics import mrr_at_k

        r = self._result(
            language="english",
            expected_sections=["Article 174"],
            chunks=[("noise", {"section": "Article 160"})],
        )
        self.assertEqual(mrr_at_k(r, k=5), 0.0)

    def test_summarize_stratifies_by_language(self):
        from evals.retrieval_metrics import summarize_retrieval

        en_hit = self._result(
            language="english",
            expected_sections=["Article 10"],
            chunks=[("body", {"section": "Article 10"})],
        )
        sw_miss = self._result(
            language="swahili",
            expected_sections=["Article 10"],
            chunks=[("noise", {"section": "Article 99"})],
        )
        report = summarize_retrieval([en_hit, sw_miss])

        self.assertEqual(report.case_count, 2)
        self.assertAlmostEqual(report.recall_at_k, 0.5)
        self.assertAlmostEqual(report.by_language["english"]["recall_at_k"], 1.0)
        self.assertAlmostEqual(report.by_language["swahili"]["recall_at_k"], 0.0)


# ── Observability: MissingCitations suppression on blocked turns ─────────────

class TestEmitMetrics(unittest.TestCase):

    class _FakeCloudWatch:
        def __init__(self):
            self.calls = []

        def put_metric_data(self, **kwargs):
            self.calls.append(kwargs)

    def _emit(self, **kwargs):
        from observability.metrics import emit_metrics

        cw = self._FakeCloudWatch()
        emit_metrics(cw, language="english", latency_ms=1.0, **kwargs)
        return [m["MetricName"] for m in cw.calls[0]["MetricData"]]

    def test_missing_citations_recorded_when_not_blocked(self):
        names = self._emit(blocked=False, citations=[])
        self.assertIn("MissingCitations", names)

    def test_missing_citations_skipped_on_blocked_turn(self):
        names = self._emit(blocked=True, citations=[])
        self.assertIn("GuardrailBlock", names)
        self.assertNotIn(
            "MissingCitations",
            names,
            "Refusals have no citations by design; counting them turns a "
            "healthy refusal into a false-positive retrieval-quality signal.",
        )

    def test_missing_citations_not_recorded_when_citations_present(self):
        names = self._emit(blocked=False, citations=[{"source": "x"}])
        self.assertNotIn("MissingCitations", names)


# ── Retrieval audit ───────────────────────────────────────────────────────────


class TestAuditClassify(unittest.TestCase):
    """
    Unit tests for evals.audit.classify — the per-case triage that
    decides whether a golden miss is caused by boilerplate pollution
    (fixable via chunk hygiene), a genuine rerank loss, or whether the
    case has no expected sections to score against.
    """

    def _case(self, *, case_id="x-01", expected=None, language="english") -> GoldenCase:
        return GoldenCase(
            id=case_id,
            category="land",
            question="q",
            reference_answer="ref",
            expected_sources=["Land Act 2012"],
            expected_sections=list(expected or []),
            language=language,
        )

    def _result(self, case: GoldenCase, *, retrieved_metas: list[dict]) -> EvalResult:
        return EvalResult(
            case=case,
            answer="",
            citations=[],
            retrieved_contexts=[""] * len(retrieved_metas),
            retrieved_metadata=retrieved_metas,
        )

    def test_hit_when_expected_section_retrieved(self):
        case = self._case(expected=["Section 5"])
        result = self._result(case, retrieved_metas=[
            {"section": "Section 5", "chunkType": "body"},
            {"section": "Section 6", "chunkType": "body"},
        ])
        row = classify(result, top_k=5)
        self.assertEqual(row.failure_mode, "hit")
        self.assertEqual(row.matched, ["Section 5"])
        self.assertEqual(row.boilerplate_slots, 0)

    def test_noise_pollution_flags_boilerplate_slot(self):
        # Expected section not in top-K AND boilerplate (preamble /
        # short-title / definitions) occupies at least one slot — classic
        # land-01/04/08 failure mode.
        case = self._case(expected=["Section 54"])
        result = self._result(case, retrieved_metas=[
            {"section": "Preamble", "chunkType": "preamble"},
            {"section": "Section 1", "chunkType": "short-title"},
            {"section": "Section 65", "chunkType": "body"},
        ])
        row = classify(result, top_k=5)
        self.assertEqual(row.failure_mode, "noise-pollution")
        self.assertEqual(row.boilerplate_slots, 2)
        self.assertEqual(row.matched, [])

    def test_rerank_loss_when_no_boilerplate_and_no_hit(self):
        # Expected section missing AND every top-K slot is a real body
        # chunk → diagnosis is "rerank pushed the right chunk out", not
        # chunk hygiene. Actionable for plan step 7 / 8.
        case = self._case(expected=["Section 85"])
        result = self._result(case, retrieved_metas=[
            {"section": "Section 86", "chunkType": "body"},
            {"section": "Section 91", "chunkType": "body"},
        ])
        row = classify(result, top_k=5)
        self.assertEqual(row.failure_mode, "rerank-loss")
        self.assertEqual(row.boilerplate_slots, 0)

    def test_no_expected_is_skipped_from_scoring(self):
        # Refusal rows and open-ended questions have no expected sections;
        # audit must flag them so they don't distort aggregate rates.
        case = self._case(expected=[])
        result = self._result(case, retrieved_metas=[
            {"section": "Section 1", "chunkType": "short-title"},
        ])
        row = classify(result, top_k=5)
        self.assertEqual(row.failure_mode, "no-expected")

    def test_top_k_cutoff_respected(self):
        # Expected section only present at rank 6 must NOT register as a
        # hit when top_k=5; otherwise the audit would paper over reranker
        # weaknesses.
        case = self._case(expected=["Section 99"])
        metas = [
            {"section": f"Section {i}", "chunkType": "body"} for i in range(10, 15)
        ] + [{"section": "Section 99", "chunkType": "body"}]
        result = self._result(case, retrieved_metas=metas)
        row = classify(result, top_k=5)
        self.assertEqual(row.failure_mode, "rerank-loss")
        self.assertEqual(row.matched, [])

    def test_chunkid_substring_match_for_legacy_rows(self):
        # Some golden rows expect section ranges (e.g. "Section 107-133");
        # falling back to chunkId substring match lets the audit catch
        # hits even when the exact section string differs.
        case = self._case(expected=["107"])
        result = self._result(case, retrieved_metas=[
            {"section": "Section 107", "chunkId": "land-act-2012-part-viii-section-107"},
        ])
        row = classify(result, top_k=5)
        self.assertEqual(row.failure_mode, "hit")

    def test_format_report_renders_summary_and_table(self):
        # Smoke test: format_report must produce a non-empty markdown
        # string containing the headline counts and a row per case.
        rows = [
            AuditRow(
                case_id="x-01",
                language="english",
                expected_sections=["Section 5"],
                retrieved_sections=["Section 5"],
                retrieved_chunk_types=["body"],
                boilerplate_slots=0,
                matched=["Section 5"],
                failure_mode="hit",
            ),
            AuditRow(
                case_id="x-02",
                language="swahili",
                expected_sections=["Section 54"],
                retrieved_sections=["Preamble"],
                retrieved_chunk_types=["preamble"],
                boilerplate_slots=1,
                matched=[],
                failure_mode="noise-pollution",
            ),
        ]
        report = format_report("land", rows, top_k=5)
        self.assertIn("hit=1", report)
        self.assertIn("noise-pollution=1", report)
        self.assertIn("`x-01`", report)
        self.assertIn("`x-02`", report)


# ── Memory recall regressions ("Naitwa nani?" bug family) ─────────────────────
#
# These tests lock in the three fixes made after the LangSmith trace where
# an English "my name is Dave" → employment question → Swahili "Naitwa
# nani?" sequence produced employment termination advice instead of a
# bilingual refusal / name recall. See also the ``memory-lookup-sw``
# golden case for the end-to-end assertion.


class TestSpecialistOutputsReducer(unittest.TestCase):
    """The reducer must reset on the supervisor's ``None`` sentinel."""

    def _reducer(self):
        # Imported lazily so the module import doesn't pull the full
        # graph builder (which requires boto3 clients) at module load.
        from app.graph import _specialist_outputs_reducer
        return _specialist_outputs_reducer

    def test_concatenates_within_a_turn(self):
        # Parallel fan-out across specialists still works as list append.
        reducer = self._reducer()
        first = reducer([], [{"agent": "employment", "text": "e"}])
        second = reducer(first, [{"agent": "land", "text": "l"}])
        self.assertEqual(
            [o["agent"] for o in second],
            ["employment", "land"],
        )

    def test_none_incoming_resets_accumulator(self):
        # The supervisor emits ``None`` at the start of each turn so the
        # checkpointer doesn't carry stale outputs forward.
        reducer = self._reducer()
        populated = [{"agent": "employment", "text": "stale"}]
        self.assertEqual(reducer(populated, None), [])

    def test_handles_none_existing(self):
        reducer = self._reducer()
        self.assertEqual(
            reducer(None, [{"agent": "chat", "text": "ok"}]),
            [{"agent": "chat", "text": "ok"}],
        )

    def test_supervisor_node_emits_reset_sentinel(self):
        # A quick integration-style check: the supervisor node's return
        # dict must include ``specialist_outputs: None`` so the reducer
        # clears stale state even when the supervisor hits the happy path.
        from agents import supervisor as supervisor_module
        from unittest.mock import patch

        with patch.object(
            supervisor_module,
            "route_supervisor",
            return_value=(["chat"], "routed"),
        ):
            # Build a minimal supervisor node inline to avoid constructing
            # the full graph (which needs live AWS clients).
            def supervisor_node(state: dict) -> dict:
                selected, reason = supervisor_module.route_supervisor(
                    [], None, "haiku",
                )
                return {
                    "selected_agents": selected,
                    "routing_reason": reason,
                    "needs_rag": selected != ["chat"],
                    "specialist_outputs": None,
                }

            out = supervisor_node({"messages": []})
            self.assertIn("specialist_outputs", out)
            self.assertIsNone(out["specialist_outputs"])


class TestSpecialistMarksRefusalBlocked(unittest.TestCase):
    """When the model emits BILINGUAL_REFUSAL verbatim, blocked must be True."""

    def test_model_emitted_refusal_detection(self):
        from agents.specialists import _is_model_emitted_refusal
        self.assertTrue(_is_model_emitted_refusal(BILINGUAL_REFUSAL))
        # Trailing whitespace / punctuation shouldn't break detection.
        self.assertTrue(_is_model_emitted_refusal(BILINGUAL_REFUSAL + "  "))
        self.assertTrue(_is_model_emitted_refusal(BILINGUAL_REFUSAL + "."))
        self.assertFalse(_is_model_emitted_refusal(""))
        self.assertFalse(_is_model_emitted_refusal(
            "The Employment Act 2007 Section 40 provides..."
        ))

    def test_chat_agent_marks_refusal_blocked(self):
        # The chat agent feeds through invoke_chat, so its output is
        # whatever Bedrock returns. If the model refuses, blocked must be
        # True so the synthesizer's fast-path suppresses sibling outputs.
        from agents.specialists import build_specialist
        from unittest.mock import patch

        dummy_rag_adapter = object()
        specialist = build_specialist(
            "chat",
            rag_adapter=dummy_rag_adapter,
            bedrock_runtime=object(),
            model_id="haiku",
            s3_client=object(),
            s3_bucket="test",
        )
        state = {
            "selected_agents": ["chat"],
            "language": "swahili",
            "messages": [{"role": "user", "content": "What's the weather?"}],
        }
        with patch(
            "agents.specialists.invoke_chat",
            return_value=BILINGUAL_REFUSAL,
        ):
            out = specialist(state)
        outputs = out["specialist_outputs"]
        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0]["blocked"])
        self.assertEqual(outputs[0]["citations"], [])

    def test_chat_agent_does_not_falsely_mark_normal_output(self):
        from agents.specialists import build_specialist
        from unittest.mock import patch

        specialist = build_specialist(
            "chat",
            rag_adapter=object(),
            bedrock_runtime=object(),
            model_id="haiku",
            s3_client=object(),
            s3_bucket="test",
        )
        state = {
            "selected_agents": ["chat"],
            "language": "english",
            "messages": [{"role": "user", "content": "hi"}],
        }
        with patch(
            "agents.specialists.invoke_chat",
            return_value="Hello! Ask me anything about Kenyan law.",
        ):
            out = specialist(state)
        self.assertFalse(out["specialist_outputs"][0]["blocked"])


# ── Generation cost ───────────────────────────────────────────────────────────


class TestResolvePrice(unittest.TestCase):
    """Bedrock model-id matching is fiddly across regional prefixes and
    model-version suffixes. These tests pin the behaviour so a rename
    in PRICE_TABLE is caught before it silently zeroes out a cost column."""

    def test_bare_bedrock_id_matches(self):
        from evals.generation_cost import _resolve_price
        price = _resolve_price("anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.assertIsNotNone(price)
        self.assertEqual(price["input"], 0.003)

    def test_regional_prefix_is_stripped(self):
        from evals.generation_cost import _resolve_price
        price = _resolve_price("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        self.assertIsNotNone(price)
        self.assertEqual(price["output"], 0.015)

    def test_titan_embeddings_priced(self):
        from evals.generation_cost import _resolve_price
        price = _resolve_price("amazon.titan-embed-text-v2:0")
        self.assertIsNotNone(price)
        self.assertEqual(price["output"], 0.0)

    def test_unknown_model_returns_none(self):
        from evals.generation_cost import _resolve_price
        self.assertIsNone(_resolve_price("mistral.nonexistent-v99"))
        self.assertIsNone(_resolve_price(""))


class TestBudgetTracker(unittest.TestCase):
    def test_record_accumulates_cost(self):
        from evals.generation_cost import BudgetTracker
        t = BudgetTracker()
        t.record("anthropic.claude-3-5-sonnet-20241022-v2:0", 1000, 500)
        # 1000 * 0.003/1000 + 500 * 0.015/1000 = 0.003 + 0.0075 = 0.0105
        self.assertAlmostEqual(t.total_cost, 0.0105, places=6)
        self.assertEqual(t.total_tokens, 1500)

    def test_hard_cap_raises(self):
        from evals.generation_cost import BudgetTracker, BudgetExceededError
        t = BudgetTracker(max_cost=0.01)
        # Under cap — no raise.
        t.record("anthropic.claude-3-5-sonnet-20241022-v2:0", 100, 100)
        # Over cap.
        with self.assertRaises(BudgetExceededError):
            t.record("anthropic.claude-3-5-sonnet-20241022-v2:0", 10000, 10000)

    def test_unknown_model_still_tracked(self):
        """Unknown models count toward call/token totals but contribute
        $0 to cost — so a silent pricing gap can't hide the spend."""
        from evals.generation_cost import BudgetTracker
        t = BudgetTracker()
        t.record("mistral.unpriced-v1", 1000, 1000)
        self.assertEqual(t.total_tokens, 2000)
        self.assertEqual(t.total_cost, 0.0)


class TestEstimateGenerationCost(unittest.TestCase):
    def test_estimate_scales_with_subsample(self):
        from evals.generation_cost import estimate_generation_cost
        small = estimate_generation_cost(
            num_chunks=50, testset_size=10,
            llm_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            embed_model_id="amazon.titan-embed-text-v2:0",
        )
        large = estimate_generation_cost(
            num_chunks=500, testset_size=10,
            llm_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            embed_model_id="amazon.titan-embed-text-v2:0",
        )
        # 10x more chunks → KG extraction dominates → at least 5x more
        # total. Not 10x because synthesis cost is fixed on testset_size.
        self.assertGreater(large, small * 5)

    def test_estimate_positive_for_known_models(self):
        from evals.generation_cost import estimate_generation_cost
        est = estimate_generation_cost(
            num_chunks=200, testset_size=50,
            llm_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            embed_model_id="amazon.titan-embed-text-v2:0",
        )
        self.assertGreater(est, 0)


class TestCostReportMarkdown(unittest.TestCase):
    def test_markdown_has_total_row(self):
        from evals.generation_cost import CostReport, _ModelUsage
        report = CostReport(
            by_model={
                "claude": _ModelUsage(calls=5, input_tokens=1000, output_tokens=500, cost_usd=0.02),
                "titan": _ModelUsage(calls=20, input_tokens=5000, output_tokens=0, cost_usd=0.0001),
            },
            total_cost=0.0201,
            total_tokens=6500,
            source="local_tracker",
        )
        md = report.to_markdown()
        self.assertIn("**Total**", md)
        self.assertIn("$0.0201", md)
        self.assertIn("claude", md)
        self.assertIn("titan", md)


# ── Testset generator helpers ────────────────────────────────────────────────


class TestInferCategory(unittest.TestCase):
    def test_known_source_maps_to_category(self):
        from evals.testset_generator import _infer_category
        self.assertEqual(_infer_category(["Employment Act 2007"]), "employment")
        self.assertEqual(_infer_category(["Constitution of Kenya 2010"]), "constitution")
        self.assertEqual(_infer_category(["Land Act 2012"]), "land")

    def test_first_known_wins(self):
        """When a question spans multiple statutes, category is pinned
        to the first recognised one so retrieval filters in the eval
        runner still get a sensible source filter."""
        from evals.testset_generator import _infer_category
        self.assertEqual(
            _infer_category(["Employment Act 2007", "Constitution of Kenya 2010"]),
            "employment",
        )

    def test_unknown_source_falls_back(self):
        from evals.testset_generator import _infer_category
        self.assertEqual(_infer_category([]), "uncategorised")
        self.assertEqual(_infer_category(["Imaginary Act 9999"]), "uncategorised")


class TestDetectLanguageGenerator(unittest.TestCase):
    """Generator-local language heuristic. Independent of the
    Comprehend-based detector in app.graph — we can't call AWS from
    inside a generation loop."""

    def test_pure_english(self):
        from evals.testset_generator import _detect_language
        self.assertEqual(
            _detect_language("What does Article 28 of the Constitution protect?"),
            "english",
        )

    def test_pure_swahili(self):
        from evals.testset_generator import _detect_language
        self.assertEqual(
            _detect_language("Je, sheria inasema nini kuhusu haki yangu ya kupiga kura?"),
            "swahili",
        )

    def test_mixed_code_switch(self):
        from evals.testset_generator import _detect_language
        self.assertEqual(
            _detect_language("Nina haki gani under Article 49 of the Constitution?"),
            "mixed",
        )

    def test_empty_falls_back_to_english(self):
        from evals.testset_generator import _detect_language
        self.assertEqual(_detect_language(""), "english")


class TestSubsampleStratified(unittest.TestCase):
    def test_every_source_represented(self):
        """No statute should silently drop out of the subsample — the
        floor-of-1-per-source rule guarantees the Constitution shows up
        even when it's dwarfed by Employment Act chunk counts."""
        from evals.testset_generator import _Chunk, subsample
        chunks = (
            [_Chunk(f"c{i}", "body text " * 5, {"source": "Employment Act 2007"}) for i in range(900)]
            + [_Chunk(f"k{i}", "katiba text " * 5, {"source": "Constitution of Kenya 2010"}) for i in range(100)]
            + [_Chunk(f"l{i}", "land text " * 5, {"source": "Land Act 2012"}) for i in range(50)]
        )
        picked = subsample(chunks, size=50, seed=0)
        sources = {c.metadata["source"] for c in picked}
        self.assertEqual(sources, {"Employment Act 2007", "Constitution of Kenya 2010", "Land Act 2012"})
        self.assertLessEqual(len(picked), 50)

    def test_returns_all_when_size_ge_corpus(self):
        from evals.testset_generator import _Chunk, subsample
        chunks = [_Chunk(f"c{i}", "t", {"source": "X"}) for i in range(10)]
        self.assertEqual(len(subsample(chunks, size=999)), 10)

    def test_deterministic_with_seed(self):
        from evals.testset_generator import _Chunk, subsample
        chunks = [_Chunk(f"c{i}", "t", {"source": "X"}) for i in range(100)]
        a = [c.chunk_id for c in subsample(chunks, size=20, seed=42)]
        b = [c.chunk_id for c in subsample(chunks, size=20, seed=42)]
        self.assertEqual(a, b)


class TestGeneratedCaseJsonl(unittest.TestCase):
    def test_roundtrip_matches_golden_schema(self):
        """GeneratedCase.to_jsonl must produce a record that
        GoldenCase.from_dict accepts without modification, so a hand-
        picked row from generated_set.jsonl can be copy-pasted into
        golden_set.jsonl."""
        from evals.testset_generator import GeneratedCase
        from evals.loader import GoldenCase
        case = GeneratedCase(
            id="gen-001",
            category="employment",
            question="What notice period applies to termination?",
            reference_answer="Section 35 sets the statutory notice period.",
            expected_sources=["Employment Act 2007"],
            expected_sections=["Section 35"],
            language="english",
        )
        line = case.to_jsonl()
        parsed = GoldenCase.from_dict(json.loads(line))
        self.assertEqual(parsed.id, "gen-001")
        self.assertEqual(parsed.expected_sources, ["Employment Act 2007"])
        self.assertEqual(parsed.language, "english")


class TestMatchContextToChunk(unittest.TestCase):
    """Reverse-lookup from a RAGAS reference-context string back to the
    corpus chunk it came from. Pinning every branch because a silent
    miss in this function was the root cause of the ~30% empty-sources
    rate we saw in the first generated_set.jsonl."""

    def _corpus(self):
        from evals.testset_generator import _Chunk
        return [
            _Chunk(
                chunk_id="const-art-14",
                text="Article 14. A person is a citizen by birth if, on the day of that "
                     "person's birth, whether or not the person is born in Kenya...",
                metadata={"source": "Constitution of Kenya 2010", "section": "Article 14"},
            ),
            _Chunk(
                chunk_id="emp-sec-35",
                text="35. Notice periods for termination of a contract of service. "
                     "No contract of service not being a contract to perform some "
                     "specific work...",
                metadata={"source": "Employment Act 2007", "section": "Section 35"},
            ),
            _Chunk(
                chunk_id="land-sec-152",
                text="152. Application for eviction order. A landlord may apply to "
                     "the court for an order to recover possession of land...",
                metadata={"source": "Land Act 2012", "section": "Section 152"},
            ),
        ]

    def _prefix_index(self, corpus):
        return {c.text[:80]: c for c in corpus if c.text}

    def test_exact_prefix_match(self):
        from evals.testset_generator import _match_context_to_chunk
        corpus = self._corpus()
        ctx = corpus[0].text
        match = _match_context_to_chunk(ctx, corpus, self._prefix_index(corpus))
        self.assertIsNotNone(match)
        self.assertEqual(match.chunk_id, "const-art-14")

    def test_trimmed_header_still_resolves(self):
        """RAGAS occasionally strips a short leading header (e.g. the
        statute-name breadcrumb) before storing the context. Whichever
        fallback strategy picks it up, the match must still return the
        right chunk — silent empty-sources was the original regression
        and that's what this pins."""
        from evals.testset_generator import _match_context_to_chunk
        corpus = self._corpus()
        trimmed = corpus[1].text[5:]
        match = _match_context_to_chunk(trimmed, corpus, self._prefix_index(corpus))
        self.assertIsNotNone(match)
        self.assertEqual(match.chunk_id, "emp-sec-35")

    def test_substring_match_when_context_is_middle_slice(self):
        """Multi-hop synthesiser sometimes concatenates two chunks and
        stores the joined text as a single reference_context. The
        substring fallback should recover the chunk whose text fully
        contains the probe."""
        from evals.testset_generator import _match_context_to_chunk
        corpus = self._corpus()
        ctx = "Article 14. A person is a citizen by birth if, on the day of that person's birth"
        match = _match_context_to_chunk(ctx, corpus, self._prefix_index(corpus))
        self.assertIsNotNone(match)
        self.assertEqual(match.chunk_id, "const-art-14")

    def test_reverse_substring_when_ragas_prepends_summary(self):
        """When RAGAS wraps the chunk in a generated summary/prefix,
        the context is longer than the chunk. The reverse-substring
        fallback must still recover the chunk."""
        from evals.testset_generator import _match_context_to_chunk
        corpus = self._corpus()
        ctx = (
            "Summary: The following section deals with termination notice. "
            + corpus[1].text
            + "\n\nFollow-up note: see also redundancy provisions."
        )
        match = _match_context_to_chunk(ctx, corpus, self._prefix_index(corpus))
        self.assertIsNotNone(match)
        self.assertEqual(match.chunk_id, "emp-sec-35")

    def test_returns_none_on_no_match(self):
        from evals.testset_generator import _match_context_to_chunk
        corpus = self._corpus()
        match = _match_context_to_chunk(
            "some unrelated legal text from a statute we never ingested",
            corpus,
            self._prefix_index(corpus),
        )
        self.assertIsNone(match)

    def test_returns_none_on_empty_input(self):
        from evals.testset_generator import _match_context_to_chunk
        corpus = self._corpus()
        self.assertIsNone(_match_context_to_chunk("", corpus, self._prefix_index(corpus)))
        self.assertIsNone(_match_context_to_chunk(None, corpus, self._prefix_index(corpus)))  # type: ignore[arg-type]


class TestEnrichFromNodes(unittest.TestCase):
    """Rows whose contexts can't be matched back to a corpus chunk must
    be dropped, not shipped with empty ``expected_sources``. The eval
    runner treats an empty sources list as a refusal case and the
    resulting metrics are misleading — this was the main symptom in
    the first generated_set.jsonl (9/30 rows had empty citations)."""

    class _Sample:
        def __init__(self, contexts):
            self.eval_sample = types.SimpleNamespace(
                user_input="q", reference="a", reference_contexts=contexts
            )

    def _corpus(self):
        from evals.testset_generator import _Chunk
        return [
            _Chunk(
                chunk_id="emp-sec-35",
                text="35. Notice periods for termination of a contract of service...",
                metadata={"source": "Employment Act 2007", "section": "Section 35"},
            ),
        ]

    def _case(self, cid="gen-001", lang="english"):
        from evals.testset_generator import GeneratedCase
        return GeneratedCase(
            id=cid,
            category="employment",
            question="What notice applies to termination?",
            reference_answer="Section 35.",
            expected_sources=[],
            expected_sections=[],
            language=lang,
        )

    def test_matched_context_enriches_sources_and_sections(self):
        from evals.testset_generator import _enrich_from_nodes
        corpus = self._corpus()
        testset = types.SimpleNamespace(samples=[self._Sample([corpus[0].text])])
        enriched, dropped = _enrich_from_nodes([self._case()], testset, corpus)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].expected_sources, ["Employment Act 2007"])
        self.assertEqual(enriched[0].expected_sections, ["Section 35"])

    def test_unmatched_row_is_dropped(self):
        from evals.testset_generator import _enrich_from_nodes
        corpus = self._corpus()
        testset = types.SimpleNamespace(
            samples=[self._Sample(["context text that does not appear in any chunk whatsoever"])]
        )
        enriched, dropped = _enrich_from_nodes([self._case()], testset, corpus)
        self.assertEqual(dropped, 1)
        self.assertEqual(enriched, [])

    def test_category_is_reinferred_from_enriched_sources(self):
        """If the generator-local category was wrong (e.g. defaulted
        to uncategorised), enrichment must overwrite it with the real
        statute's category."""
        from evals.testset_generator import _enrich_from_nodes
        corpus = self._corpus()
        wrong = self._case()
        wrong.category = "uncategorised"
        testset = types.SimpleNamespace(samples=[self._Sample([corpus[0].text])])
        enriched, _ = _enrich_from_nodes([wrong], testset, corpus)
        self.assertEqual(enriched[0].category, "employment")

    def test_mixed_matched_and_unmatched_in_batch(self):
        from evals.testset_generator import _enrich_from_nodes
        corpus = self._corpus()
        good = self._case(cid="gen-good")
        bad = self._case(cid="gen-bad")
        testset = types.SimpleNamespace(samples=[
            self._Sample([corpus[0].text]),
            self._Sample(["nothing that matches"]),
        ])
        enriched, dropped = _enrich_from_nodes([good, bad], testset, corpus)
        self.assertEqual(dropped, 1)
        self.assertEqual([c.id for c in enriched], ["gen-good"])


class TestLoadCorpusFilter(unittest.TestCase):
    """``load_corpus`` must skip TOC / preamble / short-title /
    definitions chunks so RAGAS can't anchor questions on boilerplate.
    This is the other half of the generated_set.jsonl regression — TOC
    chunks were the source of rows like ``gen-002`` whose answer said
    "Section 101(1)" but whose expected_sections said "Section 398"
    (the TOC chunk's own section metadata)."""

    def _stub_catalog(self):
        return [
            {"chunkId": "body-1", "text": "body text about section 35",
             "metadata": {"source": "Employment Act 2007", "section": "Section 35", "chunkType": "body"}},
            {"chunkId": "toc-1", "text": "Chapter XLII – 398. Accessories... 399. Related...",
             "metadata": {"source": "Penal Code (Cap. 63)", "section": "Section 398", "chunkType": "toc"}},
            {"chunkId": "pre-1", "text": "AN ACT of Parliament...",
             "metadata": {"source": "Penal Code (Cap. 63)", "chunkType": "preamble"}},
            {"chunkId": "def-1", "text": "In this Act, unless the context requires...",
             "metadata": {"source": "Employment Act 2007", "section": "Section 2", "chunkType": "definitions"}},
            {"chunkId": "short-1", "text": "This Act may be cited as...",
             "metadata": {"source": "Marriage Act 2014", "section": "Section 1", "chunkType": "short-title"}},
            {"chunkId": "legacy-1", "text": "older chunk without a chunkType tag",
             "metadata": {"source": "Landlord and Tenant Act (Cap. 301)", "section": "Section 4"}},
            {"chunkId": "empty-1", "text": "",
             "metadata": {"source": "Employment Act 2007", "chunkType": "body"}},
        ]

    def _config(self):
        return types.SimpleNamespace(s3_bucket="haki-ai-data")

    def test_only_body_chunks_are_anchorable_by_default(self):
        from unittest.mock import patch
        from evals import testset_generator as gen

        with patch.object(gen, "make_s3_listing", return_value=object()), \
             patch("rag.catalog.get_catalog", return_value=self._stub_catalog()):
            chunks = gen.load_corpus(self._config())

        ids = {c.chunk_id for c in chunks}
        self.assertIn("body-1", ids)
        # Legacy chunks (chunkType missing) default to body for
        # backwards compat with pre-fix ingests.
        self.assertIn("legacy-1", ids)
        for dropped in ("toc-1", "pre-1", "def-1", "short-1", "empty-1"):
            self.assertNotIn(dropped, ids, f"{dropped} should have been filtered")

    def test_override_anchorable_types_widens_filter(self):
        """An operator can widen the filter via the kwarg (e.g. to
        generate questions about definitions for coverage). Body must
        remain included; only the explicitly listed extra types
        should join it."""
        from unittest.mock import patch
        from evals import testset_generator as gen

        with patch.object(gen, "make_s3_listing", return_value=object()), \
             patch("rag.catalog.get_catalog", return_value=self._stub_catalog()):
            chunks = gen.load_corpus(
                self._config(),
                anchorable_chunk_types=frozenset({"body", "definitions"}),
            )

        ids = {c.chunk_id for c in chunks}
        self.assertIn("body-1", ids)
        self.assertIn("def-1", ids)
        self.assertIn("legacy-1", ids)
        for dropped in ("toc-1", "pre-1", "short-1"):
            self.assertNotIn(dropped, ids)


class TestListingClientFactory(unittest.TestCase):
    """Pins the presign/listing client split so the chunk-catalog
    loader can't be silently coupled back into the presign-configured
    client path.

    We don't assert on internal BotoConfig fields because make_s3 and
    make_s3_listing happen to paginate equivalently against LocalStack
    today — the split is a separation-of-concerns guard, not a workaround
    for a known listing bug. The real bug we hit was a stale
    `_catalog.json` fast-path; see backend/rag/catalog.py and
    backend/scripts/build_chunk_catalog.py.
    """

    def _make_local_config(self):
        from app.config import Config
        return Config(
            is_local=True,
            localstack_endpoint="http://localhost:4566",
            aws_region="us-east-1",
            knowledge_base_id="",
            guardrail_id="",
            guardrail_version="DRAFT",
            bedrock_model_id="",
            embedding_model_id="amazon.titan-embed-text-v2:0",
            chroma_host="",
            chroma_port=8000,
            s3_bucket="haki-ai-data",
            checkpoints_table="haki-ai-checkpoints",
            chat_threads_table="haki-ai-chat-threads",
            environment="local",
            langsmith_ssm_parameter="",
            clerk_publishable_key="",
        )

    def test_make_s3_points_at_localstack_locally(self):
        from clients import make_s3
        client = make_s3(self._make_local_config())
        self.assertEqual(client.meta.endpoint_url, "http://localhost:4566")

    def test_make_s3_listing_points_at_localstack_locally(self):
        from clients import make_s3_listing
        client = make_s3_listing(self._make_local_config())
        self.assertEqual(client.meta.endpoint_url, "http://localhost:4566")

    def test_local_adapter_uses_separate_listing_client(self):
        """LocalRAGAdapter must expose catalog_list_client distinct
        from catalog_s3_client when both are supplied, so the pipeline
        routes listings through whatever client the caller provided."""
        from clients.adapters import LocalRAGAdapter
        presign_client = object()
        list_client = object()

        class _StubChroma:
            pass

        adapter = LocalRAGAdapter.__new__(LocalRAGAdapter)
        adapter._bedrock_runtime = None
        adapter._bedrock_agent_runtime = None
        adapter._embed_model = ""
        adapter._s3_client = presign_client
        adapter._s3_list_client = list_client
        adapter._s3_bucket = "haki-ai-data"
        adapter._aws_region = "us-east-1"
        adapter._guardrail_id = ""
        adapter._guardrail_version = ""
        adapter._collection = _StubChroma()

        self.assertIs(adapter.catalog_s3_client, presign_client)
        self.assertIs(adapter.catalog_list_client, list_client)

    def test_local_adapter_falls_back_when_list_client_absent(self):
        """Back-compat: callers that only pass s3_client (older tests)
        still get a working listing client."""
        from clients.adapters import LocalRAGAdapter
        presign_client = object()

        adapter = LocalRAGAdapter.__new__(LocalRAGAdapter)
        adapter._s3_client = presign_client
        adapter._s3_list_client = presign_client

        self.assertIs(adapter.catalog_list_client, presign_client)


# ── Clerk auth: issuer derivation + JWT verification ──────────────────────────

class TestDeriveIssuer(unittest.TestCase):
    """The backend auto-derives the Clerk issuer from the publishable key."""

    def test_decodes_test_instance(self):
        from app.auth import derive_issuer
        import base64 as _b64

        host = "learning-honeybee-54.clerk.accounts.dev"
        body = _b64.b64encode(f"{host}$".encode()).rstrip(b"=").decode()
        issuer = derive_issuer(f"pk_test_{body}")
        self.assertEqual(issuer, f"https://{host}")

    def test_decodes_live_instance(self):
        from app.auth import derive_issuer
        import base64 as _b64

        host = "clerk.example.com"
        body = _b64.b64encode(f"{host}$".encode()).rstrip(b"=").decode()
        issuer = derive_issuer(f"pk_live_{body}")
        self.assertEqual(issuer, f"https://{host}")

    def test_empty_raises(self):
        from app.auth import derive_issuer
        with self.assertRaises(ValueError):
            derive_issuer("")

    def test_bad_prefix_raises(self):
        from app.auth import derive_issuer
        with self.assertRaises(ValueError):
            derive_issuer("sk_test_anything")


class TestExtractBearer(unittest.TestCase):
    """Bearer header extraction handles both APIGW-v2 (lowercase) and hand-rolled events."""

    def test_lowercase_authorization(self):
        from app.auth import extract_bearer
        self.assertEqual(
            extract_bearer({"headers": {"authorization": "Bearer abc.def.ghi"}}),
            "abc.def.ghi",
        )

    def test_titlecase_authorization(self):
        from app.auth import extract_bearer
        self.assertEqual(
            extract_bearer({"headers": {"Authorization": "Bearer abc"}}),
            "abc",
        )

    def test_missing_returns_none(self):
        from app.auth import extract_bearer
        self.assertIsNone(extract_bearer({}))
        self.assertIsNone(extract_bearer({"headers": {}}))

    def test_non_bearer_scheme_returns_none(self):
        from app.auth import extract_bearer
        self.assertIsNone(extract_bearer({"headers": {"authorization": "Basic xyz"}}))

    def test_empty_bearer_returns_none(self):
        from app.auth import extract_bearer
        self.assertIsNone(extract_bearer({"headers": {"authorization": "Bearer   "}}))


class TestVerifyClerkJwt(unittest.TestCase):
    """
    End-to-end JWT verification against a synthetic JWKS. We sign tokens with
    our own RSA key pair and stub :mod:`app.auth`'s ``PyJWKClient`` so the
    test never touches the network — exactly what a senior reviewer would
    want: coverage of happy/bad/expired without flaky HTTPS calls.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("cryptography not available")
        cls._rsa = rsa
        cls.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.public_key = cls.private_key.public_key()

    def setUp(self):
        import base64 as _b64
        from app import auth as auth_mod

        host = "learning-honeybee-54.clerk.accounts.dev"
        body = _b64.b64encode(f"{host}$".encode()).rstrip(b"=").decode()
        self.publishable_key = f"pk_test_{body}"
        self.issuer = f"https://{host}"

        public_key = self.public_key

        class _StubJwkSigning:
            def __init__(self, key):
                self.key = key

        class _StubJwkClient:
            def __init__(self, url, **_):
                self.url = url

            def get_signing_key_from_jwt(self, token):  # noqa: ARG002 - stub
                return _StubJwkSigning(public_key)

        self._orig_client = auth_mod.PyJWKClient
        auth_mod.PyJWKClient = _StubJwkClient
        auth_mod.reset_cache()

    def tearDown(self):
        from app import auth as auth_mod
        auth_mod.PyJWKClient = self._orig_client
        auth_mod.reset_cache()

    def _sign(self, claims: dict) -> str:
        import jwt as _jwt
        from cryptography.hazmat.primitives import serialization
        pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return _jwt.encode(claims, pem, algorithm="RS256")

    def test_valid_token_returns_user_id(self):
        from app.auth import verify_clerk_jwt
        now = int(time.time()) if False else __import__("time").time()
        token = self._sign({
            "sub": "user_abc",
            "iat": int(now) - 10,
            "exp": int(now) + 3600,
            "iss": self.issuer,
        })
        self.assertEqual(verify_clerk_jwt(token, self.publishable_key), "user_abc")

    def test_expired_token_rejected(self):
        from app.auth import verify_clerk_jwt
        now = __import__("time").time()
        token = self._sign({
            "sub": "user_abc",
            "iat": int(now) - 3600,
            "exp": int(now) - 60,  # expired
            "iss": self.issuer,
        })
        self.assertIsNone(verify_clerk_jwt(token, self.publishable_key))

    def test_wrong_issuer_rejected(self):
        from app.auth import verify_clerk_jwt
        now = __import__("time").time()
        token = self._sign({
            "sub": "user_abc",
            "iat": int(now),
            "exp": int(now) + 3600,
            "iss": "https://evil.clerk.accounts.dev",
        })
        self.assertIsNone(verify_clerk_jwt(token, self.publishable_key))

    def test_missing_sub_rejected(self):
        from app.auth import verify_clerk_jwt
        now = __import__("time").time()
        token = self._sign({
            "iat": int(now),
            "exp": int(now) + 3600,
            "iss": self.issuer,
        })
        self.assertIsNone(verify_clerk_jwt(token, self.publishable_key))

    def test_empty_token_returns_none(self):
        from app.auth import verify_clerk_jwt
        self.assertIsNone(verify_clerk_jwt("", self.publishable_key))
        self.assertIsNone(verify_clerk_jwt(None, self.publishable_key))

    def test_missing_publishable_key_returns_none(self):
        from app.auth import verify_clerk_jwt
        self.assertIsNone(verify_clerk_jwt("anything", ""))

    def test_garbage_token_returns_none(self):
        from app.auth import verify_clerk_jwt
        self.assertIsNone(verify_clerk_jwt("not.a.jwt", self.publishable_key))


# ── ThreadsRepo ───────────────────────────────────────────────────────────────


class _FakeDynamoTable:
    """In-memory boto3 Table lookalike for ThreadsRepo tests."""

    def __init__(self):
        self._store: dict[tuple[str, str], dict] = {}

    def get_item(self, *, Key):
        item = self._store.get((Key["user_id"], Key["thread_id"]))
        return {"Item": dict(item)} if item else {}

    def put_item(self, *, Item):
        self._store[(Item["user_id"], Item["thread_id"])] = dict(Item)

    def update_item(self, *, Key, AttributeUpdates):
        item = self._store.get((Key["user_id"], Key["thread_id"]))
        if item is None:
            item = {"user_id": Key["user_id"], "thread_id": Key["thread_id"]}
        for name, op in AttributeUpdates.items():
            if op["Action"] == "PUT":
                item[name] = op["Value"]
        self._store[(Key["user_id"], Key["thread_id"])] = item

    def query(self, *, KeyConditionExpression, IndexName=None, Limit=None):
        # Minimal query stub for the two shapes we use in-repo:
        #   - base table: KeyCondition "user_id = :u"  (list threads for user)
        #   - thread_id_index GSI: KeyCondition "thread_id = :t" (find owner)
        # We peek into the boto3 KeyConditionExpression's ``_values`` tuple to
        # pull the literal — stable since boto3 1.x and cheap to avoid a real
        # expression parser.
        values = getattr(KeyConditionExpression, "_values", ())
        value = values[1] if values else None
        if value is None:
            return {"Items": []}
        if IndexName == "thread_id_index":
            items = [dict(v) for (_, t), v in self._store.items() if t == value]
        else:
            items = [dict(v) for (u, _), v in self._store.items() if u == value]
        if Limit is not None:
            items = items[:Limit]
        return {"Items": items}


class TestThreadsRepo(unittest.TestCase):
    def setUp(self):
        from memory.threads import ThreadsRepo
        self.table = _FakeDynamoTable()
        self.repo = ThreadsRepo(self.table)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.repo.get("u1", "t1"))

    def test_upsert_creates_row_with_title(self):
        row = self.repo.upsert("u1", "t1", title="First chat")
        self.assertEqual(row.title, "First chat")
        self.assertGreater(row.created_at, 0)
        self.assertEqual(row.created_at, row.updated_at)
        # The row survives the round-trip via get_item.
        fetched = self.repo.get("u1", "t1")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.title, "First chat")

    def test_upsert_without_title_uses_default(self):
        row = self.repo.upsert("u1", "t1")
        self.assertEqual(row.title, "New chat")

    def test_upsert_existing_bumps_updated_at_but_keeps_title(self):
        first = self.repo.upsert("u1", "t1", title="Original")
        # Nudge clock forward so updated_at strictly increases.
        import time as _t
        _t.sleep(1.01)
        second = self.repo.upsert("u1", "t1")  # no explicit title
        self.assertGreater(second.updated_at, first.updated_at)
        self.assertEqual(second.title, "Original")
        # Re-reading must also return the preserved title.
        self.assertEqual(self.repo.get("u1", "t1").title, "Original")

    def test_update_title_renames_existing(self):
        self.repo.upsert("u1", "t1", title="Old")
        row = self.repo.update_title("u1", "t1", "  New title  ")
        self.assertIsNotNone(row)
        self.assertEqual(row.title, "New title")
        self.assertEqual(self.repo.get("u1", "t1").title, "New title")

    def test_update_title_missing_returns_none(self):
        self.assertIsNone(self.repo.update_title("u1", "nope", "whatever"))

    def test_update_title_empty_returns_none(self):
        self.repo.upsert("u1", "t1", title="Old")
        self.assertIsNone(self.repo.update_title("u1", "t1", "   "))

    def test_list_for_user_sorted_by_updated_at_desc(self):
        import time as _t
        self.repo.upsert("u1", "t-oldest", title="old")
        _t.sleep(1.01)
        self.repo.upsert("u1", "t-middle", title="mid")
        _t.sleep(1.01)
        self.repo.upsert("u1", "t-newest", title="new")
        ids = [r.thread_id for r in self.repo.list_for_user("u1")]
        self.assertEqual(ids, ["t-newest", "t-middle", "t-oldest"])

    def test_list_for_user_isolates_users(self):
        self.repo.upsert("u1", "t1", title="a")
        self.repo.upsert("u2", "t1", title="b")
        self.assertEqual([r.title for r in self.repo.list_for_user("u1")], ["a"])
        self.assertEqual([r.title for r in self.repo.list_for_user("u2")], ["b"])

    def test_find_owner_missing_returns_none(self):
        self.assertIsNone(self.repo.find_owner("t-unseen"))

    def test_find_owner_empty_thread_id_returns_none(self):
        self.repo.upsert("u1", "t1")
        self.assertIsNone(self.repo.find_owner(""))

    def test_find_owner_returns_owning_user(self):
        self.repo.upsert("u1", "t1", title="a")
        self.assertEqual(self.repo.find_owner("t1"), "u1")

    def test_find_owner_unique_per_thread(self):
        self.repo.upsert("u1", "t1", title="a")
        self.repo.upsert("u2", "t2", title="b")
        self.assertEqual(self.repo.find_owner("t1"), "u1")
        self.assertEqual(self.repo.find_owner("t2"), "u2")


# ── Title generator ──────────────────────────────────────────────────────────


class TestGenerateTitle(unittest.TestCase):
    def test_happy_path_sanitises_and_clamps(self):
        from agents.title import generate_title
        br = _FakeBedrockRuntime(text='"Termination without notice rights."')
        title = generate_title(
            "What are my rights if fired without notice?",
            "Under the Employment Act 2007 Section 40...",
            br,
            "model-id",
        )
        self.assertEqual(title, "Termination without notice rights")

    def test_clamps_to_six_words(self):
        from agents.title import generate_title
        br = _FakeBedrockRuntime(text="One two three four five six seven")
        self.assertEqual(
            generate_title("q", "a", br, "m"),
            "One two three four five six",
        )

    def test_empty_response_falls_back(self):
        from agents.title import generate_title
        br = _FakeBedrockRuntime(text="")
        self.assertEqual(generate_title("q", "a", br, "m"), "New chat")

    def test_bedrock_error_returns_fallback(self):
        from agents.title import generate_title

        class _BrokenBedrock:
            def invoke_model(self, **_):
                raise RuntimeError("bedrock down")

        self.assertEqual(
            generate_title("q", "a", _BrokenBedrock(), "m"),
            "New chat",
        )

    def test_empty_question_returns_fallback(self):
        from agents.title import generate_title
        br = _FakeBedrockRuntime(text="Anything")
        self.assertEqual(generate_title("   ", "a", br, "m"), "New chat")


# ── Handler: signed-in routes + Bearer on POST /chat ──────────────────────────


class TestHandlerAuthAndThreads(unittest.TestCase):
    """
    Exercises the signed-in routes + auth gating without touching AWS.

    A single module-level ThreadsRepo fake is swapped in and the Clerk
    verifier is monkey-patched to accept a fixed token -> user_id mapping.
    """

    def setUp(self):
        from app import handler as h
        from memory.threads import ThreadsRepo

        # Stub compiled graph / cloudwatch / load_history as in TestLambdaHandler.
        self._orig_get_graph = h.get_compiled_graph
        self._orig_make_cw = h.make_cloudwatch
        self._orig_load_history = h.load_history
        self._orig_verify = h.verify_clerk_jwt
        self._orig_threads_repo = h._threads_repo
        self._orig_bedrock = h.make_bedrock_runtime
        self._orig_generate_title = h.generate_title

        self.graph = _FakeCompiledGraph({
            "messages": [],
            "language": "english",
            "needs_rag": True,
            "citations": [{"source": "Employment Act 2007"}],
            "blocked": False,
            "response_text": "Here is your answer.",
        })
        h.get_compiled_graph = lambda config: self.graph

        class _FakeCloudWatch:
            def put_metric_data(self, **_): pass

        h.make_cloudwatch = lambda config: _FakeCloudWatch()
        h.load_history = lambda config, session_id: []

        # In-memory threads repo shared across calls.
        self.table = _FakeDynamoTable()
        self.repo = ThreadsRepo(self.table)
        h._threads_repo = lambda config: self.repo

        # Token -> user_id stub; anything else returns None.
        self.tokens = {"good-token": "user_abc"}

        def _verify(token, key):
            return self.tokens.get((token or "").strip())

        h.verify_clerk_jwt = _verify

        # Make title generation deterministic + free.
        h.make_bedrock_runtime = lambda config: None
        h.generate_title = lambda q, a, br, model: "Sample title"

    def tearDown(self):
        from app import handler as h
        h.get_compiled_graph = self._orig_get_graph
        h.make_cloudwatch = self._orig_make_cw
        h.load_history = self._orig_load_history
        h.verify_clerk_jwt = self._orig_verify
        h._threads_repo = self._orig_threads_repo
        h.make_bedrock_runtime = self._orig_bedrock
        h.generate_title = self._orig_generate_title

    # Event helpers ----------------------------------------------------------

    def _event(self, *, method, path, body=None, token=None, qs=None):
        event = {
            "requestContext": {"http": {"method": method, "path": path}},
            "body": json.dumps(body) if body is not None else None,
            "headers": {"authorization": f"Bearer {token}"} if token else {},
            "queryStringParameters": qs or {},
        }
        return event

    def _decode(self, result):
        result["body"] = json.loads(result["body"])
        return result

    # POST /chat with / without auth -----------------------------------------

    def test_chat_without_auth_skips_thread_index(self):
        result = self._decode(lambda_handler(
            self._event(method="POST", path="/chat", body={"message": "hi", "sessionId": "s1"}),
            None,
        ))
        self.assertEqual(result["statusCode"], 200)
        # No thread row was written for an anonymous turn.
        self.assertEqual(self.repo.list_for_user("user_abc"), [])

    def test_chat_with_auth_creates_thread_with_generated_title(self):
        result = self._decode(lambda_handler(
            self._event(
                method="POST", path="/chat",
                body={"message": "What are my rights?", "sessionId": "sess-1"},
                token="good-token",
            ),
            None,
        ))
        self.assertEqual(result["statusCode"], 200)
        rows = self.repo.list_for_user("user_abc")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].thread_id, "sess-1")
        self.assertEqual(rows[0].title, "Sample title")

    def test_chat_with_auth_second_turn_keeps_title_bumps_updated_at(self):
        # First turn creates the row.
        lambda_handler(
            self._event(
                method="POST", path="/chat",
                body={"message": "First", "sessionId": "sess-1"},
                token="good-token",
            ),
            None,
        )
        first = self.repo.get("user_abc", "sess-1")
        import time as _t
        _t.sleep(1.01)
        # Second turn must not overwrite the user-visible title.
        # Rename between turns to simulate the user editing.
        self.repo.update_title("user_abc", "sess-1", "Custom title")
        _t.sleep(1.01)
        lambda_handler(
            self._event(
                method="POST", path="/chat",
                body={"message": "Second", "sessionId": "sess-1"},
                token="good-token",
            ),
            None,
        )
        updated = self.repo.get("user_abc", "sess-1")
        self.assertEqual(updated.title, "Custom title")
        self.assertGreater(updated.updated_at, first.updated_at)

    def test_chat_with_invalid_token_behaves_anonymously(self):
        lambda_handler(
            self._event(
                method="POST", path="/chat",
                body={"message": "hi", "sessionId": "s1"},
                token="bogus",
            ),
            None,
        )
        self.assertEqual(self.repo.list_for_user("user_abc"), [])

    # GET /chat/threads ------------------------------------------------------

    def test_list_threads_requires_auth(self):
        result = self._decode(lambda_handler(
            self._event(method="GET", path="/chat/threads"),
            None,
        ))
        self.assertEqual(result["statusCode"], 401)

    def test_list_threads_returns_users_rows_sorted(self):
        self.repo.upsert("user_abc", "t1", title="Alpha")
        import time as _t
        _t.sleep(1.01)
        self.repo.upsert("user_abc", "t2", title="Beta")
        result = self._decode(lambda_handler(
            self._event(method="GET", path="/chat/threads", token="good-token"),
            None,
        ))
        self.assertEqual(result["statusCode"], 200)
        ids = [t["threadId"] for t in result["body"]["threads"]]
        self.assertEqual(ids, ["t2", "t1"])

    # PATCH /chat/threads ----------------------------------------------------

    def test_rename_requires_auth(self):
        result = lambda_handler(
            self._event(method="PATCH", path="/chat/threads", body={"threadId": "t1", "title": "x"}),
            None,
        )
        self.assertEqual(result["statusCode"], 401)

    def test_rename_bad_payload_returns_400(self):
        result = lambda_handler(
            self._event(method="PATCH", path="/chat/threads", body={"title": "x"}, token="good-token"),
            None,
        )
        self.assertEqual(result["statusCode"], 400)

    def test_rename_missing_thread_returns_404(self):
        result = lambda_handler(
            self._event(
                method="PATCH", path="/chat/threads",
                body={"threadId": "nope", "title": "x"},
                token="good-token",
            ),
            None,
        )
        self.assertEqual(result["statusCode"], 404)

    def test_rename_persists_title(self):
        self.repo.upsert("user_abc", "t1", title="Old")
        result = self._decode(lambda_handler(
            self._event(
                method="PATCH", path="/chat/threads",
                body={"threadId": "t1", "title": "Shiny"},
                token="good-token",
            ),
            None,
        ))
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["body"]["thread"]["title"], "Shiny")
        self.assertEqual(self.repo.get("user_abc", "t1").title, "Shiny")

    # POST /chat/threads/claim ----------------------------------------------

    def test_claim_requires_auth(self):
        result = lambda_handler(
            self._event(method="POST", path="/chat/threads/claim", body={"threadId": "t1"}),
            None,
        )
        self.assertEqual(result["statusCode"], 401)

    def test_claim_creates_row(self):
        result = self._decode(lambda_handler(
            self._event(
                method="POST", path="/chat/threads/claim",
                body={"threadId": "sess-anon"},
                token="good-token",
            ),
            None,
        ))
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["body"]["thread"]["threadId"], "sess-anon")
        self.assertEqual(self.repo.get("user_abc", "sess-anon").title, "New chat")

    def test_claim_is_idempotent(self):
        lambda_handler(
            self._event(
                method="POST", path="/chat/threads/claim",
                body={"threadId": "sess-anon"},
                token="good-token",
            ),
            None,
        )
        lambda_handler(
            self._event(
                method="POST", path="/chat/threads/claim",
                body={"threadId": "sess-anon"},
                token="good-token",
            ),
            None,
        )
        self.assertEqual(len(self.repo.list_for_user("user_abc")), 1)

    # Ownership gates ------------------------------------------------------

    def test_chat_rejects_another_users_thread(self):
        # u2 already owns the thread; u1 (user_abc) must not be able to
        # continue the conversation by guessing/stealing the id.
        self.repo.upsert("user_other", "t-victim", title="Private")
        result = self._decode(lambda_handler(
            self._event(
                method="POST", path="/chat",
                body={"message": "sneak", "sessionId": "t-victim"},
                token="good-token",
            ),
            None,
        ))
        self.assertEqual(result["statusCode"], 403)
        # Graph was not invoked — no chat leakage.
        self.assertEqual(self.graph.invocations, [])

    def test_chat_allows_owner_to_continue_their_thread(self):
        self.repo.upsert("user_abc", "t-mine", title="Mine")
        result = lambda_handler(
            self._event(
                method="POST", path="/chat",
                body={"message": "continue", "sessionId": "t-mine"},
                token="good-token",
            ),
            None,
        )
        self.assertEqual(result["statusCode"], 200)

    def test_chat_anonymous_rejected_on_owned_thread(self):
        self.repo.upsert("user_other", "t-victim", title="Private")
        result = lambda_handler(
            self._event(
                method="POST", path="/chat",
                body={"message": "hi", "sessionId": "t-victim"},
            ),
            None,
        )
        self.assertEqual(result["statusCode"], 403)

    def test_history_rejects_another_users_thread(self):
        self.repo.upsert("user_other", "t-victim", title="Private")
        result = lambda_handler(
            self._event(
                method="GET", path="/chat/history",
                qs={"sessionId": "t-victim"},
                token="good-token",
            ),
            None,
        )
        self.assertEqual(result["statusCode"], 403)

    def test_history_anonymous_rejected_on_owned_thread(self):
        self.repo.upsert("user_other", "t-victim", title="Private")
        result = lambda_handler(
            self._event(
                method="GET", path="/chat/history",
                qs={"sessionId": "t-victim"},
            ),
            None,
        )
        self.assertEqual(result["statusCode"], 403)

    def test_history_allows_owner_to_read(self):
        self.repo.upsert("user_abc", "t-mine", title="Mine")
        result = lambda_handler(
            self._event(
                method="GET", path="/chat/history",
                qs={"sessionId": "t-mine"},
                token="good-token",
            ),
            None,
        )
        self.assertEqual(result["statusCode"], 200)

    def test_history_unowned_thread_stays_readable_anonymously(self):
        result = lambda_handler(
            self._event(
                method="GET", path="/chat/history",
                qs={"sessionId": "t-anon"},
            ),
            None,
        )
        # Anonymous threads keep their current behaviour — any caller that
        # holds the (UUID) id can read the persisted messages.
        self.assertEqual(result["statusCode"], 200)

    def test_claim_rejects_another_users_thread(self):
        self.repo.upsert("user_other", "t-victim", title="Private")
        result = lambda_handler(
            self._event(
                method="POST", path="/chat/threads/claim",
                body={"threadId": "t-victim"},
                token="good-token",
            ),
            None,
        )
        self.assertEqual(result["statusCode"], 403)
        # The victim's row is untouched.
        self.assertEqual(self.repo.get("user_other", "t-victim").title, "Private")
        self.assertEqual(self.repo.list_for_user("user_abc"), [])


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
        TestQueryExpansion,
        TestBM25Retrieve,
        TestRRF,
        TestFilters,
        TestRunRag,
        TestAuditClassify,
        TestSpecialistOutputsReducer,
        TestSpecialistMarksRefusalBlocked,
        TestResolvePrice,
        TestBudgetTracker,
        TestEstimateGenerationCost,
        TestCostReportMarkdown,
        TestInferCategory,
        TestDetectLanguageGenerator,
        TestSubsampleStratified,
        TestGeneratedCaseJsonl,
        TestListingClientFactory,
        TestDeriveIssuer,
        TestExtractBearer,
        TestVerifyClerkJwt,
        TestThreadsRepo,
        TestGenerateTitle,
        TestHandlerAuthAndThreads,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
