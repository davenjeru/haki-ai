"""
Unit tests for backend business logic.

No AWS or LocalStack required — all external dependencies are stubbed.
Tests cover edge cases not exercised by test_language_detection.py or
test_e2e_local.py.

Usage:
  uv run test_unit.py
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from handler import detect_language, lambda_handler, _make_rag_adapter
from adapters import ComprehendAdapter, StubRAGAdapter
from prompts import build_system_prompt, BILINGUAL_REFUSAL
from rag import check_guardrail_block, blocked_response
from citations import extract_citations
from config import load_config


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _FixedComprehendClient:
    """Returns a fixed language list."""
    def __init__(self, languages: list[dict]):
        self._languages = languages

    def detect_dominant_language(self, Text: str) -> dict:
        return {"Languages": self._languages}


def _adapter(languages: list[dict]) -> ComprehendAdapter:
    return ComprehendAdapter(_FixedComprehendClient(languages), is_local=False)


# ── detect_language ───────────────────────────────────────────────────────────

class TestDetectLanguage(unittest.TestCase):

    def test_high_confidence_english(self):
        adapter = _adapter([{"LanguageCode": "en", "Score": 0.97}])
        self.assertEqual(detect_language("hello", adapter), "english")

    def test_high_confidence_swahili(self):
        adapter = _adapter([{"LanguageCode": "sw", "Score": 0.91}])
        self.assertEqual(detect_language("habari", adapter), "swahili")

    def test_mixed_when_both_languages_present(self):
        adapter = _adapter([
            {"LanguageCode": "en", "Score": 0.72},
            {"LanguageCode": "sw", "Score": 0.26},
        ])
        self.assertEqual(detect_language("hello habari", adapter), "mixed")

    def test_english_below_threshold_no_swahili_defaults_english(self):
        # en is top but below 0.85, no sw present → default english
        adapter = _adapter([{"LanguageCode": "en", "Score": 0.70}])
        self.assertEqual(detect_language("bonjour", adapter), "english")

    def test_swahili_below_threshold_no_english_still_swahili(self):
        # sw is top but below 0.85, no en present → swahili fallback
        adapter = _adapter([{"LanguageCode": "sw", "Score": 0.70}])
        self.assertEqual(detect_language("habari", adapter), "swahili")

    def test_unknown_language_defaults_english(self):
        adapter = _adapter([{"LanguageCode": "fr", "Score": 0.95}])
        self.assertEqual(detect_language("bonjour", adapter), "english")

    def test_empty_language_list_defaults_english(self):
        adapter = _adapter([])
        self.assertEqual(detect_language("", adapter), "english")

    def test_empty_message_defaults_english(self):
        adapter = _adapter([{"LanguageCode": "en", "Score": 0.99}])
        self.assertEqual(detect_language("", adapter), "english")


# ── build_system_prompt ───────────────────────────────────────────────────────

class TestBuildSystemPrompt(unittest.TestCase):

    def test_english_prompt_contains_english_instruction(self):
        prompt = build_system_prompt("english")
        self.assertIn("Respond in English.", prompt)

    def test_swahili_prompt_contains_swahili_instruction(self):
        prompt = build_system_prompt("swahili")
        self.assertIn("Kiswahili", prompt)

    def test_mixed_prompt_contains_both_language_instructions(self):
        prompt = build_system_prompt("mixed")
        self.assertIn("English", prompt)
        self.assertIn("Swahili", prompt)

    def test_unknown_language_falls_back_to_english(self):
        prompt = build_system_prompt("klingon")
        self.assertIn("Respond in English.", prompt)

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


# ── check_guardrail_block ─────────────────────────────────────────────────────

class TestCheckGuardrailBlock(unittest.TestCase):

    def test_guardrail_intervened_returns_true(self):
        self.assertTrue(check_guardrail_block({"stopReason": "guardrail_intervened"}))

    def test_end_turn_returns_false(self):
        self.assertFalse(check_guardrail_block({"stopReason": "end_turn"}))

    def test_missing_stop_reason_returns_false(self):
        self.assertFalse(check_guardrail_block({}))

    def test_other_stop_reason_returns_false(self):
        self.assertFalse(check_guardrail_block({"stopReason": "max_tokens"}))

    def test_blocked_response_contains_bilingual_refusal(self):
        self.assertEqual(blocked_response("english"), BILINGUAL_REFUSAL)
        self.assertEqual(blocked_response("swahili"), BILINGUAL_REFUSAL)


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
        self.assertEqual(result[0]["chunkId"], "chunk-1")
        self.assertEqual(result[1]["chunkId"], "chunk-2")

    def test_page_image_key_included_when_present(self):
        result = extract_citations(self._rag_result([self._ref("chunk-1")]))
        self.assertIn("pageImageKey", result[0])

    def test_page_image_key_absent_when_missing(self):
        ref = self._ref("chunk-1")
        del ref["metadata"]["pageImageKey"]
        result = extract_citations(self._rag_result([ref]))
        self.assertNotIn("pageImageKey", result[0])

    def test_empty_citations_returns_empty_list(self):
        result = extract_citations({"output": {"text": ""}, "citations": [], "stopReason": "end_turn"})
        self.assertEqual(result, [])

    def test_falls_back_to_s3_uri_when_chunk_id_missing(self):
        ref = self._ref("chunk-1")
        del ref["metadata"]["chunkId"]
        result = extract_citations(self._rag_result([ref]))
        self.assertEqual(result[0]["chunkId"], "s3://haki-ai-data/processed-chunks/chunk-1.txt")

    def test_deduplicates_across_multiple_citation_groups(self):
        rag_result = {
            "output": {"text": "answer"},
            "citations": [
                {"retrievedReferences": [self._ref("chunk-1")]},
                {"retrievedReferences": [self._ref("chunk-1"), self._ref("chunk-2")]},
            ],
            "stopReason": "end_turn",
        }
        result = extract_citations(rag_result)
        self.assertEqual(len(result), 2)

    def test_missing_metadata_fields_return_empty_strings(self):
        ref = {"content": {"text": "text"}, "metadata": {"chunkId": "chunk-x"}, "location": {}}
        result = extract_citations(self._rag_result([ref]))
        self.assertEqual(result[0]["source"], "")
        self.assertEqual(result[0]["chapter"], "")
        self.assertEqual(result[0]["section"], "")


# ── lambda_handler ────────────────────────────────────────────────────────────

class TestLambdaHandler(unittest.TestCase):
    """
    Tests lambda_handler directly using StubRAGAdapter.
    LocalStack not required.
    """

    def setUp(self):
        # Patch adapters so no AWS calls are made
        import handler as h
        import adapters as a
        import metrics as m

        self._orig_make_comprehend = h.make_comprehend
        self._orig_make_cloudwatch = h.make_cloudwatch
        self._orig_make_rag = h._make_rag_adapter

        # Comprehend: return English
        class _FakeComprehendClient:
            def detect_dominant_language(self, Text):
                return {"Languages": [{"LanguageCode": "en", "Score": 0.98}]}

        h.make_comprehend = lambda config: _FakeComprehendClient()

        # CloudWatch: no-op
        class _FakeCloudWatch:
            def put_metric_data(self, **kwargs): pass

        h.make_cloudwatch = lambda config: _FakeCloudWatch()

        # RAG: stub
        h._make_rag_adapter = lambda config: StubRAGAdapter()

    def tearDown(self):
        import handler as h
        h.make_comprehend = self._orig_make_comprehend
        h.make_cloudwatch = self._orig_make_cloudwatch
        h._make_rag_adapter = self._orig_make_rag

    def _invoke(self, message: str) -> dict:
        event = {"body": json.dumps({"message": message})}
        result = lambda_handler(event, None)
        result["body"] = json.loads(result["body"])
        return result

    def test_valid_message_returns_200(self):
        result = self._invoke("What are my rights?")
        self.assertEqual(result["statusCode"], 200)

    def test_empty_message_returns_400(self):
        result = self._invoke("")
        self.assertEqual(result["statusCode"], 400)
        self.assertIn("error", result["body"])

    def test_whitespace_only_message_returns_400(self):
        result = self._invoke("   ")
        self.assertEqual(result["statusCode"], 400)

    def test_missing_body_returns_400(self):
        result = lambda_handler({"body": None}, None)
        body = json.loads(result["body"])
        self.assertEqual(result["statusCode"], 400)

    def test_response_shape_is_complete(self):
        result = self._invoke("What are my land rights?")
        body = result["body"]
        self.assertIn("response", body)
        self.assertIn("citations", body)
        self.assertIn("language", body)
        self.assertIn("blocked", body)

    def test_cors_header_present(self):
        result = self._invoke("test")
        self.assertEqual(result["headers"]["Access-Control-Allow-Origin"], "*")

    def test_blocked_response_when_guardrail_fires(self):
        import handler as h

        class _BlockingStub:
            def retrieve_and_generate(self, query, system_prompt, model_id):
                return {"output": {"text": ""}, "citations": [], "stopReason": "guardrail_intervened"}

        h._make_rag_adapter = lambda config: _BlockingStub()
        result = self._invoke("What are my rights?")
        body = result["body"]
        self.assertEqual(result["statusCode"], 200)
        self.assertTrue(body["blocked"])
        self.assertEqual(body["response"], BILINGUAL_REFUSAL)
        self.assertEqual(body["citations"], [])


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestDetectLanguage,
        TestBuildSystemPrompt,
        TestCheckGuardrailBlock,
        TestExtractCitations,
        TestLambdaHandler,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
