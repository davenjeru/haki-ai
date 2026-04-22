"""
Language detection test runner.

Two modes:
  --mock    Injects a MockComprehendClient into ComprehendAdapter.
            No AWS or LocalStack needed. Validates detect_language() logic.

  (default) Uses a real boto3 Comprehend client via the standard factory.
            Set ENV=local for LocalStack, or AWS_PROFILE for real AWS.

Usage:
  uv run test_language_detection.py --mock
  ENV=local uv run test_language_detection.py
  AWS_PROFILE=my-profile uv run test_language_detection.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph import _detect_language as detect_language
from clients.adapters import ComprehendAdapter
from clients import make_comprehend
from app.config import load_config

# ── Test cases ────────────────────────────────────────────────────────────────

CASES = [
    {
        "text": "What are my rights under the Employment Act if I am unfairly dismissed?",
        "expected": "english",
        "label": "pure English",
    },
    {
        "text": "Haki zangu ni zipi chini ya Sheria ya Ajira nikiachwa kazi bila sababu?",
        "expected": "swahili",
        "label": "pure Swahili",
    },
    {
        "text": "What are haki zangu under the Employment Act?",
        "expected": "mixed",
        "label": "mixed English/Swahili",
    },
]

# ── Mock client ───────────────────────────────────────────────────────────────

class MockComprehendClient:
    """
    Returns canned Comprehend-style payloads to test detect_language() in
    isolation. Scores mirror what real Comprehend typically returns.
    Injected directly into ComprehendAdapter — same path as production.
    """

    _SWAHILI_WORDS = {
        "haki", "zangu", "chini", "sheria", "ajira",
        "nikiachwa", "kazi", "bila", "sababu", "ni", "zipi",
    }
    _ENGLISH_WORDS = {
        "what", "are", "my", "rights", "under", "the",
        "act", "if", "i", "am", "unfairly", "dismissed",
    }

    def detect_dominant_language(self, Text: str) -> dict:
        words = set(Text.lower().split())
        has_sw = bool(words & self._SWAHILI_WORDS)
        has_en = bool(words & self._ENGLISH_WORDS)

        if has_sw and has_en:
            languages = [
                {"LanguageCode": "en", "Score": 0.72},
                {"LanguageCode": "sw", "Score": 0.26},
            ]
        elif has_sw:
            languages = [
                {"LanguageCode": "sw", "Score": 0.94},
                {"LanguageCode": "en", "Score": 0.04},
            ]
        else:
            languages = [
                {"LanguageCode": "en", "Score": 0.98},
                {"LanguageCode": "sw", "Score": 0.01},
            ]
        return {"Languages": languages}

# ── Runner ────────────────────────────────────────────────────────────────────

def run(adapter: ComprehendAdapter) -> int:
    failures = 0
    for case in CASES:
        try:
            result = detect_language(case["text"], adapter)
        except Exception as err:
            print(f"  [ERROR] {case['label']}: {err}")
            failures += 1
            continue

        passed = result == case["expected"]
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1

        print(f"  [{status}] {case['label']}")
        print(f"         expected={case['expected']!r}  got={result!r}")
        print(f"         text: {case['text'][:65]}...")
        print()

    return failures


def main():
    use_mock = "--mock" in sys.argv

    if use_mock:
        print("Mode: mock (no AWS calls)\n")
        adapter = ComprehendAdapter(MockComprehendClient(), is_local=False)
    else:
        env = os.environ.get("ENV", "prod")
        print(f"Mode: boto3  ENV={env}\n")
        config = load_config()
        adapter = ComprehendAdapter(make_comprehend(config), config.is_local)

    failures = run(adapter)
    total = len(CASES)
    passed = total - failures

    print(f"Result: {passed}/{total} passed", end="")
    if failures:
        print(f"  ({failures} failed)")
        sys.exit(1)
    else:
        print(" ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
