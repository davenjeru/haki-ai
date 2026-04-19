"""
Adapters for AWS service clients.

Each adapter wraps a boto3 client and handles two things:
  1. Presents a simplified interface (only the methods handler.py needs).
  2. Papers over LocalStack limitations so business logic stays clean.

Business logic functions receive adapters, not raw boto3 clients.
"""

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
