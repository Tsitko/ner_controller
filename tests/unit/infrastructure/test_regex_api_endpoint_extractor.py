"""Unit tests for RegexApiEndpointExtractor."""

import sys
import path_setup

path_setup.add_src_path()

# Mock any modules that might import gliner
from unittest.mock import MagicMock
sys.modules["gliner"] = MagicMock()

import unittest

from ner_controller.infrastructure.ner.regex_api_endpoint_extractor import (
    RegexApiEndpointExtractor,
)


class TestRegexApiEndpointExtractor(unittest.TestCase):
    """Tests regex-based API endpoint extraction behavior."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.extractor = RegexApiEndpointExtractor()

    def test_extract_finds_post_endpoints(self) -> None:
        """Extractor finds POST endpoints in text."""
        text = "Use POST /billing/getUserLimitBalanceAllPoints to get data."
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("POST /billing/getUserLimitBalanceAllPoints", result)

    def test_extract_finds_get_endpoints(self) -> None:
        """Extractor finds GET endpoints in text."""
        text = "Call GET /api/users to retrieve users."
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("GET /api/users", result)

    def test_extract_finds_delete_endpoints(self) -> None:
        """Extractor finds DELETE endpoints in text."""
        text = "Use DELETE /cashback/precalculation to remove."
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("DELETE /cashback/precalculation", result)

    def test_extract_finds_multiple_endpoints(self) -> None:
        """Extractor finds multiple different endpoints in text."""
        text = """
        POST /billing/getUserLimitBalanceAllPoints
        GET /api/users
        DELETE /cashback/precalculation
        PUT /api/users/update
        """
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 4)
        self.assertIn("POST /billing/getUserLimitBalanceAllPoints", result)
        self.assertIn("GET /api/users", result)
        self.assertIn("DELETE /cashback/precalculation", result)
        self.assertIn("PUT /api/users/update", result)

    def test_extract_handles_case_insensitive(self) -> None:
        """Extractor handles HTTP methods case-insensitively."""
        text = "use post /api/test or POST /api/test2 or Post /api/test3"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 3)
        self.assertIn("post /api/test", result)
        self.assertIn("POST /api/test2", result)
        self.assertIn("Post /api/test3", result)

    def test_extract_supports_all_http_methods(self) -> None:
        """Extractor supports all common HTTP methods."""
        text = """
        GET /api/get
        POST /api/post
        PUT /api/put
        DELETE /api/delete
        PATCH /api/patch
        HEAD /api/head
        OPTIONS /api/options
        """
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 7)
        self.assertIn("GET /api/get", result)
        self.assertIn("POST /api/post", result)
        self.assertIn("PUT /api/put", result)
        self.assertIn("DELETE /api/delete", result)
        self.assertIn("PATCH /api/patch", result)
        self.assertIn("HEAD /api/head", result)
        self.assertIn("OPTIONS /api/options", result)

    def test_extract_handles_paths_with_underscores_and_hyphens(self) -> None:
        """Extractor handles paths with underscores and hyphens."""
        text = "GET /api/user-profile/get_user_data"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("GET /api/user-profile/get_user_data", result)

    def test_extract_handles_nested_paths(self) -> None:
        """Extractor handles deeply nested API paths."""
        text = "POST /api/v1/billing/accounts/user-id/transactions"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("POST /api/v1/billing/accounts/user-id/transactions", result)

    def test_extract_returns_empty_list_for_no_endpoints(self) -> None:
        """Extractor returns empty list when no endpoints found."""
        text = "This is just regular text without API endpoints."
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, list)

    def test_extract_returns_empty_list_for_empty_text(self) -> None:
        """Extractor returns empty list for empty text."""
        result = self.extractor.extract("", ["API Endpoint"])

        self.assertEqual(len(result), 0)

    def test_extract_ignores_entity_types_parameter(self) -> None:
        """Extractor ignores entity_types parameter (for interface compatibility)."""
        text = "GET /api/test"
        result1 = self.extractor.extract(text, ["API Endpoint"])
        result2 = self.extractor.extract(text, ["Person", "Organization"])
        result3 = self.extractor.extract(text, [])

        # Should return same results regardless of entity_types
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        self.assertEqual(len(result3), 1)

    def test_extract_normalizes_whitespace(self) -> None:
        """Extractor normalizes whitespace in endpoints."""
        text = "GET   /api/test   with extra spaces"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("GET /api/test", result)

    def test_extract_preserves_duplicates(self) -> None:
        """Extractor preserves duplicate endpoints (deduplication handled by composite)."""
        text = "GET /api/test and GET /api/test again"
        result = self.extractor.extract(text, ["API Endpoint"])

        # Returns duplicates, composite extractor will deduplicate
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "GET /api/test")
        self.assertEqual(result[1], "GET /api/test")

    def test_extract_handles_endpoints_with_numbers(self) -> None:
        """Extractor handles endpoints with numeric segments."""
        text = "POST /api/v2/users/12345/update"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("POST /api/v2/users/12345/update", result)

    def test_extract_handles_mixed_case_paths(self) -> None:
        """Extractor handles paths with mixed case."""
        text = "GET /api/UserProfile/GetUserData"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("GET /api/UserProfile/GetUserData", result)

    def test_extract_handles_unicode_in_paths(self) -> None:
        """Extractor handles Unicode characters in paths."""
        text = "GET /api/пользователь/данные"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("GET /api/пользователь/данные", result)

    def test_extract_only_matches_word_boundaries(self) -> None:
        """Extractor only matches complete HTTP methods at word boundaries."""
        text = "myPOST /api/test should not match, but POST /api/test should"
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 1)
        self.assertIn("POST /api/test", result)

    def test_extract_handles_realistic_api_documentation(self) -> None:
        """Extractor handles realistic API documentation text."""
        text = """
        API Documentation:

        To get user balance, use:
        POST /billing/getUserLimitBalanceAllPoints

        To retrieve users:
        GET /api/users

        To delete precalculation:
        DELETE /cashback/precalculation

        For updates:
        PUT /api/users/{id}
        """
        result = self.extractor.extract(text, ["API Endpoint"])

        self.assertEqual(len(result), 4)
        self.assertIn("POST /billing/getUserLimitBalanceAllPoints", result)
        self.assertIn("GET /api/users", result)
        self.assertIn("DELETE /cashback/precalculation", result)
        self.assertIn("PUT /api/users/{id}", result)
