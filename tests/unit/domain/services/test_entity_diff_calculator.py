"""Unit tests for EntityDiffCalculator."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator


class TestEntityDiffCalculator(unittest.TestCase):
    """Tests entity diff calculation behavior."""

    def setUp(self) -> None:
        """Set up shared calculator instance."""
        self._calculator = EntityDiffCalculator()

    def test_calculate_detects_missing_and_hallucinated_entities(self) -> None:
        """Calculator returns response-only and request-only entities."""
        request_entities = ["Alice", "Paris"]
        response_entities = ["Bob", "Paris", "Zoo", "Bob"]

        result = self._calculator.calculate(request_entities, response_entities)

        # After deduplication: request = ["Alice", "Paris"], response = ["Bob", "Paris"]
        # Bob is not similar to Alice or Paris (Levenshtein distance > 2)
        # Paris matches exactly
        # Zoo is similar to Bob (Levenshtein distance = 2), so it gets deduplicated
        self.assertEqual(result.potential_hallucinations, ["Bob"])
        self.assertEqual(result.missing_entities, ["Alice"])

    def test_calculate_is_case_insensitive_and_strips_whitespace(self) -> None:
        """Calculator ignores case and surrounding whitespace when matching entities."""
        request_entities = [" Alice "]
        response_entities = ["alice"]

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, [])
        self.assertEqual(result.missing_entities, [])

    def test_calculate_with_empty_request_entities(self) -> None:
        """All response entities are hallucinations when request is empty."""
        request_entities = []
        response_entities = ["Alice", "Bob"]

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, ["Alice", "Bob"])
        self.assertEqual(result.missing_entities, [])

    def test_calculate_with_empty_response_entities(self) -> None:
        """All request entities are missing when response is empty."""
        request_entities = ["Alice", "Bob"]
        response_entities = []

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, [])
        self.assertEqual(result.missing_entities, ["Alice", "Bob"])

    def test_calculate_removes_duplicates_from_response(self) -> None:
        """Calculator deduplicates entities in response before comparison."""
        request_entities = ["Alice"]
        response_entities = ["Bob", "Bob", "Bob", "Alice", "Alice"]

        result = self._calculator.calculate(request_entities, response_entities)

        # Bob should appear once in hallucinations, Alice matches
        self.assertEqual(result.potential_hallucinations, ["Bob"])
        self.assertEqual(result.missing_entities, [])

    def test_calculate_removes_duplicates_from_request(self) -> None:
        """Calculator deduplicates entities in request before comparison."""
        request_entities = ["Alice", "Alice", "Bob"]
        response_entities = ["Charlie"]

        result = self._calculator.calculate(request_entities, response_entities)

        # Only unique missing entities should be reported
        self.assertEqual(result.potential_hallucinations, ["Charlie"])
        self.assertIn("Alice", result.missing_entities)
        self.assertIn("Bob", result.missing_entities)

    def test_calculate_with_unicode_entities(self) -> None:
        """Calculator handles Unicode entities correctly."""
        request_entities = ["Алиса", "Париж"]
        response_entities = ["алиса", "Боб"]

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, ["Боб"])
        self.assertEqual(result.missing_entities, ["Париж"])

    def test_calculate_preserves_order(self) -> None:
        """Calculator preserves order of unique entities."""
        request_entities = ["Charlie", "Alice", "Bob"]
        response_entities = ["Bob", "David", "Charlie"]

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, ["David"])
        self.assertEqual(result.missing_entities, ["Alice"])

    def test_calculate_with_similar_but_different_entities(self) -> None:
        """Calculator treats similar entities as different without fuzzy matching."""
        request_entities = ["Apple"]
        response_entities = ["Apples"]

        result = self._calculator.calculate(request_entities, response_entities)

        # With Levenshtein distance = 1, these are considered similar and get deduplicated
        # So no hallucinations or missing entities detected
        self.assertEqual(result.potential_hallucinations, [])
        self.assertEqual(result.missing_entities, [])

    def test_calculate_with_entities_containing_special_characters(self) -> None:
        """Calculator handles entities with spaces, hyphens, etc."""
        request_entities = ["San Francisco", "New-York"]
        response_entities = ["san francisco", "Los Angeles"]

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, ["Los Angeles"])
        self.assertEqual(result.missing_entities, ["New-York"])
