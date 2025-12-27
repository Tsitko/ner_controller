"""Integration tests for hallucination detection service."""

import path_setup

path_setup.add_src_path()


import unittest
from typing import Sequence

from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator
from ner_controller.domain.services.hallucination_detection_service import (
    HallucinationDetectionService,
)


class StaticEntityExtractor(EntityExtractorInterface):
    """Simple extractor that returns preconfigured entities by text."""

    def __init__(self, mapping: dict[str, Sequence[str]]) -> None:
        """Initialize extractor with text-to-entities mapping."""
        self._mapping = mapping

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[str]:
        """Return entities for the given text, ignoring entity types."""
        return list(self._mapping.get(text, []))


class TestHallucinationDetectionIntegration(unittest.TestCase):
    """Integration test for service with real diff calculator."""

    def test_service_detects_expected_differences(self) -> None:
        """Service integrates extractor and diff calculator to produce results."""
        request_entities = ["Alice", "Paris"]
        response_entities = ["Bob", "Paris"]
        extractor = StaticEntityExtractor({"req": request_entities, "resp": response_entities})
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())

        result = service.detect("req", "resp", ["PERSON", "LOCATION"])

        self.assertEqual(result.potential_hallucinations, ["Bob"])
        self.assertEqual(result.missing_entities, ["Alice"])

    def test_service_with_deduplicated_entities(self) -> None:
        """Service handles deduplication correctly."""
        # Extractor returns duplicates
        request_entities = ["Alice", "Alice", "Bob", "Bob"]
        response_entities = ["Charlie", "Charlie", "Alice", "Alice"]
        extractor = StaticEntityExtractor({
            "req": request_entities,
            "resp": response_entities
        })
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())

        result = service.detect("req", "resp", ["PERSON"])

        # Should deduplicate before comparison
        self.assertEqual(result.potential_hallucinations, ["Charlie"])
        self.assertEqual(result.missing_entities, ["Bob"])

    def test_service_preserves_order_of_unique_entities(self) -> None:
        """Service preserves order of first occurrence for unique entities."""
        request_entities = ["Charlie", "Alice", "Bob"]
        response_entities = ["Bob", "David", "Charlie"]
        extractor = StaticEntityExtractor({
            "req": request_entities,
            "resp": response_entities
        })
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())

        result = service.detect("req", "resp", ["PERSON"])

        # Order should be preserved based on first occurrence
        self.assertEqual(result.potential_hallucinations, ["David"])
        self.assertEqual(result.missing_entities, ["Alice"])

    def test_service_with_unicode_entities(self) -> None:
        """Service handles Unicode entities correctly."""
        request_entities = ["Алиса", "Париж"]
        response_entities = ["алиса", "Боб"]

        extractor = StaticEntityExtractor({
            "Запрос": request_entities,  # Use actual request text as key
            "Ответ": response_entities   # Use actual response text as key
        })

        service = HallucinationDetectionService(extractor, EntityDiffCalculator())

        result = service.detect("Запрос", "Ответ", ["PERSON", "LOCATION"])

        # Case-insensitive matching: "алиса" matches "Алиса"
        # So only "Боб" is hallucination and "Париж" is missing
        self.assertEqual(result.potential_hallucinations, ["Боб"])
        self.assertEqual(result.missing_entities, ["Париж"])

    def test_service_with_empty_entity_lists(self) -> None:
        """Service handles empty entity lists correctly."""
        extractor = StaticEntityExtractor({"req": [], "resp": []})
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())

        result = service.detect("req", "resp", ["PERSON"])

        self.assertEqual(result.potential_hallucinations, [])
        self.assertEqual(result.missing_entities, [])

    def test_service_with_similar_entities(self) -> None:
        """Service applies Levenshtein-based deduplication."""
        request_entities = ["Apple", "Banana"]
        response_entities = ["Apples", "Banana"]
        extractor = StaticEntityExtractor({
            "req": request_entities,
            "resp": response_entities
        })
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())

        result = service.detect("req", "resp", ["PRODUCT"])

        # "Apples" and "Apple" have Levenshtein distance 1, so they match
        # No hallucinations since "Apples" matches "Apple"
        # No missing entities since all request entities are covered
        self.assertEqual(result.potential_hallucinations, [])
