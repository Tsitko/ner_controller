"""Integration tests for hallucination detection service."""

import path_setup

path_setup.add_src_path()


import unittest
from typing import Sequence

from ner_controller.domain.entities.entity import Entity
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator
from ner_controller.domain.services.hallucination_detection_service import (
    HallucinationDetectionService,
)


class StaticEntityExtractor(EntityExtractorInterface):
    """Simple extractor that returns preconfigured entities by text."""

    def __init__(self, mapping: dict[str, Sequence[Entity]]) -> None:
        """Initialize extractor with text-to-entities mapping."""
        self._mapping = mapping

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[Entity]:
        """Return entities for the given text, ignoring entity types."""
        return list(self._mapping.get(text, []))


class TestHallucinationDetectionIntegration(unittest.TestCase):
    """Integration test for service with real diff calculator."""

    def test_service_detects_expected_differences(self) -> None:
        """Service integrates extractor and diff calculator to produce results."""
        request_entities = [
            Entity(text="Alice", label="PERSON", start=0, end=5),
            Entity(text="Paris", label="LOCATION", start=10, end=15),
        ]
        response_entities = [
            Entity(text="Bob", label="PERSON", start=0, end=3),
            Entity(text="Paris", label="LOCATION", start=10, end=15),
        ]
        extractor = StaticEntityExtractor({"req": request_entities, "resp": response_entities})
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())

        result = service.detect("req", "resp", ["PERSON", "LOCATION"])

        self.assertEqual(result.potential_hallucinations, ["Bob"])
        self.assertEqual(result.missing_entities, ["Alice"])
