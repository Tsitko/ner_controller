"""Unit tests for HallucinationDetectionService."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock

from ner_controller.domain.entities.entity import Entity
from ner_controller.domain.entities.entity_diff_result import EntityDiffResult
from ner_controller.domain.entities.hallucination_detection_result import (
    HallucinationDetectionResult,
)
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator
from ner_controller.domain.services.hallucination_detection_service import (
    HallucinationDetectionService,
)


class TestHallucinationDetectionService(unittest.TestCase):
    """Tests orchestration inside HallucinationDetectionService."""

    def test_detect_uses_extractor_and_diff_calculator(self) -> None:
        """Service calls dependencies and returns a detection result."""
        extractor = Mock(spec=EntityExtractorInterface)
        diff_calculator = Mock(spec=EntityDiffCalculator)

        request_entities = [Entity(text="Alice", label="PERSON", start=0, end=5)]
        response_entities = [Entity(text="Bob", label="PERSON", start=0, end=3)]
        diff_result = EntityDiffResult(
            potential_hallucinations=["Bob"],
            missing_entities=["Alice"],
        )

        extractor.extract.side_effect = [request_entities, response_entities]
        diff_calculator.calculate.return_value = diff_result

        service = HallucinationDetectionService(extractor, diff_calculator)

        result = service.detect("req", "resp", ["PERSON"])

        extractor.extract.assert_any_call("req", ["PERSON"])
        extractor.extract.assert_any_call("resp", ["PERSON"])
        diff_calculator.calculate.assert_called_once_with(request_entities, response_entities)
        self.assertIsInstance(result, HallucinationDetectionResult)
        self.assertEqual(result.potential_hallucinations, ["Bob"])
        self.assertEqual(result.missing_entities, ["Alice"])
