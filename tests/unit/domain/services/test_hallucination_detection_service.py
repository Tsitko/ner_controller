"""Unit tests for HallucinationDetectionService."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock

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

        request_entities = ["Alice"]
        response_entities = ["Bob"]
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

    def test_detect_with_empty_request_text(self) -> None:
        """Service handles empty request text."""
        extractor = Mock(spec=EntityExtractorInterface)
        diff_calculator = Mock(spec=EntityDiffCalculator)

        extractor.extract.side_effect = [[], []]
        diff_result = EntityDiffResult(potential_hallucinations=[], missing_entities=[])
        diff_calculator.calculate.return_value = diff_result

        service = HallucinationDetectionService(extractor, diff_calculator)

        result = service.detect("", "resp", ["PERSON"])

        self.assertEqual(result.potential_hallucinations, [])
        self.assertEqual(result.missing_entities, [])

    def test_detect_with_empty_response_text(self) -> None:
        """Service handles empty response text."""
        extractor = Mock(spec=EntityExtractorInterface)
        diff_calculator = Mock(spec=EntityDiffCalculator)

        extractor.extract.side_effect = [["Alice"], []]
        diff_result = EntityDiffResult(potential_hallucinations=[], missing_entities=["Alice"])
        diff_calculator.calculate.return_value = diff_result

        service = HallucinationDetectionService(extractor, diff_calculator)

        result = service.detect("req", "", ["PERSON"])

        self.assertEqual(result.potential_hallucinations, [])
        self.assertEqual(result.missing_entities, ["Alice"])

    def test_detect_with_multiple_entity_types(self) -> None:
        """Service passes all entity types to extractor."""
        extractor = Mock(spec=EntityExtractorInterface)
        diff_calculator = Mock(spec=EntityDiffCalculator)

        entity_types = ["PERSON", "ORG", "LOCATION"]
        extractor.extract.side_effect = [["Alice"], ["Bob"]]
        diff_result = EntityDiffResult(potential_hallucinations=["Bob"], missing_entities=["Alice"])
        diff_calculator.calculate.return_value = diff_result

        service = HallucinationDetectionService(extractor, diff_calculator)

        service.detect("req", "resp", entity_types)

        extractor.extract.assert_any_call("req", entity_types)
        extractor.extract.assert_any_call("resp", entity_types)

    def test_detect_preserves_order_of_results(self) -> None:
        """Service preserves order from diff calculator result."""
        extractor = Mock(spec=EntityExtractorInterface)
        diff_calculator = Mock(spec=EntityDiffCalculator)

        diff_result = EntityDiffResult(
            potential_hallucinations=["Charlie", "Bob"],
            missing_entities=["Alice", "David"],
        )
        extractor.extract.side_effect = [[], []]
        diff_calculator.calculate.return_value = diff_result

        service = HallucinationDetectionService(extractor, diff_calculator)

        result = service.detect("req", "resp", ["PERSON"])

        self.assertEqual(result.potential_hallucinations, ["Charlie", "Bob"])
        self.assertEqual(result.missing_entities, ["Alice", "David"])

    def test_detect_with_unicode_texts(self) -> None:
        """Service handles Unicode text correctly."""
        extractor = Mock(spec=EntityExtractorInterface)
        diff_calculator = Mock(spec=EntityDiffCalculator)

        extractor.extract.side_effect = [["Алиса"], ["Боб"]]
        diff_result = EntityDiffResult(potential_hallucinations=["Боб"], missing_entities=["Алиса"])
        diff_calculator.calculate.return_value = diff_result

        service = HallucinationDetectionService(extractor, diff_calculator)

        result = service.detect("Запрос", "Ответ", ["PERSON"])

        self.assertEqual(result.potential_hallucinations, ["Боб"])
        self.assertEqual(result.missing_entities, ["Алиса"])

    def test_detect_with_deduplicated_entities(self) -> None:
        """Service returns deduplicated entities from diff calculator."""
        extractor = Mock(spec=EntityExtractorInterface)
        diff_calculator = Mock(spec=EntityDiffCalculator)

        # Extractor returns duplicates, but diff calculator deduplicates
        extractor.extract.side_effect = [["Alice", "Alice"], ["Bob", "Bob", "Bob"]]
        diff_result = EntityDiffResult(potential_hallucinations=["Bob"], missing_entities=["Alice"])
        diff_calculator.calculate.return_value = diff_result

        service = HallucinationDetectionService(extractor, diff_calculator)

        result = service.detect("req", "resp", ["PERSON"])

        # Should be deduplicated
        self.assertEqual(result.potential_hallucinations, ["Bob"])
        self.assertEqual(result.missing_entities, ["Alice"])
