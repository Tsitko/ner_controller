"""Domain service for hallucination detection."""

from typing import Sequence

from ner_controller.domain.entities.hallucination_detection_result import (
    HallucinationDetectionResult,
)
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator


class HallucinationDetectionService:
    """Coordinates NER extraction and entity comparison."""

    def __init__(
        self,
        extractor: EntityExtractorInterface,
        diff_calculator: EntityDiffCalculator,
    ) -> None:
        """Initialize service with extractor and diff calculator."""
        self._extractor = extractor
        self._diff_calculator = diff_calculator

    def detect(
        self,
        request_text: str,
        response_text: str,
        entity_types: Sequence[str],
    ) -> HallucinationDetectionResult:
        """Detect hallucinations and missing entities between request and response."""
        request_entities = self._extractor.extract(request_text, entity_types)
        response_entities = self._extractor.extract(response_text, entity_types)
        diff = self._diff_calculator.calculate(request_entities, response_entities)
        return HallucinationDetectionResult(
            potential_hallucinations=diff.potential_hallucinations,
            missing_entities=diff.missing_entities,
        )
