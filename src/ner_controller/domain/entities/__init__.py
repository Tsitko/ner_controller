"""Domain entity definitions."""

from ner_controller.domain.entities.entity import Entity
from ner_controller.domain.entities.entity_diff_result import EntityDiffResult
from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult
from ner_controller.domain.entities.hallucination_detection_result import (
    HallucinationDetectionResult,
)
from ner_controller.domain.entities.text_processing_result import TextProcessingResult

__all__ = [
    "Entity",
    "FileChunk",
    "FileProcessingResult",
    "HallucinationDetectionResult",
    "EntityDiffResult",
    "TextProcessingResult",
]
