"""Schema package for API payloads."""

from ner_controller.api.schemas.file_process_request import FileProcessRequest
from ner_controller.api.schemas.file_process_response import (
    ChunkSchema,
    FileProcessResponse,
)
from ner_controller.api.schemas.hallucination_check_request import HallucinationCheckRequest
from ner_controller.api.schemas.hallucination_check_response import HallucinationCheckResponse
from ner_controller.api.schemas.text_process_request import TextProcessRequest
from ner_controller.api.schemas.text_process_response import TextProcessResponse

__all__ = [
    "FileProcessRequest",
    "FileProcessResponse",
    "ChunkSchema",
    "HallucinationCheckRequest",
    "HallucinationCheckResponse",
    "TextProcessRequest",
    "TextProcessResponse",
]
