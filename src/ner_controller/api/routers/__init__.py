"""Router package for API endpoints."""

from ner_controller.api.routers.file_router import FileRouter
from ner_controller.api.routers.hallucination_router import HallucinationRouter
from ner_controller.api.routers.text_router import TextRouter

__all__ = [
    "FileRouter",
    "HallucinationRouter",
    "TextRouter",
]
