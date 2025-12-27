"""Router package for API endpoints."""

from ner_controller.api.routers.file_router import FileRouter
from ner_controller.api.routers.hallucination_router import HallucinationRouter

__all__ = [
    "FileRouter",
    "HallucinationRouter",
]
