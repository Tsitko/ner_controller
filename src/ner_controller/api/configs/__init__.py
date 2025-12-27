"""Configuration package for API layer."""

from ner_controller.api.configs.file_router_config import FileRouterConfig
from ner_controller.api.configs.hallucination_router_config import HallucinationRouterConfig

__all__ = [
    "FileRouterConfig",
    "HallucinationRouterConfig",
]
