"""Application entrypoint and FastAPI wiring."""

from typing import Optional

from fastapi import FastAPI

from ner_controller.api.configs.hallucination_router_config import HallucinationRouterConfig
from ner_controller.api.routers.hallucination_router import HallucinationRouter
from ner_controller.configs.app_config import AppConfig
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator
from ner_controller.domain.services.hallucination_detection_service import HallucinationDetectionService
from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)
from ner_controller.infrastructure.ner.gliner_entity_extractor import GlinerEntityExtractor


class ApplicationFactory:
    """Builds and configures the FastAPI application."""

    def __init__(
        self,
        config: AppConfig,
        service: Optional[HallucinationDetectionService] = None,
    ) -> None:
        """Initialize the factory with application configuration."""
        self._config = config
        self._service = service

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application instance."""
        app = FastAPI(title=self._config.title, docs_url=self._config.docs_url)

        service = self._service or self._build_service()
        router_config = HallucinationRouterConfig()
        router = HallucinationRouter(router_config, service).create_router()

        app.include_router(router, prefix=router_config.prefix, tags=list(router_config.tags))

        return app

    def _build_service(self) -> HallucinationDetectionService:
        """Build the default hallucination detection service."""
        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())
        diff_calculator = EntityDiffCalculator()
        return HallucinationDetectionService(extractor, diff_calculator)


app = ApplicationFactory(AppConfig()).create_app()
