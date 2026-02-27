"""Application entrypoint and FastAPI wiring."""

from typing import Optional

from fastapi import FastAPI

from ner_controller.api.configs.file_router_config import FileRouterConfig
from ner_controller.api.configs.hallucination_router_config import HallucinationRouterConfig
from ner_controller.api.configs.text_router_config import TextRouterConfig
from ner_controller.api.routers.file_router import FileRouter
from ner_controller.api.routers.hallucination_router import HallucinationRouter
from ner_controller.api.routers.text_router import TextRouter
from ner_controller.configs.app_config import AppConfig
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator
from ner_controller.domain.services.file_processing_service import FileProcessingService
from ner_controller.domain.services.hallucination_detection_service import HallucinationDetectionService
from ner_controller.domain.services.text_processing_service import TextProcessingService
from ner_controller.infrastructure.chunking.configs.text_chunker_config import TextChunkerConfig
from ner_controller.infrastructure.chunking.text_chunker import TextChunker
from ner_controller.infrastructure.embedding.configs.lm_studio_embedding_generator_config import (
    LmStudioEmbeddingGeneratorConfig,
)
from ner_controller.infrastructure.embedding.lm_studio_embedding_generator import (
    LmStudioEmbeddingGenerator,
)
from ner_controller.infrastructure.ner.composite_entity_extractor import CompositeEntityExtractor
from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)
from ner_controller.infrastructure.ner.configs.natasha_entity_extractor_config import (
    NatashaEntityExtractorConfig,
)
from ner_controller.infrastructure.ner.gliner_entity_extractor import GlinerEntityExtractor
from ner_controller.infrastructure.ner.natasha_entity_extractor import NatashaEntityExtractor
from ner_controller.infrastructure.ner.regex_api_endpoint_extractor import (
    RegexApiEndpointExtractor,
)


class ApplicationFactory:
    """Builds and configures the FastAPI application."""

    def __init__(
        self,
        config: AppConfig,
        service: Optional[HallucinationDetectionService] = None,
        file_processing_service: Optional[FileProcessingService] = None,
        text_processing_service: Optional[TextProcessingService] = None,
    ) -> None:
        """Initialize the factory with application configuration."""
        self._config = config
        self._service = service
        self._file_processing_service = file_processing_service
        self._text_processing_service = text_processing_service

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application instance."""
        app = FastAPI(title=self._config.title, docs_url=self._config.docs_url)

        hallucination_service = self._service or self._build_hallucination_service()
        hallucination_router_config = HallucinationRouterConfig()
        hallucination_router = HallucinationRouter(hallucination_router_config, hallucination_service).create_router()
        app.include_router(
            hallucination_router,
            prefix=hallucination_router_config.prefix,
        )

        file_processing_service = self._file_processing_service or self._build_file_processing_service()
        file_router_config = FileRouterConfig()
        file_router = FileRouter(file_router_config, file_processing_service).create_router()
        app.include_router(file_router)

        text_processing_service = self._text_processing_service or self._build_text_processing_service()
        text_router_config = TextRouterConfig()
        text_router = TextRouter(text_router_config, text_processing_service).create_router()
        app.include_router(text_router)

        return app

    def _build_hallucination_service(self) -> HallucinationDetectionService:
        """Build the default hallucination detection service."""
        extractor = self.create_entity_extractor()
        diff_calculator = EntityDiffCalculator()
        return HallucinationDetectionService(extractor, diff_calculator)

    def _build_file_processing_service(self) -> FileProcessingService:
        """Build the default file processing service."""
        entity_extractor = self.create_entity_extractor()
        embedding_generator = LmStudioEmbeddingGenerator(LmStudioEmbeddingGeneratorConfig())
        text_chunker = TextChunker(TextChunkerConfig())
        return FileProcessingService(entity_extractor, embedding_generator, text_chunker)

    def _build_text_processing_service(self) -> TextProcessingService:
        """Build the default text processing service."""
        entity_extractor = self.create_entity_extractor()
        embedding_generator = LmStudioEmbeddingGenerator(LmStudioEmbeddingGeneratorConfig())
        return TextProcessingService(entity_extractor, embedding_generator)

    def create_entity_extractor(self) -> CompositeEntityExtractor:
        """
        Create a composite entity extractor instance.

        The composite extractor runs GLiNER, Natasha, and regex extractors.
        Results are combined and deduplicated.
        """
        gliner_extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())
        natasha_extractor = NatashaEntityExtractor(NatashaEntityExtractorConfig())
        regex_extractor = RegexApiEndpointExtractor()
        return CompositeEntityExtractor([gliner_extractor, natasha_extractor, regex_extractor])
