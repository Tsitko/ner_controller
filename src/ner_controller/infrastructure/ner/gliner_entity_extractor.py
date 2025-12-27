"""GLiNER-based entity extraction implementation."""

from typing import Optional, Sequence

from gliner import GLiNER

from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.levenshtein_utils import deduplicate_entities
from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)


class GlinerEntityExtractor(EntityExtractorInterface):
    """Extracts entities using the GLiNER model."""

    def __init__(self, config: GlinerEntityExtractorConfig) -> None:
        """Initialize the extractor with model configuration."""
        self._config = config
        self._model: Optional[GLiNER] = None

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[str]:
        """Extract entities from text using GLiNER."""
        if not text or not entity_types:
            return []

        model = self._load_model()
        predictions = model.predict_entities(text, list(entity_types))

        # Extract only entity text from predictions
        entities = [prediction.get("text", "") for prediction in predictions]

        # Apply deduplication with Levenshtein distance (threshold <= 2)
        return deduplicate_entities(entities, threshold=2)

    def _load_model(self) -> GLiNER:
        """Load the GLiNER model if it hasn't been loaded yet."""
        if self._model is None:
            self._model = GLiNER.from_pretrained(
                self._config.model_name,
                cache_dir=self._config.cache_dir,
                local_files_only=self._config.local_files_only,
            )
            if hasattr(self._model, "to"):
                self._model.to(self._config.device)
        return self._model

