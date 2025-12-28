"""Service for processing single text with NER and embeddings."""

from typing import Sequence

from ner_controller.domain.entities.text_processing_result import TextProcessingResult
from ner_controller.domain.interfaces.embedding_generator_interface import EmbeddingGeneratorInterface
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface


# Reuse default entity types from FileProcessingService
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Product",
    "Service",
    "Technology",
    "Concept",
    "Time",
    "Money",
    "Quantity",
    "ClassName",
    "Library",
    "Framework",
    "Language",
    "Tool",
    "Methodology",
    "Standard",
    "Protocol",
    "API Endpoint",
    "Date",
]


class TextProcessingService:
    """
    Orchestrates single text processing: NER extraction and embedding generation.

    Responsibilities:
    - Validate input text
    - Extract entities from the full text (no chunking)
    - Generate a single embedding for the full text
    - Return aggregated result
    """

    def __init__(
        self,
        entity_extractor: EntityExtractorInterface,
        embedding_generator: EmbeddingGeneratorInterface,
    ) -> None:
        """
        Initialize service with dependencies.

        Args:
            entity_extractor: NER engine for extracting entities from text.
            embedding_generator: Service for generating text embeddings.
        """
        self._entity_extractor = entity_extractor
        self._embedding_generator = embedding_generator

    def process_text(
        self,
        text: str,
        entity_types: Sequence[str] | None = None,
    ) -> TextProcessingResult:
        """
        Process a single text through the complete pipeline.

        Pipeline steps:
        1. Validate and normalize text
        2. Extract entities from the full text
        3. Generate embedding for the full text
        4. Aggregate results

        Args:
            text: Plain text string to process.
            entity_types: Entity types to extract. Uses DEFAULT_ENTITY_TYPES if None.

        Returns:
            TextProcessingResult with entities and embedding.

        Raises:
            ValueError: If text is empty or parameters are invalid.
            EmbeddingGenerationError: If embedding generation fails.
        """
        # Validate and normalize text
        normalized_text = self._validate_and_normalize_text(text)

        # Determine entity types to use
        target_entity_types = list(entity_types) if entity_types else DEFAULT_ENTITY_TYPES

        # Extract entities from the full text
        entities = self._extract_entities(normalized_text, target_entity_types)

        # Generate embedding for the full text
        embedding = self._generate_embedding(normalized_text)

        return TextProcessingResult(
            text=normalized_text,
            entities=tuple(entities),
            embedding=tuple(embedding),
        )

    def _validate_and_normalize_text(self, text: str) -> str:
        """
        Validate and normalize input text.

        Args:
            text: Raw input text.

        Returns:
            Normalized text (stripped of leading/trailing whitespace).

        Raises:
            ValueError: If text is empty or only whitespace.
        """
        if not text:
            raise ValueError("Text cannot be empty")

        normalized = text.strip()
        if not normalized:
            raise ValueError("Text cannot be empty or only whitespace")

        return normalized

    def _extract_entities(
        self,
        text: str,
        entity_types: list[str],
    ) -> list[str]:
        """
        Extract entities from text.

        Args:
            text: Text to extract entities from.
            entity_types: Entity types to extract.

        Returns:
            List of deduplicated entity strings.

        Raises:
            Exception: If entity extraction fails.
        """
        entities = self._entity_extractor.extract(text, entity_types)
        return list(entities)

    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to generate embedding for.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingGenerationError: If embedding generation fails.
        """
        embeddings = self._embedding_generator.generate_embeddings([text])

        if not embeddings or embeddings[0] is None:
            from ner_controller.infrastructure.embedding.ollama_embedding_generator import (
                EmbeddingGenerationError,
            )
            raise EmbeddingGenerationError("Failed to generate embedding for text")

        return list(embeddings[0])
