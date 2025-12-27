"""Domain interfaces for dependency inversion."""

from ner_controller.domain.interfaces.embedding_generator_interface import (
    EmbeddingGeneratorInterface,
)
from ner_controller.domain.interfaces.entity_extractor_interface import (
    EntityExtractorInterface,
)
from ner_controller.domain.interfaces.text_chunker_interface import TextChunkerInterface

__all__ = [
    "EmbeddingGeneratorInterface",
    "EntityExtractorInterface",
    "TextChunkerInterface",
]
