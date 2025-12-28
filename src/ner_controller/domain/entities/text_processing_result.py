"""Result of single text processing with NER and embeddings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextProcessingResult:
    """
    Result of processing a single text with entities and embedding.

    Attributes:
        text: Original input text.
        entities: Deduplicated list of extracted entity strings.
        embedding: Embedding vector for the full text.
    """

    text: str
    entities: tuple[str, ...]
    embedding: tuple[float, ...]
