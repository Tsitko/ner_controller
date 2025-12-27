"""Configuration for text chunker."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunkerConfig:
    """Configuration values for text chunking strategy."""

    # Chunking behavior settings (currently not used but available for future extensions)
    preserve_sentences: bool = True
    min_chunk_size: int = 100
