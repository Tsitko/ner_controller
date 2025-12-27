"""Chunk of text extracted from a file."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FileChunk:
    """Represents a text chunk with extracted entities and embedding."""

    id: int
    text: str
    entities: tuple[str, ...]
    embedding: tuple[float, ...] | None
