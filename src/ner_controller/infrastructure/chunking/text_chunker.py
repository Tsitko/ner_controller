"""Text chunking implementation with overlap support."""

from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.interfaces.text_chunker_interface import TextChunkerInterface
from ner_controller.infrastructure.chunking.configs.text_chunker_config import TextChunkerConfig


class TextChunker(TextChunkerInterface):
    """
    Splits text into overlapping chunks.

    Implements a simple character-based chunking strategy with overlap.
    Chunks are created by stepping through the text with stride = (chunk_size - overlap).
    """

    def __init__(self, config: TextChunkerConfig) -> None:
        """
        Initialize chunker with configuration.

        Args:
            config: Configuration for chunking behavior.
        """
        self._config = config

    def split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        start_id: int = 0,
    ) -> list[FileChunk]:
        """
        Split text into chunks with overlap.

        Args:
            text: The input text to split.
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of characters to overlap between chunks.
            start_id: Starting ID for the first chunk.

        Returns:
            List of FileChunk objects with sequential IDs.
            Entities and embeddings are initially empty.

        Raises:
            ValueError: If parameters are invalid.
        """
        self._validate_parameters(chunk_size, chunk_overlap)

        if not text:
            return []

        chunks = []
        stride = chunk_size - chunk_overlap
        position = 0
        chunk_id = start_id

        while True:
            # Check if we have enough text for a full chunk
            if position >= len(text):
                break

            # Extract chunk text
            end_position = min(position + chunk_size, len(text))
            chunk_text = text[position:end_position]

            # Only create chunk if it has content
            if chunk_text:
                chunk = FileChunk(
                    id=chunk_id,
                    text=chunk_text,
                    entities=(),
                    embedding=None,
                )
                chunks.append(chunk)
                chunk_id += 1

            # Move to next chunk
            position += stride

            # Stop if we've covered the entire text
            if position >= len(text):
                break

        return chunks

    def _validate_parameters(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Validate chunking parameters.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap characters between chunks.

        Raises:
            ValueError: If parameters are invalid.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap must be < chunk_size, got chunk_overlap={chunk_overlap}, chunk_size={chunk_size}")
