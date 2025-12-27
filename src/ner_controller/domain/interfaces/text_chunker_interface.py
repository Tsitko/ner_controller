"""Interface for splitting text into chunks."""

from abc import ABC, abstractmethod

from ner_controller.domain.entities.file_chunk import FileChunk


class TextChunkerInterface(ABC):
    """Contract for splitting text into overlapping chunks."""

    @abstractmethod
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
            Entities and embeddings are initially empty/None.

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size.
        """
        raise NotImplementedError
