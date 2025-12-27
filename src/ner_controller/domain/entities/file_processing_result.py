"""Result of file processing with NER and embeddings."""

from dataclasses import dataclass

from ner_controller.domain.entities.file_chunk import FileChunk


@dataclass(frozen=True)
class FileProcessingResult:
    """Result of processing a file into chunks with entities and embeddings."""

    file_id: str
    entities: tuple[str, ...]
    chunks: tuple[FileChunk, ...]
