"""Service for processing files with NER and embeddings."""

import base64
from typing import Sequence

from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult
from ner_controller.domain.interfaces.embedding_generator_interface import EmbeddingGeneratorInterface
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.interfaces.text_chunker_interface import TextChunkerInterface


# Default entity types as specified in requirements
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

DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 300


class FileProcessingService:
    """
    Orchestrates file processing: decode, chunk, NER extraction, and embedding generation.

    Responsibilities:
    - Decode base64 file content
    - Split text into chunks
    - Extract entities from each chunk
    - Generate embeddings for each chunk
    - Aggregate results
    """

    def __init__(
        self,
        entity_extractor: EntityExtractorInterface,
        embedding_generator: EmbeddingGeneratorInterface,
        text_chunker: TextChunkerInterface,
    ) -> None:
        """
        Initialize service with dependencies.

        Args:
            entity_extractor: NER engine for extracting entities from text.
            embedding_generator: Service for generating text embeddings.
            text_chunker: Strategy for splitting text into chunks.
        """
        self._entity_extractor = entity_extractor
        self._embedding_generator = embedding_generator
        self._text_chunker = text_chunker

    def process_file(
        self,
        file_base64: str,
        file_id: str,
        entity_types: Sequence[str] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> FileProcessingResult:
        """
        Process a base64-encoded file through the complete pipeline.

        Pipeline steps:
        1. Decode base64 to text
        2. Split text into chunks
        3. Extract entities from each chunk
        4. Generate embeddings for each chunk
        5. Aggregate all entities and chunks

        Args:
            file_base64: Base64-encoded file content.
            file_id: Unique identifier for the file.
            entity_types: Entity types to extract. Uses DEFAULT_ENTITY_TYPES if None.
            chunk_size: Maximum characters per chunk. Default: 3000.
            chunk_overlap: Overlap characters between chunks. Default: 300.

        Returns:
            FileProcessingResult with all entities and chunk data.

        Raises:
            ValueError: If base64 decoding fails or parameters are invalid.
            EmbeddingGenerationError: If embedding generation fails.
        """
        # Determine entity types to use
        target_entity_types = list(entity_types) if entity_types else DEFAULT_ENTITY_TYPES

        # Decode file content
        text = self._decode_base64(file_base64)

        # Split into chunks
        chunks = self._text_chunker.split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_id=0,
        )

        # Extract entities from each chunk
        chunks_with_entities = self._extract_entities_from_chunks(chunks, target_entity_types)

        # Generate embeddings for chunks
        chunks_with_embeddings = self._generate_embeddings_for_chunks(chunks_with_entities)

        # Collect all unique entities from all chunks
        all_entities = self._collect_all_entities(chunks_with_embeddings)

        return FileProcessingResult(
            file_id=file_id,
            entities=tuple(all_entities),
            chunks=tuple(chunks_with_embeddings),
        )

    def _decode_base64(self, encoded: str) -> str:
        """
        Decode base64-encoded string to text.

        Args:
            encoded: Base64-encoded string.

        Returns:
            Decoded text string.

        Raises:
            ValueError: If decoding fails or result is not valid UTF-8.
        """
        try:
            decoded_bytes = base64.b64decode(encoded)
            text = decoded_bytes.decode("utf-8")
            return text.strip()
        except Exception as e:
            raise ValueError(f"Failed to decode base64 content: {str(e)}")

    def _extract_entities_from_chunks(
        self,
        chunks: list[FileChunk],
        entity_types: list[str],
    ) -> list[FileChunk]:
        """
        Extract entities from each chunk in parallel or batch.

        Args:
            chunks: List of chunks without entities.
            entity_types: Entity types to extract.

        Returns:
            List of chunks with populated entities.
        """
        chunks_with_entities = []
        for chunk in chunks:
            entities = self._entity_extractor.extract(chunk.text, entity_types)
            updated_chunk = FileChunk(
                id=chunk.id,
                text=chunk.text,
                entities=tuple(entities),
                embedding=chunk.embedding,
            )
            chunks_with_entities.append(updated_chunk)
        return chunks_with_entities

    def _generate_embeddings_for_chunks(
        self,
        chunks: list[FileChunk],
    ) -> list[FileChunk]:
        """
        Generate embeddings for all chunks in batches.

        Args:
            chunks: List of chunks with entities but without embeddings.

        Returns:
            List of chunks with populated embeddings.

        Raises:
            EmbeddingGenerationError: If embedding generation fails.
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embedding_generator.generate_embeddings(texts)

        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            updated_chunk = FileChunk(
                id=chunk.id,
                text=chunk.text,
                entities=chunk.entities,
                embedding=tuple(embedding) if embedding is not None else None,
            )
            chunks_with_embeddings.append(updated_chunk)

        return chunks_with_embeddings

    def _collect_all_entities(self, chunks: list[FileChunk]) -> list[str]:
        """
        Collect unique entities from all chunks.

        Args:
            chunks: List of chunks with entities.

        Returns:
            List of unique entities across all chunks, deduplicated using Levenshtein distance.
        """
        from ner_controller.domain.services.levenshtein_utils import deduplicate_entities

        # Collect all entities from all chunks
        all_entities: list[str] = []
        for chunk in chunks:
            all_entities.extend(chunk.entities)

        # Deduplicate using Levenshtein distance with threshold <= 2
        return deduplicate_entities(all_entities, threshold=2)
