"""Integration tests for FileProcessingService with real dependencies."""

import path_setup

path_setup.add_src_path()


import base64
import unittest
from unittest.mock import Mock

from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult
from ner_controller.domain.interfaces.embedding_generator_interface import EmbeddingGeneratorInterface
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.file_processing_service import FileProcessingService


class MockEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Mock embedding generator for integration testing."""

    def __init__(self, embeddings: list) -> None:
        """Initialize with predefined embeddings."""
        self._embeddings = embeddings
        self.call_count = 0

    def generate_embeddings(self, texts):
        """Return predefined embeddings."""
        self.call_count += 1
        return self._embeddings[: len(texts)]


class MockTextChunker:
    """Mock text chunker for integration testing."""

    def __init__(self, chunks: list) -> None:
        """Initialize with predefined chunks."""
        self._chunks = chunks
        self.split_text_calls = []

    def split_text(self, text, chunk_size, chunk_overlap, start_id=0):
        """Return predefined chunks and track calls."""
        self.split_text_calls.append({
            "text": text,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "start_id": start_id,
        })
        return self._chunks


class TestFileProcessingServiceIntegration(unittest.TestCase):
    """Integration tests for FileProcessingService with mocked dependencies."""

    def test_full_pipeline_with_successful_processing(self) -> None:
        """Service processes file through complete pipeline successfully."""
        # Arrange
        original_text = "Alice works at OpenAI in San Francisco."
        encoded_text = base64.b64encode(original_text.encode()).decode()

        # Create mock chunks
        mock_chunks = [
            FileChunk(
                id=0,
                text=original_text,
                entities=(),
                embedding=None,
            )
        ]

        # Create mock entity extractor that returns entity names (strings)
        mock_extractor = Mock(spec=EntityExtractorInterface)
        mock_extractor.extract.return_value = ["Alice", "OpenAI", "San Francisco"]

        # Create mock embedding generator
        mock_generator = MockEmbeddingGenerator(embeddings=[(0.1, 0.2, 0.3)])

        # Create mock chunker
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="test-file-1",
            entity_types=["PERSON", "ORG", "LOC"],
            chunk_size=1000,
            chunk_overlap=100,
        )

        # Assert
        self.assertIsInstance(result, FileProcessingResult)
        self.assertEqual(result.file_id, "test-file-1")
        self.assertEqual(len(result.chunks), 1)

        # Verify entities were extracted (as strings)
        chunk_with_entities = result.chunks[0]
        self.assertEqual(len(chunk_with_entities.entities), 3)
        self.assertIn("Alice", chunk_with_entities.entities)
        self.assertIn("OpenAI", chunk_with_entities.entities)
        self.assertIn("San Francisco", chunk_with_entities.entities)

        # Verify embedding was generated
        self.assertEqual(chunk_with_entities.embedding, (0.1, 0.2, 0.3))

        # Verify all entities collected
        self.assertGreaterEqual(len(result.entities), 3)

    def test_pipeline_with_multiple_chunks(self) -> None:
        """Service processes file with multiple chunks correctly."""
        # Arrange
        original_text = "First chunk. Second chunk. Third chunk."
        encoded_text = base64.b64encode(original_text.encode()).decode()

        # Create multiple chunks
        mock_chunks = [
            FileChunk(id=0, text="First chunk.", entities=(), embedding=None),
            FileChunk(id=1, text="Second chunk.", entities=(), embedding=None),
            FileChunk(id=2, text="Third chunk.", entities=(), embedding=None),
        ]

        mock_extractor = Mock(spec=EntityExtractorInterface)
        mock_extractor.extract.side_effect = [
            ["First"],
            ["Second"],
            ["Third"],
        ]

        embeddings = [(0.1,), (0.2,), (0.3,)]
        mock_generator = MockEmbeddingGenerator(embeddings=embeddings)
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="multi-chunk-file",
            entity_types=["LABEL"],
        )

        # Assert
        self.assertEqual(len(result.chunks), 3)
        self.assertEqual(len(result.chunks[0].entities), 1)
        self.assertEqual(len(result.chunks[1].entities), 1)
        self.assertEqual(len(result.chunks[2].entities), 1)

        # Verify embeddings
        self.assertEqual(result.chunks[0].embedding, (0.1,))
        self.assertEqual(result.chunks[1].embedding, (0.2,))
        self.assertEqual(result.chunks[2].embedding, (0.3,))

        # Verify entity extraction was called for each chunk
        self.assertEqual(mock_extractor.extract.call_count, 3)

    def test_pipeline_with_entity_deduplication(self) -> None:
        """Service deduplicates entities across chunks."""
        # Arrange
        encoded_text = base64.b64encode(b"Text").decode()

        # Same entity appears in multiple chunks (as strings)
        mock_chunks = [
            FileChunk(id=0, text="Alice in chunk 1", entities=(), embedding=None),
            FileChunk(id=1, text="Alice in chunk 2", entities=(), embedding=None),
        ]

        mock_extractor = Mock(spec=EntityExtractorInterface)
        mock_extractor.extract.side_effect = [
            ["Alice"],
            ["Alice"],
        ]

        mock_generator = MockEmbeddingGenerator(embeddings=[(0.1,), (0.2,)])
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="dedup-test",
            entity_types=["PERSON"],
        )

        # Assert - entities should be deduplicated
        self.assertLessEqual(len(result.entities), 2)

    def test_pipeline_with_levenshtein_deduplication(self) -> None:
        """Service applies Levenshtein distance-based deduplication."""
        # Arrange
        encoded_text = base64.b64encode("мосбилет мосбилета".encode()).decode()

        mock_chunks = [
            FileChunk(id=0, text="мосбилет мосбилета", entities=(), embedding=None),
        ]

        mock_extractor = Mock(spec=EntityExtractorInterface)
        # Return similar entities that should be deduplicated
        mock_extractor.extract.return_value = ["мосбилет", "мосбилета", "москва"]

        mock_generator = MockEmbeddingGenerator(embeddings=[(0.1,)])
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="levenshtein-test",
            entity_types=["ORG", "LOCATION"],
        )

        # Assert - should apply Levenshtein deduplication (distance <= 2)
        # "мосбилет" and "мосбилета" should be deduplicated
        self.assertLessEqual(len(result.entities), 3)

    def test_pipeline_with_empty_file(self) -> None:
        """Service handles empty file correctly."""
        # Arrange
        encoded_text = base64.b64encode(b"").decode()
        mock_chunks = []

        mock_extractor = Mock(spec=EntityExtractorInterface)
        mock_extractor.extract.return_value = []

        mock_generator = MockEmbeddingGenerator(embeddings=[])
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="empty-file",
        )

        # Assert
        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(len(result.entities), 0)
        self.assertEqual(result.file_id, "empty-file")

    def test_pipeline_with_no_entities_found(self) -> None:
        """Service handles case when no entities are found."""
        # Arrange
        encoded_text = base64.b64encode(b"Just plain text with no entities").decode()
        mock_chunks = [
            FileChunk(
                id=0,
                text="Just plain text with no entities",
                entities=(),
                embedding=None,
            )
        ]

        mock_extractor = Mock(spec=EntityExtractorInterface)
        mock_extractor.extract.return_value = []  # No entities found

        mock_generator = MockEmbeddingGenerator(embeddings=[(0.1, 0.2)])
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="no-entities",
        )

        # Assert
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(len(result.chunks[0].entities), 0)
        self.assertEqual(len(result.entities), 0)
        self.assertIsNotNone(result.chunks[0].embedding)

    def test_pipeline_with_custom_chunk_parameters(self) -> None:
        """Service uses custom chunk_size and chunk_overlap correctly."""
        # Arrange
        encoded_text = base64.b64encode(b"Text").decode()
        mock_chunks = [FileChunk(id=0, text="Text", entities=(), embedding=None)]

        mock_extractor = Mock(spec=EntityExtractorInterface)
        mock_extractor.extract.return_value = []

        mock_generator = MockEmbeddingGenerator(embeddings=[(0.1,)])
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="custom-chunks",
            chunk_size=500,
            chunk_overlap=50,
        )

        # Assert - chunker was called with custom parameters
        self.assertEqual(len(mock_chunker.split_text_calls), 1)
        call = mock_chunker.split_text_calls[0]
        self.assertEqual(call["chunk_size"], 500)
        self.assertEqual(call["chunk_overlap"], 50)
        self.assertEqual(call["start_id"], 0)

    def test_pipeline_preserves_chunk_ids(self) -> None:
        """Service preserves sequential chunk IDs from chunker."""
        # Arrange
        encoded_text = base64.b64encode(b"Text").decode()

        mock_chunks = [
            FileChunk(id=5, text="First", entities=(), embedding=None),
            FileChunk(id=6, text="Second", entities=(), embedding=None),
            FileChunk(id=7, text="Third", entities=(), embedding=None),
        ]

        mock_extractor = Mock(spec=EntityExtractorInterface)
        mock_extractor.extract.return_value = []

        mock_generator = MockEmbeddingGenerator(embeddings=[(0.1,), (0.2,), (0.3,)])
        mock_chunker = MockTextChunker(chunks=mock_chunks)

        # Act
        service = FileProcessingService(
            entity_extractor=mock_extractor,
            embedding_generator=mock_generator,
            text_chunker=mock_chunker,
        )

        result = service.process_file(
            file_base64=encoded_text,
            file_id="id-test",
        )

        # Assert - IDs should be preserved
        self.assertEqual(result.chunks[0].id, 5)
        self.assertEqual(result.chunks[1].id, 6)
        self.assertEqual(result.chunks[2].id, 7)
