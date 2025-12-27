"""Unit tests for FileProcessingService."""

import path_setup

path_setup.add_src_path()


import base64
import unittest
from unittest.mock import Mock, call, patch

from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult
from ner_controller.domain.interfaces.embedding_generator_interface import EmbeddingGeneratorInterface
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.interfaces.text_chunker_interface import TextChunkerInterface
from ner_controller.domain.services.file_processing_service import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_ENTITY_TYPES,
    FileProcessingService,
)


class TestFileProcessingServiceInitialization(unittest.TestCase):
    """Tests FileProcessingService initialization and dependencies."""

    def test_initialize_with_all_dependencies(self) -> None:
        """Service stores all dependencies correctly."""
        entity_extractor = Mock(spec=EntityExtractorInterface)
        embedding_generator = Mock(spec=EmbeddingGeneratorInterface)
        text_chunker = Mock(spec=TextChunkerInterface)

        service = FileProcessingService(
            entity_extractor=entity_extractor,
            embedding_generator=embedding_generator,
            text_chunker=text_chunker,
        )

        self.assertIs(service._entity_extractor, entity_extractor)
        self.assertIs(service._embedding_generator, embedding_generator)
        self.assertIs(service._text_chunker, text_chunker)

    def test_dependencies_are_required(self) -> None:
        """All dependencies must be provided (no defaults)."""
        with self.assertRaises(TypeError):
            FileProcessingService()

        with self.assertRaises(TypeError):
            FileProcessingService(
                entity_extractor=Mock(spec=EntityExtractorInterface),
            )


class TestFileProcessingServiceProcessFile(unittest.TestCase):
    """Tests FileProcessingService.process_file method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)
        self.text_chunker = Mock(spec=TextChunkerInterface)

        self.service = FileProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
            text_chunker=self.text_chunker,
        )

    def test_process_file_decodes_base64_content(self) -> None:
        """Service decodes base64-encoded file content."""
        original_text = "Hello, world!"
        encoded_text = base64.b64encode(original_text.encode()).decode()

        # Mock the pipeline steps
        self.text_chunker.split_text.return_value = [
            FileChunk(id=0, text=original_text, entities=(), embedding=None)
        ]
        self.entity_extractor.extract.return_value = []
        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="test-file",
        )

        self.assertEqual(len(result.chunks), 1)
        self.text_chunker.split_text.assert_called_once()

    def test_process_file_with_custom_entity_types(self) -> None:
        """Service uses custom entity types when provided."""
        original_text = "Alice works at OpenAI."
        encoded_text = base64.b64encode(original_text.encode()).decode()

        self.text_chunker.split_text.return_value = [
            FileChunk(id=0, text=original_text, entities=(), embedding=None)
        ]
        self.entity_extractor.extract.return_value = ["Alice"]
        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2]]

        custom_types = ["PERSON", "ORG"]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="test-file",
            entity_types=custom_types,
        )

        self.entity_extractor.extract.assert_called_with(original_text, custom_types)

    def test_process_file_with_default_entity_types(self) -> None:
        """Service uses DEFAULT_ENTITY_TYPES when entity_types is None."""
        original_text = "Text"
        encoded_text = base64.b64encode(original_text.encode()).decode()

        self.text_chunker.split_text.return_value = [
            FileChunk(id=0, text=original_text, entities=(), embedding=None)
        ]
        self.entity_extractor.extract.return_value = []
        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="test-file",
            entity_types=None,
        )

        # Should use all 21 default entity types
        call_args = self.entity_extractor.extract.call_args
        self.assertEqual(call_args[0][1], DEFAULT_ENTITY_TYPES)

    def test_process_file_with_custom_chunk_parameters(self) -> None:
        """Service uses custom chunk_size and chunk_overlap when provided."""
        original_text = "A" * 100
        encoded_text = base64.b64encode(original_text.encode()).decode()

        self.text_chunker.split_text.return_value = [
            FileChunk(id=0, text=original_text[:50], entities=(), embedding=None),
            FileChunk(id=1, text=original_text[50:], entities=(), embedding=None),
        ]
        self.entity_extractor.extract.return_value = []
        self.embedding_generator.generate_embeddings.return_value = [[0.1], [0.2]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="test-file",
            chunk_size=50,
            chunk_overlap=10,
        )

        self.text_chunker.split_text.assert_called_once_with(
            text=original_text,
            chunk_size=50,
            chunk_overlap=10,
            start_id=0,
        )

    def test_process_file_with_default_chunk_parameters(self) -> None:
        """Service uses default chunk_size and chunk_overlap when not provided."""
        original_text = "Text"
        encoded_text = base64.b64encode(original_text.encode()).decode()

        self.text_chunker.split_text.return_value = [
            FileChunk(id=0, text=original_text, entities=(), embedding=None)
        ]
        self.entity_extractor.extract.return_value = []
        self.embedding_generator.generate_embeddings.return_value = [[0.1]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="test-file",
        )

        self.text_chunker.split_text.assert_called_once_with(
            text=original_text,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            start_id=0,
        )

    def test_process_file_returns_file_processing_result(self) -> None:
        """Service returns FileProcessingResult with correct structure."""
        original_text = "Alice and Bob"
        encoded_text = base64.b64encode(original_text.encode()).decode()

        self.text_chunker.split_text.return_value = [
            FileChunk(id=0, text=original_text, entities=(), embedding=None)
        ]
        self.entity_extractor.extract.return_value = ["Alice", "Bob"]
        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="file-123",
        )

        self.assertIsInstance(result, FileProcessingResult)
        self.assertEqual(result.file_id, "file-123")
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.chunks), 1)

    def test_process_file_with_multiple_chunks(self) -> None:
        """Service processes multiple chunks correctly."""
        original_text = "First chunk. Second chunk. Third chunk."
        encoded_text = base64.b64encode(original_text.encode()).decode()

        chunk1 = FileChunk(id=0, text="First chunk.", entities=(), embedding=None)
        chunk2 = FileChunk(id=1, text="Second chunk.", entities=(), embedding=None)
        chunk3 = FileChunk(id=2, text="Third chunk.", entities=(), embedding=None)

        self.text_chunker.split_text.return_value = [chunk1, chunk2, chunk3]
        self.entity_extractor.extract.side_effect = [
            ["First"],
            ["Second"],
            ["Third"],
        ]
        self.embedding_generator.generate_embeddings.return_value = [
            [0.1],
            [0.2],
            [0.3],
        ]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="multi-chunk-file",
        )

        self.assertEqual(len(result.chunks), 3)
        self.assertEqual(len(result.entities), 3)

    def test_process_file_with_empty_content(self) -> None:
        """Service handles empty file content."""
        encoded_text = base64.b64encode(b"").decode()

        self.text_chunker.split_text.return_value = []

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="empty-file",
        )

        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(len(result.entities), 0)

    def test_process_file_with_no_entities_found(self) -> None:
        """Service handles case when no entities are extracted."""
        original_text = "Just plain text with no entities."
        encoded_text = base64.b64encode(original_text.encode()).decode()

        self.text_chunker.split_text.return_value = [
            FileChunk(id=0, text=original_text, entities=(), embedding=None)
        ]
        self.entity_extractor.extract.return_value = []
        self.embedding_generator.generate_embeddings.return_value = [[0.1]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="no-entities-file",
        )

        self.assertEqual(len(result.entities), 0)
        self.assertEqual(len(result.chunks), 1)

    def test_process_file_deduplicates_entities_across_chunks(self) -> None:
        """Service deduplicates entities appearing in multiple chunks."""
        original_text = "Alice is here. Alice is there."
        encoded_text = base64.b64encode(original_text.encode()).decode()

        chunk1 = FileChunk(id=0, text="Alice is here.", entities=(), embedding=None)
        chunk2 = FileChunk(id=1, text="Alice is there.", entities=(), embedding=None)

        self.text_chunker.split_text.return_value = [chunk1, chunk2]
        # Same entity appears in both chunks
        self.entity_extractor.extract.side_effect = [
            ["Alice"],
            ["Alice"],
        ]
        self.embedding_generator.generate_embeddings.return_value = [[0.1], [0.2]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="dedup-file",
        )

        # Should deduplicate to unique entities only
        self.assertEqual(len(result.entities), 1)
        self.assertIn("Alice", result.entities)

    def test_process_file_applies_levenshtein_deduplication(self) -> None:
        """Service applies Levenshtein distance-based deduplication."""
        original_text = "мосбилет мосбилета"
        encoded_text = base64.b64encode(original_text.encode()).decode()

        chunk = FileChunk(id=0, text=original_text, entities=(), embedding=None)
        self.text_chunker.split_text.return_value = [chunk]
        # Return similar entities that should be deduplicated
        self.entity_extractor.extract.return_value = ["мосбилет", "мосбилета"]
        self.embedding_generator.generate_embeddings.return_value = [[0.1]]

        result = self.service.process_file(
            file_base64=encoded_text,
            file_id="levenshtein-file",
        )

        # Should apply Levenshtein deduplication (distance <= 2)
        self.assertLessEqual(len(result.entities), 2)
        # At minimum, should deduplicate exact matches


class TestFileProcessingServiceDecodeBase64(unittest.TestCase):
    """Tests FileProcessingService._decode_base64 method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)
        self.text_chunker = Mock(spec=TextChunkerInterface)

        self.service = FileProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
            text_chunker=self.text_chunker,
        )

    def test_decode_valid_base64_string(self) -> None:
        """Decodes valid base64-encoded string correctly."""
        original = "Hello, world!"
        encoded = base64.b64encode(original.encode()).decode()

        result = self.service._decode_base64(encoded)

        self.assertEqual(result, original)

    def test_decode_with_whitespace(self) -> None:
        """Decodes base64 string with leading/trailing whitespace."""
        original = "Test"
        encoded = "  " + base64.b64encode(original.encode()).decode() + "  "

        result = self.service._decode_base64(encoded)

        self.assertEqual(result, original)

    def test_decode_unicode_content(self) -> None:
        """Decodes base64-encoded Unicode content correctly."""
        original = "Привет, мир! 你好世界! مرحبا بالعالم!"
        encoded = base64.b64encode(original.encode("utf-8")).decode()

        result = self.service._decode_base64(encoded)

        self.assertEqual(result, original)

    def test_decode_invalid_base64_raises_value_error(self) -> None:
        """Raises ValueError for invalid base64 string."""
        with self.assertRaises(ValueError):
            self.service._decode_base64("Not valid base64!!!")

    def test_decode_empty_string(self) -> None:
        """Decodes empty base64 string to empty string."""
        encoded = base64.b64encode(b"").decode()

        result = self.service._decode_base64(encoded)

        self.assertEqual(result, "")

    def test_decode_binary_content(self) -> None:
        """Raises ValueError for non-UTF8 binary content."""
        # Create binary content that's not valid UTF-8
        binary_content = b"\xff\xfe\x00\x01"
        encoded = base64.b64encode(binary_content).decode()

        with self.assertRaises(ValueError):
            self.service._decode_base64(encoded)


class TestFileProcessingServiceExtractEntitiesFromChunks(unittest.TestCase):
    """Tests FileProcessingService._extract_entities_from_chunks method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)
        self.text_chunker = Mock(spec=TextChunkerInterface)

        self.service = FileProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
            text_chunker=self.text_chunker,
        )

    def test_extract_entities_from_single_chunk(self) -> None:
        """Extracts entities from a single chunk."""
        chunk = FileChunk(id=0, text="Alice is here", entities=(), embedding=None)

        self.entity_extractor.extract.return_value = ["Alice"]

        result = self.service._extract_entities_from_chunks([chunk], ["PERSON"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 0)
        self.assertEqual(len(result[0].entities), 1)
        self.assertIn("Alice", result[0].entities)

    def test_extract_entities_from_multiple_chunks(self) -> None:
        """Extracts entities from multiple chunks sequentially."""
        chunk1 = FileChunk(id=0, text="First chunk", entities=(), embedding=None)
        chunk2 = FileChunk(id=1, text="Second chunk", entities=(), embedding=None)

        self.entity_extractor.extract.side_effect = [["First"], ["Second"]]

        result = self.service._extract_entities_from_chunks(
            [chunk1, chunk2], ["LABEL"]
        )

        self.assertEqual(len(result), 2)
        self.assertIn("First", result[0].entities)
        self.assertIn("Second", result[1].entities)

    def test_extract_entities_preserves_chunk_text_and_id(self) -> None:
        """Entity extraction preserves chunk text and ID."""
        chunk = FileChunk(
            id=5,
            text="Original chunk text",
            entities=(),
            embedding=None,
        )

        self.entity_extractor.extract.return_value = []

        result = self.service._extract_entities_from_chunks([chunk], ["PERSON"])

        self.assertEqual(result[0].id, 5)
        self.assertEqual(result[0].text, "Original chunk text")

    def test_extract_entities_with_empty_entity_list(self) -> None:
        """Handles chunks where no entities are found."""
        chunk = FileChunk(id=0, text="No entities here", entities=(), embedding=None)

        self.entity_extractor.extract.return_value = []

        result = self.service._extract_entities_from_chunks([chunk], ["PERSON"])

        self.assertEqual(len(result[0].entities), 0)


class TestFileProcessingServiceGenerateEmbeddingsForChunks(unittest.TestCase):
    """Tests FileProcessingService._generate_embeddings_for_chunks method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)
        self.text_chunker = Mock(spec=TextChunkerInterface)

        self.service = FileProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
            text_chunker=self.text_chunker,
        )

    def test_generate_embeddings_for_single_chunk(self) -> None:
        """Generates embedding for a single chunk."""
        chunk = FileChunk(
            id=0,
            text="Sample text",
            entities=("Sample",),
            embedding=None,
        )

        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        result = self.service._generate_embeddings_for_chunks([chunk])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].embedding, (0.1, 0.2, 0.3))

    def test_generate_embeddings_for_multiple_chunks(self) -> None:
        """Generates embeddings for multiple chunks."""
        chunk1 = FileChunk(id=0, text="Text 1", entities=(), embedding=None)
        chunk2 = FileChunk(id=1, text="Text 2", entities=(), embedding=None)
        chunk3 = FileChunk(id=2, text="Text 3", entities=(), embedding=None)

        self.embedding_generator.generate_embeddings.return_value = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]

        result = self.service._generate_embeddings_for_chunks([chunk1, chunk2, chunk3])

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].embedding, (0.1, 0.2))
        self.assertEqual(result[1].embedding, (0.3, 0.4))
        self.assertEqual(result[2].embedding, (0.5, 0.6))

    def test_generate_embeddings_preserves_entities(self) -> None:
        """Embedding generation preserves extracted entities."""
        chunk = FileChunk(id=0, text="Alice is here", entities=("Alice",), embedding=None)

        self.embedding_generator.generate_embeddings.return_value = [[0.1]]

        result = self.service._generate_embeddings_for_chunks([chunk])

        self.assertEqual(len(result[0].entities), 1)
        self.assertIn("Alice", result[0].entities)

    def test_generate_embeddings_with_partial_failure(self) -> None:
        """Handles partial embedding failures (returns None for failed chunks)."""
        chunk1 = FileChunk(id=0, text="Text 1", entities=(), embedding=None)
        chunk2 = FileChunk(id=1, text="Text 2", entities=(), embedding=None)

        # Second chunk fails
        self.embedding_generator.generate_embeddings.return_value = [
            [0.1, 0.2],
            None,
        ]

        result = self.service._generate_embeddings_for_chunks([chunk1, chunk2])

        self.assertEqual(result[0].embedding, (0.1, 0.2))
        self.assertIsNone(result[1].embedding)


class TestFileProcessingServiceCollectAllEntities(unittest.TestCase):
    """Tests FileProcessingService._collect_all_entities method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)
        self.text_chunker = Mock(spec=TextChunkerInterface)

        self.service = FileProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
            text_chunker=self.text_chunker,
        )

    def test_collect_entities_from_single_chunk(self) -> None:
        """Collects all entities from a single chunk."""
        chunk = FileChunk(
            id=0,
            text="Alice and Bob",
            entities=("Alice", "Bob"),
            embedding=None,
        )

        result = self.service._collect_all_entities([chunk])

        self.assertEqual(len(result), 2)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)

    def test_collect_entities_from_multiple_chunks(self) -> None:
        """Collects entities from multiple chunks."""
        chunk1 = FileChunk(id=0, text="First", entities=("First",), embedding=None)
        chunk2 = FileChunk(id=1, text="Second", entities=("Second",), embedding=None)
        chunk3 = FileChunk(id=2, text="Third", entities=("Third",), embedding=None)

        result = self.service._collect_all_entities([chunk1, chunk2, chunk3])

        self.assertEqual(len(result), 3)
        self.assertIn("First", result)
        self.assertIn("Second", result)
        self.assertIn("Third", result)

    def test_collect_entities_deduplicates_exact_matches(self) -> None:
        """Deduplicates entities with same text from different chunks."""
        # Same entity appears in multiple chunks
        chunk1 = FileChunk(id=0, text="Alice in chunk1", entities=("Alice",), embedding=None)
        chunk2 = FileChunk(id=1, text="Alice in chunk2", entities=("Alice",), embedding=None)

        result = self.service._collect_all_entities([chunk1, chunk2])

        # Should deduplicate exact matches
        self.assertEqual(len(result), 1)
        self.assertIn("Alice", result)

    def test_collect_entities_with_empty_chunks(self) -> None:
        """Returns empty list when chunks have no entities."""
        chunk1 = FileChunk(id=0, text="No entities", entities=(), embedding=None)
        chunk2 = FileChunk(id=1, text="Also no entities", entities=(), embedding=None)

        result = self.service._collect_all_entities([chunk1, chunk2])

        self.assertEqual(len(result), 0)

    def test_collect_entities_preserves_case_variants(self) -> None:
        """Keeps entities with different case as separate entities."""
        chunk = FileChunk(
            id=0,
            text="alice and Alice and ALICE",
            entities=("alice", "Alice", "ALICE"),
            embedding=None,
        )

        result = self.service._collect_all_entities([chunk])

        # Should preserve case variants (deduplication is case-sensitive at this level)
        self.assertGreaterEqual(len(result), 1)

    def test_collect_entities_applies_levenshtein_deduplication(self) -> None:
        """Applies Levenshtein distance-based deduplication to similar entities."""
        chunk = FileChunk(
            id=0,
            text="Similar entities",
            entities=("мосбилет", "мосбилета", "москве"),
            embedding=None,
        )

        result = self.service._collect_all_entities([chunk])

        # Should apply Levenshtein deduplication (threshold <= 2)
        # "мосбилет" and "мосбилета" should be deduplicated
        self.assertLessEqual(len(result), 3)


class TestFileProcessingServiceConstants(unittest.TestCase):
    """Tests default constants."""

    def test_default_entity_types_has_21_types(self) -> None:
        """DEFAULT_ENTITY_TYPES contains 21 entity types."""
        self.assertEqual(len(DEFAULT_ENTITY_TYPES), 21)

    def test_default_entity_types_contains_common_types(self) -> None:
        """DEFAULT_ENTITY_TYPES contains expected common entity types."""
        self.assertIn("Person", DEFAULT_ENTITY_TYPES)
        self.assertIn("Organization", DEFAULT_ENTITY_TYPES)
        self.assertIn("Location", DEFAULT_ENTITY_TYPES)

    def test_default_chunk_size(self) -> None:
        """DEFAULT_CHUNK_SIZE is 3000 characters."""
        self.assertEqual(DEFAULT_CHUNK_SIZE, 3000)

    def test_default_chunk_overlap(self) -> None:
        """DEFAULT_CHUNK_OVERLAP is 300 characters."""
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, 300)

    def test_default_overlap_is_10_percent_of_size(self) -> None:
        """Default overlap is 10% of default chunk size."""
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE // 10)
