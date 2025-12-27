"""Unit tests for FileProcessingResult."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult


class TestFileProcessingResult(unittest.TestCase):
    """Tests FileProcessingResult aggregation container."""

    def test_initialize_with_all_fields(self) -> None:
        """FileProcessingResult stores all provided values correctly."""
        entity1 = "Alice"
        entity2 = "Bob"

        chunk1 = FileChunk(
            id=0,
            text="First chunk",
            entities=(entity1,),
            embedding=(0.1, 0.2),
        )

        chunk2 = FileChunk(
            id=1,
            text="Second chunk",
            entities=(entity2,),
            embedding=(0.3, 0.4),
        )

        result = FileProcessingResult(
            file_id="file-123",
            entities=(entity1, entity2),
            chunks=(chunk1, chunk2),
        )

        self.assertEqual(result.file_id, "file-123")
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.chunks), 2)
        self.assertIn(entity1, result.entities)
        self.assertIn(entity2, result.entities)
        self.assertIn(chunk1, result.chunks)
        self.assertIn(chunk2, result.chunks)

    def test_initialize_with_empty_entities(self) -> None:
        """FileProcessingResult accepts empty tuple when no entities found."""
        chunk = FileChunk(
            id=0,
            text="Text with no entities",
            entities=(),
            embedding=None,
        )

        result = FileProcessingResult(
            file_id="file-456",
            entities=(),
            chunks=(chunk,),
        )

        self.assertEqual(result.entities, ())
        self.assertEqual(len(result.chunks), 1)

    def test_initialize_with_empty_chunks(self) -> None:
        """FileProcessingResult accepts empty tuple when file has no content."""
        result = FileProcessingResult(
            file_id="empty-file",
            entities=(),
            chunks=(),
        )

        self.assertEqual(result.chunks, ())
        self.assertEqual(result.entities, ())

    def test_frozen_dataclass_is_immutable(self) -> None:
        """FileProcessingResult is frozen and cannot be modified."""
        result = FileProcessingResult(
            file_id="file-123",
            entities=(),
            chunks=(),
        )

        with self.assertRaises(Exception):  # FrozenInstanceError
            result.file_id = "file-456"

    def test_equality_based_on_all_fields(self) -> None:
        """Two FileProcessingResults with same values are equal."""
        entity = "Alice"
        chunk = FileChunk(
            id=0,
            text="Text",
            entities=(entity,),
            embedding=(0.1, 0.2),
        )

        result1 = FileProcessingResult(
            file_id="file-123",
            entities=(entity,),
            chunks=(chunk,),
        )

        result2 = FileProcessingResult(
            file_id="file-123",
            entities=(entity,),
            chunks=(chunk,),
        )

        self.assertEqual(result1, result2)

    def test_inequality_with_different_file_id(self) -> None:
        """FileProcessingResults differ when file_id differs."""
        entity = "Alice"
        chunk = FileChunk(
            id=0,
            text="Text",
            entities=(entity,),
            embedding=(0.1, 0.2),
        )

        result1 = FileProcessingResult(
            file_id="file-123",
            entities=(entity,),
            chunks=(chunk,),
        )

        result2 = FileProcessingResult(
            file_id="file-456",
            entities=(entity,),
            chunks=(chunk,),
        )

        self.assertNotEqual(result1, result2)

    def test_chunks_order_preserved(self) -> None:
        """FileProcessingResult preserves chunk order as provided."""
        chunk1 = FileChunk(id=0, text="First", entities=(), embedding=None)
        chunk2 = FileChunk(id=1, text="Second", entities=(), embedding=None)
        chunk3 = FileChunk(id=2, text="Third", entities=(), embedding=None)

        result = FileProcessingResult(
            file_id="file-123",
            entities=(),
            chunks=(chunk1, chunk2, chunk3),
        )

        self.assertEqual(result.chunks[0], chunk1)
        self.assertEqual(result.chunks[1], chunk2)
        self.assertEqual(result.chunks[2], chunk3)

    def test_entities_deduplicated_not_required(self) -> None:
        """FileProcessingResult stores entities as provided (deduplication is service responsibility)."""
        entity = "Alice"

        # Service may include duplicates if they appear in different chunks
        result = FileProcessingResult(
            file_id="file-123",
            entities=(entity, entity),  # Duplicate allowed at result level
            chunks=(),
        )

        self.assertEqual(len(result.entities), 2)

    def test_multiple_chunks_with_different_entity_counts(self) -> None:
        """FileProcessingResult handles chunks with varying numbers of entities."""
        entity1 = "Alice"
        entity2 = "Bob"
        entity3 = "Company"

        chunk1 = FileChunk(
            id=0,
            text="First chunk with one entity",
            entities=(entity1,),
            embedding=None,
        )

        chunk2 = FileChunk(
            id=1,
            text="Second chunk with two entities",
            entities=(entity2, entity3),
            embedding=None,
        )

        chunk3 = FileChunk(
            id=2,
            text="Third chunk with no entities",
            entities=(),
            embedding=None,
        )

        result = FileProcessingResult(
            file_id="file-123",
            entities=(entity1, entity2, entity3),
            chunks=(chunk1, chunk2, chunk3),
        )

        self.assertEqual(len(result.chunks), 3)
        self.assertEqual(len(result.chunks[0].entities), 1)
        self.assertEqual(len(result.chunks[1].entities), 2)
        self.assertEqual(len(result.chunks[2].entities), 0)

    def test_uuid_as_file_id(self) -> None:
        """FileProcessingResult accepts UUID string as file_id."""
        import uuid

        file_id = str(uuid.uuid4())

        result = FileProcessingResult(
            file_id=file_id,
            entities=(),
            chunks=(),
        )

        self.assertEqual(result.file_id, file_id)

    def test_chunks_may_have_none_embeddings(self) -> None:
        """FileProcessingResult accepts chunks where some embeddings failed to generate."""
        entity1 = "Alice"
        entity2 = "Bob"

        chunk1 = FileChunk(
            id=0,
            text="Chunk with embedding",
            entities=(entity1,),
            embedding=(0.1, 0.2),
        )

        chunk2 = FileChunk(
            id=1,
            text="Chunk without embedding",
            entities=(entity2,),
            embedding=None,
        )

        result = FileProcessingResult(
            file_id="file-123",
            entities=(entity1, entity2),
            chunks=(chunk1, chunk2),
        )

        self.assertIsNotNone(result.chunks[0].embedding)
        self.assertIsNone(result.chunks[1].embedding)

    def test_large_number_of_chunks(self) -> None:
        """FileProcessingResult handles large numbers of chunks."""
        chunks = tuple(
            FileChunk(
                id=i,
                text=f"Chunk {i}",
                entities=(),
                embedding=None,
            )
            for i in range(1000)
        )

        result = FileProcessingResult(
            file_id="large-file",
            entities=(),
            chunks=chunks,
        )

        self.assertEqual(len(result.chunks), 1000)

    def test_large_number_of_entities(self) -> None:
        """FileProcessingResult handles large numbers of entities."""
        entities = tuple(
            f"Entity{i}"
            for i in range(1000)
        )

        result = FileProcessingResult(
            file_id="entity-heavy-file",
            entities=entities,
            chunks=(),
        )

        self.assertEqual(len(result.entities), 1000)

    def test_empty_string_file_id(self) -> None:
        """FileProcessingResult technically accepts empty string as file_id (though unusual)."""
        result = FileProcessingResult(
            file_id="",
            entities=(),
            chunks=(),
        )

        self.assertEqual(result.file_id, "")
