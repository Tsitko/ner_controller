"""Unit tests for FileChunk."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.file_chunk import FileChunk


class TestFileChunk(unittest.TestCase):
    """Tests FileChunk data container."""

    def test_initialize_with_all_fields(self) -> None:
        """FileChunk stores all provided values correctly."""
        entity1 = "Alice"
        entity2 = "Bob"
        embedding = (0.1, 0.2, 0.3)

        chunk = FileChunk(
            id=1,
            text="Sample text with entities.",
            entities=(entity1, entity2),
            embedding=embedding,
        )

        self.assertEqual(chunk.id, 1)
        self.assertEqual(chunk.text, "Sample text with entities.")
        self.assertEqual(len(chunk.entities), 2)
        self.assertEqual(chunk.entities[0], entity1)
        self.assertEqual(chunk.entities[1], entity2)
        self.assertEqual(chunk.embedding, embedding)

    def test_initialize_with_empty_entities(self) -> None:
        """FileChunk accepts empty tuple for entities."""
        chunk = FileChunk(
            id=0,
            text="Sample text.",
            entities=(),
            embedding=None,
        )

        self.assertEqual(chunk.entities, ())
        self.assertIsNone(chunk.embedding)

    def test_initialize_with_none_embedding(self) -> None:
        """FileChunk accepts None for embedding when not yet generated."""
        entity = "Alice"

        chunk = FileChunk(
            id=1,
            text="Text with entity.",
            entities=(entity,),
            embedding=None,
        )

        self.assertIsNone(chunk.embedding)
        self.assertEqual(len(chunk.entities), 1)

    def test_frozen_dataclass_is_immutable(self) -> None:
        """FileChunk is frozen and cannot be modified after creation."""
        chunk = FileChunk(
            id=1,
            text="Original text",
            entities=(),
            embedding=None,
        )

        with self.assertRaises(Exception):  # FrozenInstanceError
            chunk.id = 2

        with self.assertRaises(Exception):
            chunk.text = "Modified text"

    def test_equality_based_on_all_fields(self) -> None:
        """Two FileChunks with same values are equal."""
        entity = "Alice"
        embedding = (0.1, 0.2)

        chunk1 = FileChunk(
            id=1,
            text="Same text",
            entities=(entity,),
            embedding=embedding,
        )

        chunk2 = FileChunk(
            id=1,
            text="Same text",
            entities=(entity,),
            embedding=embedding,
        )

        self.assertEqual(chunk1, chunk2)

    def test_inequality_with_different_fields(self) -> None:
        """FileChunks differ when any field differs."""
        entity = "Alice"

        chunk1 = FileChunk(
            id=1,
            text="Text one",
            entities=(entity,),
            embedding=None,
        )

        chunk2 = FileChunk(
            id=2,
            text="Text one",
            entities=(entity,),
            embedding=None,
        )

        self.assertNotEqual(chunk1, chunk2)

    def test_entities_order_preserved(self) -> None:
        """FileChunk preserves entity order as provided."""
        entity1 = "First"
        entity2 = "Second"
        entity3 = "Third"

        chunk = FileChunk(
            id=1,
            text="Text",
            entities=(entity1, entity2, entity3),
            embedding=None,
        )

        self.assertEqual(chunk.entities[0], entity1)
        self.assertEqual(chunk.entities[1], entity2)
        self.assertEqual(chunk.entities[2], entity3)

    def test_embedding_stored_as_tuple(self) -> None:
        """Embedding is stored as tuple for immutability."""
        embedding_list = [0.1, 0.2, 0.3]
        embedding_tuple = tuple(embedding_list)

        chunk = FileChunk(
            id=1,
            text="Text",
            entities=(),
            embedding=embedding_tuple,
        )

        self.assertIsInstance(chunk.embedding, tuple)
        self.assertEqual(chunk.embedding, (0.1, 0.2, 0.3))

    def test_empty_text_is_valid(self) -> None:
        """FileChunk accepts empty string as text."""
        chunk = FileChunk(
            id=0,
            text="",
            entities=(),
            embedding=None,
        )

        self.assertEqual(chunk.text, "")

    def test_negative_id_is_accepted(self) -> None:
        """FileChunk accepts negative IDs (though unusual)."""
        chunk = FileChunk(
            id=-1,
            text="Text",
            entities=(),
            embedding=None,
        )

        self.assertEqual(chunk.id, -1)

    def test_zero_id_is_accepted(self) -> None:
        """FileChunk accepts zero as valid ID."""
        chunk = FileChunk(
            id=0,
            text="Text",
            entities=(),
            embedding=None,
        )

        self.assertEqual(chunk.id, 0)

    def test_large_embedding_dimensions(self) -> None:
        """FileChunk handles large embedding vectors."""
        large_embedding = tuple(float(i) for i in range(10000))

        chunk = FileChunk(
            id=1,
            text="Text",
            entities=(),
            embedding=large_embedding,
        )

        self.assertEqual(len(chunk.embedding), 10000)

    def test_multiple_entities_in_chunk(self) -> None:
        """FileChunk stores multiple entities correctly."""
        entities = (
            "Alice",
            "Company Inc",
            "New York",
        )

        chunk = FileChunk(
            id=1,
            text="Alice works at Company Inc in New York",
            entities=entities,
            embedding=None,
        )

        self.assertEqual(len(chunk.entities), 3)
        self.assertIn(entities[0], chunk.entities)
        self.assertIn(entities[1], chunk.entities)
        self.assertIn(entities[2], chunk.entities)

    def test_embedding_with_negative_values(self) -> None:
        """FileChunk accepts embeddings with negative float values."""
        embedding = (-0.5, -0.2, 0.1, 0.3)

        chunk = FileChunk(
            id=1,
            text="Text",
            entities=(),
            embedding=embedding,
        )

        self.assertEqual(chunk.embedding, (-0.5, -0.2, 0.1, 0.3))

    def test_embedding_with_zero_values(self) -> None:
        """FileChunk accepts embeddings with zero values."""
        embedding = (0.0, 0.0, 0.0)

        chunk = FileChunk(
            id=1,
            text="Text",
            entities=(),
            embedding=embedding,
        )

        self.assertEqual(chunk.embedding, (0.0, 0.0, 0.0))
