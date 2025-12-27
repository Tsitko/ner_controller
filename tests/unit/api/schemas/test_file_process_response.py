"""Unit tests for FileProcessResponse, ChunkSchema."""

import path_setup

path_setup.add_src_path()


import unittest

from pydantic import ValidationError

from ner_controller.api.schemas.file_process_response import (
    ChunkSchema,
    FileProcessResponse,
)


class TestChunkSchema(unittest.TestCase):
    """Tests ChunkSchema validation."""

    def test_valid_chunk_with_all_fields(self) -> None:
        """Accepts valid chunk with all required fields."""
        payload = {
            "id": 0,
            "text": "Sample chunk text",
            "entities": ["Alice", "Bob"],
            "embedding": [0.1, 0.2, 0.3],
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(chunk.id, 0)
        self.assertEqual(chunk.text, "Sample chunk text")
        self.assertEqual(len(chunk.entities), 2)
        self.assertEqual(chunk.embedding, [0.1, 0.2, 0.3])

    def test_missing_id_raises_validation_error(self) -> None:
        """Raises ValidationError when id field is missing."""
        payload = {
            "text": "Sample",
            "entities": [],
            "embedding": None,
        }

        with self.assertRaises(ValidationError) as context:
            ChunkSchema(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("id",) for e in errors))

    def test_missing_text_raises_validation_error(self) -> None:
        """Raises ValidationError when text field is missing."""
        payload = {
            "id": 0,
            "entities": [],
            "embedding": None,
        }

        with self.assertRaises(ValidationError) as context:
            ChunkSchema(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("text",) for e in errors))

    def test_default_entities_is_empty_list(self) -> None:
        """Uses default empty list for entities when not provided."""
        payload = {
            "id": 0,
            "text": "Sample",
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(chunk.entities, [])

    def test_default_embedding_is_none(self) -> None:
        """Uses default None for embedding when not provided."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": [],
        }

        chunk = ChunkSchema(**payload)

        self.assertIsNone(chunk.embedding)

    def test_entities_with_multiple_items(self) -> None:
        """Accepts multiple entities in entities list."""
        payload = {
            "id": 0,
            "text": "Alice and Bob",
            "entities": ["Alice", "Bob", "Charlie"],
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(len(chunk.entities), 3)

    def test_entities_field_type_validation(self) -> None:
        """Raises ValidationError for non-list entities."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": "not-a-list",
        }

        with self.assertRaises(ValidationError):
            ChunkSchema(**payload)

    def test_embedding_field_type_validation(self) -> None:
        """Raises ValidationError for non-list embedding."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": [],
            "embedding": "not-a-list",
        }

        with self.assertRaises(ValidationError):
            ChunkSchema(**payload)

    def test_embedding_with_none_value(self) -> None:
        """Accepts None for embedding field."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": [],
            "embedding": None,
        }

        chunk = ChunkSchema(**payload)

        self.assertIsNone(chunk.embedding)

    def test_embedding_with_float_values(self) -> None:
        """Accepts list of floats for embedding."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": [],
            "embedding": [0.1, 0.2, 0.3, 0.4],
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(chunk.embedding, [0.1, 0.2, 0.3, 0.4])

    def test_embedding_with_negative_values(self) -> None:
        """Accepts negative float values in embedding."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": [],
            "embedding": [-0.5, -0.2, 0.1, 0.3],
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(chunk.embedding, [-0.5, -0.2, 0.1, 0.3])

    def test_large_embedding_dimensions(self) -> None:
        """Accepts large embedding vectors."""
        large_embedding = [float(i) * 0.01 for i in range(10000)]
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": [],
            "embedding": large_embedding,
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(len(chunk.embedding), 10000)

    def test_empty_text_is_accepted(self) -> None:
        """Accepts empty string for text field."""
        payload = {
            "id": 0,
            "text": "",
            "entities": [],
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(chunk.text, "")

    def test_negative_id_is_accepted(self) -> None:
        """Accepts negative ID (though unusual)."""
        payload = {
            "id": -1,
            "text": "Sample",
            "entities": [],
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(chunk.id, -1)

    def test_id_field_type_validation(self) -> None:
        """Raises ValidationError for non-integer id."""
        payload = {
            "id": "0",
            "text": "Sample",
            "entities": [],
        }

        with self.assertRaises(ValidationError):
            ChunkSchema(**payload)

    def test_text_field_type_validation(self) -> None:
        """Raises ValidationError for non-string text."""
        payload = {
            "id": 0,
            "text": 123,
            "entities": [],
        }

        with self.assertRaises(ValidationError):
            ChunkSchema(**payload)

    def test_entities_are_strings(self) -> None:
        """Entities field contains strings."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": ["Alice", "Bob", "Paris"],
        }

        chunk = ChunkSchema(**payload)

        self.assertIsInstance(chunk.entities, list)
        for entity in chunk.entities:
            self.assertIsInstance(entity, str)

    def test_entities_with_unicode(self) -> None:
        """Accepts Unicode entities."""
        payload = {
            "id": 0,
            "text": "Sample",
            "entities": ["Алиса", "Париж", "Москва"],
        }

        chunk = ChunkSchema(**payload)

        self.assertEqual(len(chunk.entities), 3)
        self.assertIn("Алиса", chunk.entities)


class TestFileProcessResponse(unittest.TestCase):
    """Tests FileProcessResponse validation."""

    def test_valid_response_with_all_fields(self) -> None:
        """Accepts valid response with all required fields."""
        payload = {
            "file_id": "file-123",
            "entities": ["Alice", "Bob", "OpenAI"],
            "chanks": [
                {
                    "id": 0,
                    "text": "Sample text",
                    "entities": ["Alice"],
                    "embedding": [0.1, 0.2],
                }
            ],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(response.file_id, "file-123")
        self.assertEqual(len(response.entities), 3)
        self.assertEqual(len(response.chanks), 1)

    def test_missing_file_id_raises_validation_error(self) -> None:
        """Raises ValidationError when file_id field is missing."""
        payload = {
            "entities": [],
            "chanks": [],
        }

        with self.assertRaises(ValidationError) as context:
            FileProcessResponse(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("file_id",) for e in errors))

    def test_default_entities_is_empty_list(self) -> None:
        """Uses default empty list for entities when not provided."""
        payload = {
            "file_id": "file-123",
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(response.entities, [])

    def test_default_chanks_is_empty_list(self) -> None:
        """Uses default empty list for chanks when not provided."""
        payload = {
            "file_id": "file-123",
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(response.chanks, [])

    def test_response_with_empty_entities_and_chunks(self) -> None:
        """Accepts response with no entities and no chunks."""
        payload = {
            "file_id": "empty-file",
            "entities": [],
            "chanks": [],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(response.file_id, "empty-file")
        self.assertEqual(len(response.entities), 0)
        self.assertEqual(len(response.chanks), 0)

    def test_response_with_multiple_entities(self) -> None:
        """Accepts multiple entities in response."""
        payload = {
            "file_id": "file-123",
            "entities": ["Alice", "Bob", "Company", "Paris"],
            "chanks": [],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(len(response.entities), 4)

    def test_response_with_multiple_chunks(self) -> None:
        """Accepts multiple chunks in response."""
        payload = {
            "file_id": "file-123",
            "entities": [],
            "chanks": [
                {"id": 0, "text": "First", "entities": [], "embedding": None},
                {"id": 1, "text": "Second", "entities": [], "embedding": None},
                {"id": 2, "text": "Third", "entities": [], "embedding": None},
            ],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(len(response.chanks), 3)

    def test_chunks_with_entities_and_embeddings(self) -> None:
        """Accepts chunks with both entities and embeddings."""
        payload = {
            "file_id": "file-123",
            "entities": ["Alice", "Bob"],
            "chanks": [
                {
                    "id": 0,
                    "text": "Alice is here",
                    "entities": ["Alice"],
                    "embedding": [0.1, 0.2, 0.3],
                }
            ],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(len(response.chanks[0].entities), 1)
        self.assertIsNotNone(response.chanks[0].embedding)

    def test_chunks_mixed_embeddings_some_none(self) -> None:
        """Accepts chunks where some have embeddings and some don't."""
        payload = {
            "file_id": "file-123",
            "entities": [],
            "chanks": [
                {"id": 0, "text": "First", "entities": [], "embedding": [0.1]},
                {"id": 1, "text": "Second", "entities": [], "embedding": None},
                {"id": 2, "text": "Third", "entities": [], "embedding": [0.3]},
            ],
        }

        response = FileProcessResponse(**payload)

        self.assertIsNotNone(response.chanks[0].embedding)
        self.assertIsNone(response.chanks[1].embedding)
        self.assertIsNotNone(response.chanks[2].embedding)

    def test_file_id_field_type_validation(self) -> None:
        """Raises ValidationError for non-string file_id."""
        payload = {
            "file_id": 123,
            "entities": [],
            "chanks": [],
        }

        with self.assertRaises(ValidationError):
            FileProcessResponse(**payload)

    def test_entities_field_type_validation(self) -> None:
        """Raises ValidationError for non-list entities."""
        payload = {
            "file_id": "file-123",
            "entities": "not-a-list",
            "chanks": [],
        }

        with self.assertRaises(ValidationError):
            FileProcessResponse(**payload)

    def test_chanks_field_type_validation(self) -> None:
        """Raises ValidationError for non-list chanks."""
        payload = {
            "file_id": "file-123",
            "entities": [],
            "chanks": "not-a-list",
        }

        with self.assertRaises(ValidationError):
            FileProcessResponse(**payload)

    def test_unicode_file_id(self) -> None:
        """Accepts Unicode characters in file_id."""
        payload = {
            "file_id": "файл-123",
            "entities": [],
            "chanks": [],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(response.file_id, "файл-123")

    def test_uuid_as_file_id(self) -> None:
        """Accepts UUID format as file_id."""
        import uuid

        file_id = str(uuid.uuid4())
        payload = {
            "file_id": file_id,
            "entities": [],
            "chanks": [],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(response.file_id, file_id)

    def test_entities_are_unique(self) -> None:
        """Entities list should contain unique strings (semantic validation)."""
        # Schema accepts duplicates, deduplication happens at service layer
        payload = {
            "file_id": "file-123",
            "entities": ["Alice", "Bob", "Alice", "Charlie", "Bob"],
            "chanks": [],
        }

        response = FileProcessResponse(**payload)

        # Schema accepts the list as-is
        self.assertEqual(len(response.entities), 5)

    def test_entities_with_unicode(self) -> None:
        """Accepts Unicode entities."""
        payload = {
            "file_id": "file-123",
            "entities": ["Алиса", "Боб", "Париж"],
            "chanks": [],
        }

        response = FileProcessResponse(**payload)

        self.assertEqual(len(response.entities), 3)
        self.assertIn("Алиса", response.entities)
