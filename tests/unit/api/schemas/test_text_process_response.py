"""Unit tests for TextProcessResponse."""

import path_setup

path_setup.add_src_path()


import unittest

from pydantic import ValidationError

from ner_controller.api.schemas.text_process_response import TextProcessResponse


class TestTextProcessResponseValidation(unittest.TestCase):
    """Tests TextProcessResponse schema validation."""

    def test_valid_response_with_all_fields(self) -> None:
        """Accepts valid response with all required fields."""
        payload = {
            "text": "Alice works at OpenAI.",
            "entities": ["Alice", "OpenAI"],
            "embedding": [0.1, 0.2, 0.3, 0.4],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(response.text, "Alice works at OpenAI.")
        self.assertEqual(len(response.entities), 2)
        self.assertEqual(len(response.embedding), 4)

    def test_missing_text_raises_validation_error(self) -> None:
        """Raises ValidationError when text field is missing."""
        payload = {
            "entities": [],
            "embedding": [],
        }

        with self.assertRaises(ValidationError) as context:
            TextProcessResponse(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("text",) for e in errors))

    def test_missing_embedding_raises_validation_error(self) -> None:
        """Raises ValidationError when embedding field is missing."""
        payload = {
            "text": "Sample",
            "entities": [],
        }

        with self.assertRaises(ValidationError) as context:
            TextProcessResponse(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("embedding",) for e in errors))

    def test_default_entities_is_empty_list(self) -> None:
        """Uses default empty list for entities when not provided."""
        payload = {
            "text": "Sample text",
            "embedding": [0.1],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(response.entities, [])

    def test_response_with_empty_entities(self) -> None:
        """Accepts response with empty entities list."""
        payload = {
            "text": "No entities found",
            "entities": [],
            "embedding": [0.1, 0.2],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.entities), 0)

    def test_response_with_empty_embedding(self) -> None:
        """Accepts response with empty embedding vector."""
        payload = {
            "text": "No embedding",
            "entities": [],
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.embedding), 0)


class TestTextProcessResponseFieldTypes(unittest.TestCase):
    """Tests TextProcessResponse field type constraints."""

    def test_text_field_is_string(self) -> None:
        """text field is string type."""
        payload = {
            "text": "Sample text",
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIsInstance(response.text, str)

    def test_text_field_type_validation(self) -> None:
        """Raises ValidationError for non-string text."""
        payload = {
            "text": 12345,
            "embedding": [],
        }

        with self.assertRaises(ValidationError):
            TextProcessResponse(**payload)

    def test_entities_field_is_sequence(self) -> None:
        """entities field is sequence type."""
        payload = {
            "text": "Text",
            "entities": ["Alice", "Bob"],
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIsInstance(response.entities, list)

    def test_entities_field_type_validation(self) -> None:
        """Raises ValidationError for non-sequence entities."""
        payload = {
            "text": "Text",
            "entities": "not-a-list",
            "embedding": [],
        }

        with self.assertRaises(ValidationError):
            TextProcessResponse(**payload)

    def test_embedding_field_is_sequence(self) -> None:
        """embedding field is sequence type."""
        payload = {
            "text": "Text",
            "embedding": [0.1, 0.2, 0.3],
        }

        response = TextProcessResponse(**payload)

        self.assertIsInstance(response.embedding, list)

    def test_embedding_field_type_validation(self) -> None:
        """Raises ValidationError for non-sequence embedding."""
        payload = {
            "text": "Text",
            "entities": [],
            "embedding": "not-a-list",
        }

        with self.assertRaises(ValidationError):
            TextProcessResponse(**payload)


class TestTextProcessResponseEntities(unittest.TestCase):
    """Tests TextProcessResponse entities field."""

    def test_entities_with_multiple_items(self) -> None:
        """Accepts multiple entities in response."""
        payload = {
            "text": "Text",
            "entities": ["Alice", "Bob", "Charlie", "OpenAI"],
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.entities), 4)

    def test_entities_are_strings(self) -> None:
        """Entities field contains strings."""
        payload = {
            "text": "Text",
            "entities": ["Alice", "Bob", "Paris"],
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIsInstance(response.entities, list)
        for entity in response.entities:
            self.assertIsInstance(entity, str)

    def test_entities_with_unicode(self) -> None:
        """Accepts Unicode entities."""
        payload = {
            "text": "Text",
            "entities": ["Алиса", "Париж", "Москва"],
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.entities), 3)
        self.assertIn("Алиса", response.entities)

    def test_entities_with_duplicates(self) -> None:
        """Accepts entities with duplicates (deduplication happens in service)."""
        payload = {
            "text": "Text",
            "entities": ["Alice", "Bob", "Alice", "Charlie", "Bob"],
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        # Schema accepts duplicates as-is
        self.assertEqual(len(response.entities), 5)

    def test_entities_with_special_characters(self) -> None:
        """Accepts entities with special characters."""
        payload = {
            "text": "Text",
            "entities": ["C++", "C#", ".NET", "Node.js"],
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIn("C++", response.entities)
        self.assertIn(".NET", response.entities)


class TestTextProcessResponseEmbedding(unittest.TestCase):
    """Tests TextProcessResponse embedding field."""

    def test_embedding_with_float_values(self) -> None:
        """Accepts list of floats for embedding."""
        payload = {
            "text": "Text",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.embedding), 5)

    def test_embedding_with_negative_values(self) -> None:
        """Accepts negative float values in embedding."""
        payload = {
            "text": "Text",
            "embedding": [-0.5, -0.2, 0.1, 0.3, -0.8],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(response.embedding[0], -0.5)
        self.assertEqual(response.embedding[4], -0.8)

    def test_embedding_with_large_dimensions(self) -> None:
        """Accepts large embedding vectors."""
        large_embedding = [float(i) * 0.01 for i in range(10000)]
        payload = {
            "text": "Text",
            "embedding": large_embedding,
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.embedding), 10000)

    def test_embedding_with_integer_values(self) -> None:
        """Accepts integer values (coerced to floats)."""
        payload = {
            "text": "Text",
            "embedding": [1, 2, 3, 4],
        }

        response = TextProcessResponse(**payload)

        # Pydantic should coerce integers to floats
        self.assertIsInstance(response.embedding[0], (int, float))

    def test_embedding_very_small_values(self) -> None:
        """Accepts very small float values."""
        payload = {
            "text": "Text",
            "embedding": [0.0001, -0.0001, 1e-10, -1e-10],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.embedding), 4)

    def test_embedding_very_large_values(self) -> None:
        """Accepts very large float values."""
        payload = {
            "text": "Text",
            "embedding": [1e10, -1e10, 1.5e20],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.embedding), 3)


class TestTextProcessResponseText(unittest.TestCase):
    """Tests TextProcessResponse text field."""

    def test_text_with_newlines(self) -> None:
        """Accepts text with newline characters."""
        payload = {
            "text": "Line 1\nLine 2\nLine 3",
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIn("\n", response.text)

    def test_text_with_tabs(self) -> None:
        """Accepts text with tab characters."""
        payload = {
            "text": "Column 1\tColumn 2\tColumn 3",
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIn("\t", response.text)

    def test_text_with_special_chars(self) -> None:
        """Accepts text with special characters."""
        payload = {
            "text": "Email: test@example.com, URL: http://example.com",
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIn("@", response.text)
        self.assertIn("http://", response.text)

    def test_text_unicode(self) -> None:
        """Accepts Unicode text."""
        payload = {
            "text": "Алиса работает в Париже. Hello!",
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertIn("Алиса", response.text)
        self.assertIn("Hello", response.text)

    def test_text_empty_string(self) -> None:
        """Accepts empty string for text (edge case)."""
        payload = {
            "text": "",
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(response.text, "")

    def test_text_very_long(self) -> None:
        """Accepts very long text."""
        long_text = "Alice works at OpenAI. " * 10000

        payload = {
            "text": long_text,
            "embedding": [],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.text), len(long_text))


class TestTextProcessResponseSerialization(unittest.TestCase):
    """Tests TextProcessResponse serialization."""

    def test_response_serializes_to_dict(self) -> None:
        """Response can be serialized to dictionary."""
        payload = {
            "text": "Sample text",
            "entities": ["Entity1", "Entity2"],
            "embedding": [0.1, 0.2],
        }

        response = TextProcessResponse(**payload)
        response_dict = response.model_dump()

        self.assertIsInstance(response_dict, dict)
        self.assertIn("text", response_dict)
        self.assertIn("entities", response_dict)
        self.assertIn("embedding", response_dict)

    def test_response_serializes_to_json(self) -> None:
        """Response can be serialized to JSON."""
        payload = {
            "text": "Sample text",
            "entities": ["Entity1"],
            "embedding": [0.1],
        }

        response = TextProcessResponse(**payload)
        import json

        json_str = response.model_dump_json()

        self.assertIsInstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["text"], "Sample text")

    def test_model_dump_excludes_none_by_default(self) -> None:
        """model_dump excludes None values for fields with defaults."""
        payload = {
            "text": "Text",
            "embedding": [],
        }

        response = TextProcessResponse(**payload)
        response_dict = response.model_dump()

        # entities has default_factory=list, should be present even when empty
        self.assertIn("entities", response_dict)


class TestTextProcessResponseComplete(unittest.TestCase):
    """Tests complete TextProcessResponse scenarios."""

    def test_realistic_response(self) -> None:
        """Accepts realistic response with all fields populated."""
        payload = {
            "text": "Alice visited Paris and met with OpenAI researchers.",
            "entities": ["Alice", "Paris", "OpenAI", "researchers"],
            "embedding": [
                0.1234,
                -0.5678,
                0.9012,
                -0.3456,
                0.7890,
            ],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(len(response.entities), 4)
        self.assertEqual(len(response.embedding), 5)
        self.assertIn("Alice", response.entities)

    def test_minimal_response(self) -> None:
        """Accepts minimal response with only required fields."""
        payload = {
            "text": "Minimal text",
            "embedding": [0.1],
        }

        response = TextProcessResponse(**payload)

        self.assertEqual(response.text, "Minimal text")
        self.assertEqual(response.entities, [])
        self.assertEqual(len(response.embedding), 1)

    def test_response_with_unicode_everywhere(self) -> None:
        """Accepts Unicode in all fields."""
        payload = {
            "text": "Алиса работает в Париже",
            "entities": ["Алиса", "Париж"],
            "embedding": [0.1, 0.2, 0.3],
        }

        response = TextProcessResponse(**payload)

        self.assertIn("Алиса", response.text)
        self.assertIn("Париж", response.entities)
