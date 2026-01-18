"""Unit tests for TextProcessRequest."""

import path_setup

path_setup.add_src_path()


import unittest

from pydantic import ValidationError

from ner_controller.api.schemas.text_process_request import TextProcessRequest


class TestTextProcessRequestValidation(unittest.TestCase):
    """Tests TextProcessRequest schema validation."""

    def test_valid_request_with_required_fields(self) -> None:
        """Accepts valid request with required field."""
        payload = {
            "text": "Alice works at OpenAI.",
        }

        request = TextProcessRequest(**payload)

        self.assertEqual(request.text, "Alice works at OpenAI.")

    def test_valid_request_with_entity_types(self) -> None:
        """Accepts valid request with entity_types."""
        payload = {
            "text": "Alice visited Paris.",
            "entity_types": ["Person", "Location"],
        }

        request = TextProcessRequest(**payload)

        self.assertEqual(request.text, "Alice visited Paris.")
        self.assertEqual(request.entity_types, ["Person", "Location"])

    def test_valid_request_with_legacy_entities_types(self) -> None:
        """Accepts valid request with legacy entities_types alias."""
        payload = {
            "text": "Alice visited Paris.",
            "entities_types": ["Person", "Location"],
        }

        request = TextProcessRequest(**payload)

        self.assertEqual(request.text, "Alice visited Paris.")
        self.assertEqual(request.entity_types, ["Person", "Location"])

    def test_missing_text_raises_validation_error(self) -> None:
        """Raises ValidationError when text field is missing."""
        payload = {
            "entity_types": ["Person"],
        }

        with self.assertRaises(ValidationError) as context:
            TextProcessRequest(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("text",) for e in errors))

    def test_default_entity_types_is_none(self) -> None:
        """Uses default value None for entity_types when not provided."""
        payload = {
            "text": "Sample text",
        }

        request = TextProcessRequest(**payload)

        self.assertIsNone(request.entity_types)

    def test_empty_text_raises_validation_error(self) -> None:
        """Raises ValidationError for empty text (min_length=1)."""
        payload = {
            "text": "",
        }

        with self.assertRaises(ValidationError) as context:
            TextProcessRequest(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("text",) for e in errors))

    def test_whitespace_only_text_raises_validation_error(self) -> None:
        """Raises ValidationError for whitespace-only text."""
        payload = {
            "text": "   \t\n  ",
        }

        with self.assertRaises(ValidationError) as context:
            TextProcessRequest(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("text",) for e in errors))


class TestTextProcessRequestFieldTypes(unittest.TestCase):
    """Tests TextProcessRequest field type constraints."""

    def test_text_field_is_string(self) -> None:
        """text field is string type."""
        payload = {"text": "Sample text"}

        request = TextProcessRequest(**payload)

        self.assertIsInstance(request.text, str)

    def test_text_field_type_validation(self) -> None:
        """Raises ValidationError for non-string text."""
        payload = {"text": 12345}

        with self.assertRaises(ValidationError):
            TextProcessRequest(**payload)

    def test_entity_types_field_is_optional_list(self) -> None:
        """entity_types field is optional list of strings."""
        payload = {
            "text": "Text",
            "entity_types": ["PERSON", "ORG", "LOC"],
        }

        request = TextProcessRequest(**payload)

        self.assertIsInstance(request.entity_types, list)

    def test_entity_types_type_validation(self) -> None:
        """Raises ValidationError for non-list entity_types."""
        payload = {
            "text": "Text",
            "entity_types": "PERSON",
        }

        with self.assertRaises(ValidationError):
            TextProcessRequest(**payload)

    def test_entity_types_list_of_strings(self) -> None:
        """Accepts list of strings for entity_types."""
        payload = {
            "text": "Text",
            "entity_types": ["Person", "Organization", "Location"],
        }

        request = TextProcessRequest(**payload)

        self.assertEqual(len(request.entity_types), 3)

    def test_entity_types_with_empty_list(self) -> None:
        """Accepts empty list for entity_types."""
        payload = {
            "text": "Text",
            "entity_types": [],
        }

        request = TextProcessRequest(**payload)

        self.assertEqual(request.entity_types, [])


class TestTextProcessRequestUnicode(unittest.TestCase):
    """Tests TextProcessRequest with Unicode content."""

    def test_unicode_text(self) -> None:
        """Accepts Unicode characters in text."""
        payload = {
            "text": "Алиса работает в Париже.",
        }

        request = TextProcessRequest(**payload)

        self.assertEqual(request.text, "Алиса работает в Париже.")

    def test_unicode_entity_types(self) -> None:
        """Accepts Unicode characters in entity_types."""
        payload = {
            "text": "Text",
            "entity_types": ["Персона", "Организация"],
        }

        request = TextProcessRequest(**payload)

        self.assertIn("Персона", request.entity_types)

    def test_mixed_unicode_text(self) -> None:
        """Accepts mixed Unicode text."""
        payload = {
            "text": "Alice works atКомпания in Москва.",
        }

        request = TextProcessRequest(**payload)

        self.assertIn("Компания", request.text)
        self.assertIn("Москва", request.text)


class TestTextProcessRequestLargeContent(unittest.TestCase):
    """Tests TextProcessRequest with large content."""

    def test_large_text_content(self) -> None:
        """Accepts large text content."""
        large_text = "Alice works at OpenAI. " * 10000  # ~280KB

        payload = {"text": large_text}

        request = TextProcessRequest(**payload)

        self.assertEqual(len(request.text), len(large_text))

    def test_many_entity_types(self) -> None:
        """Accepts large list of entity_types."""
        many_types = [f"TYPE_{i}" for i in range(1000)]

        payload = {
            "text": "Text",
            "entity_types": many_types,
        }

        request = TextProcessRequest(**payload)

        self.assertEqual(len(request.entity_types), 1000)


class TestTextProcessRequestSpecialCharacters(unittest.TestCase):
    """Tests TextProcessRequest with special characters."""

    def test_text_with_newlines(self) -> None:
        """Accepts text with newline characters."""
        payload = {
            "text": "Line 1\nLine 2\nLine 3",
        }

        request = TextProcessRequest(**payload)

        self.assertIn("\n", request.text)

    def test_text_with_tabs(self) -> None:
        """Accepts text with tab characters."""
        payload = {
            "text": "Column 1\tColumn 2\tColumn 3",
        }

        request = TextProcessRequest(**payload)

        self.assertIn("\t", request.text)

    def test_text_with_special_chars(self) -> None:
        """Accepts text with various special characters."""
        payload = {
            "text": "Email: test@example.com, Phone: +1-555-1234! (Visit http://example.com)",
        }

        request = TextProcessRequest(**payload)

        self.assertIn("@", request.text)
        self.assertIn("http://", request.text)

    def test_text_with_emoji(self) -> None:
        """Accepts text with emoji characters."""
        payload = {
            "text": "Hello! How are you?",
        }

        request = TextProcessRequest(**payload)

        self.assertIn("", request.text)

    def test_entity_types_with_special_chars(self) -> None:
        """Accepts entity_types with special characters."""
        payload = {
            "text": "Text",
            "entity_types": ["PERSON/PRO", "ORG-LLC", "LOCATION.City"],
        }

        request = TextProcessRequest(**payload)

        self.assertIn("PERSON/PRO", request.entity_types)


class TestTextProcessRequestStrictMode(unittest.TestCase):
    """Tests TextProcessRequest strict mode behavior."""

    def test_strict_mode_type_conversion(self) -> None:
        """Strict mode prevents automatic type conversion."""
        # With strict=True, "123" string should not be coerced to integer
        # for integer fields, but text is always string
        payload = {
            "text": 123,  # Integer instead of string
        }

        with self.assertRaises(ValidationError):
            TextProcessRequest(**payload)

    def test_extra_fields_forbidden_in_strict_mode(self) -> None:
        """Strict mode rejects extra fields."""
        payload = {
            "text": "Sample",
            "extra_field": "not_allowed",  # Extra field
        }

        with self.assertRaises(ValidationError):
            TextProcessRequest(**payload)


class TestTextProcessRequestEdgeCases(unittest.TestCase):
    """Tests TextProcessRequest edge cases."""

    def test_single_character_text(self) -> None:
        """Accepts single character text (min_length=1)."""
        payload = {"text": "A"}

        request = TextProcessRequest(**payload)

        self.assertEqual(request.text, "A")

    def test_text_with_only_whitespace_is_invalid(self) -> None:
        """Rejects text with only whitespace."""
        payload = {"text": " "}

        with self.assertRaises(ValidationError):
            TextProcessRequest(**payload)

    def test_entity_types_with_duplicates(self) -> None:
        """Accepts entity_types with duplicates."""
        payload = {
            "text": "Text",
            "entity_types": ["PERSON", "ORG", "PERSON"],
        }

        request = TextProcessRequest(**payload)

        # Schema accepts duplicates as-is
        self.assertEqual(request.entity_types, ["PERSON", "ORG", "PERSON"])

    def test_text_leading_trailing_whitespace(self) -> None:
        """Accepts text with leading/trailing whitespace."""
        payload = {"text": "  Valid text with spaces  "}

        request = TextProcessRequest(**payload)

        # Schema preserves whitespace (stripping happens in service layer)
        self.assertTrue(request.text.startswith(" "))
        self.assertTrue(request.text.endswith(" "))
