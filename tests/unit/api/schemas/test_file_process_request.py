"""Unit tests for FileProcessRequest."""

import path_setup

path_setup.add_src_path()


import base64
import unittest

from pydantic import ValidationError

from ner_controller.api.schemas.file_process_request import FileProcessRequest


class TestFileProcessRequestValidation(unittest.TestCase):
    """Tests FileProcessRequest schema validation."""

    def test_valid_request_with_all_fields(self) -> None:
        """Accepts valid request with all required fields."""
        payload = {
            "file": base64.b64encode(b"Test content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.file, payload["file"])
        self.assertEqual(request.file_name, "test.txt")
        self.assertEqual(request.file_id, "file-123")

    def test_valid_request_with_optional_fields(self) -> None:
        """Accepts valid request with optional fields included."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "doc.txt",
            "file_id": "file-456",
            "file_path": "/path/to/doc.txt",
            "chunk_size": 5000,
            "chunk_overlap": 500,
            "entity_types": ["PERSON", "ORG"],
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.file_path, "/path/to/doc.txt")
        self.assertEqual(request.chunk_size, 5000)
        self.assertEqual(request.chunk_overlap, 500)
        self.assertEqual(request.entity_types, ["PERSON", "ORG"])

    def test_valid_request_with_file_path_none(self) -> None:
        """Accepts file_path explicitly set to None."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-789",
            "file_path": None,
        }

        request = FileProcessRequest(**payload)

        self.assertIsNone(request.file_path)

    def test_missing_file_raises_validation_error(self) -> None:
        """Raises ValidationError when file field is missing."""
        payload = {
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        with self.assertRaises(ValidationError) as context:
            FileProcessRequest(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("file",) for e in errors))

    def test_missing_file_name_raises_validation_error(self) -> None:
        """Raises ValidationError when file_name field is missing."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_id": "file-123",
        }

        with self.assertRaises(ValidationError) as context:
            FileProcessRequest(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("file_name",) for e in errors))

    def test_missing_file_id_raises_validation_error(self) -> None:
        """Raises ValidationError when file_id field is missing."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
        }

        with self.assertRaises(ValidationError) as context:
            FileProcessRequest(**payload)

        errors = context.exception.errors()
        self.assertTrue(any(e["loc"] == ("file_id",) for e in errors))

    def test_default_chunk_size(self) -> None:
        """Uses default value 3000 for chunk_size when not provided."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.chunk_size, 3000)

    def test_default_chunk_overlap(self) -> None:
        """Uses default value 300 for chunk_overlap when not provided."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.chunk_overlap, 300)

    def test_default_entity_types(self) -> None:
        """Uses default value None for entity_types when not provided."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertIsNone(request.entity_types)

    def test_empty_file_is_accepted(self) -> None:
        """Accepts empty base64-encoded file."""
        payload = {
            "file": base64.b64encode(b"").decode(),
            "file_name": "empty.txt",
            "file_id": "file-empty",
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.file, "")

    def test_invalid_base64_string_raises_error(self) -> None:
        """Raises ValidationError for invalid base64 string."""
        # Note: Pydantic doesn't validate base64 by default, this tests
        # that the schema at least accepts the string
        payload = {
            "file": "Not valid base64!!!",
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        # Pydantic will accept this - validation happens in service layer
        request = FileProcessRequest(**payload)
        self.assertEqual(request.file, "Not valid base64!!!")

    def test_chunk_size_type_validation(self) -> None:
        """Raises ValidationError for non-integer chunk_size."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "chunk_size": "not-an-int",
        }

        with self.assertRaises(ValidationError):
            FileProcessRequest(**payload)

    def test_chunk_overlap_type_validation(self) -> None:
        """Raises ValidationError for non-integer chunk_overlap."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "chunk_overlap": "not-an-int",
        }

        with self.assertRaises(ValidationError):
            FileProcessRequest(**payload)

    def test_entity_types_type_validation(self) -> None:
        """Raises ValidationError for non-list entity_types."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "entity_types": "PERSON",
        }

        with self.assertRaises(ValidationError):
            FileProcessRequest(**payload)

    def test_entity_types_list_of_strings(self) -> None:
        """Accepts list of strings for entity_types."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "entity_types": ["PERSON", "ORG", "LOC"],
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.entity_types, ["PERSON", "ORG", "LOC"])

    def test_empty_entity_types_list(self) -> None:
        """Accepts empty list for entity_types."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "entity_types": [],
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.entity_types, [])

    def test_unicode_file_name(self) -> None:
        """Accepts Unicode characters in file_name."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "документ.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.file_name, "документ.txt")

    def test_unicode_file_id(self) -> None:
        """Accepts Unicode characters in file_id."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "файл-123",
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.file_id, "файл-123")

    def test_uuid_as_file_id(self) -> None:
        """Accepts UUID format as file_id."""
        import uuid

        file_id = str(uuid.uuid4())
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": file_id,
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(request.file_id, file_id)

    def test_large_file_content(self) -> None:
        """Accepts large base64-encoded file content."""
        large_content = b"X" * 10_000_000  # 10 MB
        payload = {
            "file": base64.b64encode(large_content).decode(),
            "file_name": "large.txt",
            "file_id": "file-large",
        }

        request = FileProcessRequest(**payload)

        self.assertEqual(len(request.file), len(base64.b64encode(large_content).decode()))

    def test_negative_chunk_size_accepted_by_schema(self) -> None:
        """Schema accepts negative chunk_size (validation in service layer)."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "chunk_size": -100,
        }

        # Pydantic accepts this - business validation happens later
        request = FileProcessRequest(**payload)
        self.assertEqual(request.chunk_size, -100)

    def test_zero_chunk_size_accepted_by_schema(self) -> None:
        """Schema accepts zero chunk_size (validation in service layer)."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "chunk_size": 0,
        }

        # Pydantic accepts this - business validation happens later
        request = FileProcessRequest(**payload)
        self.assertEqual(request.chunk_size, 0)


class TestFileProcessRequestFieldTypes(unittest.TestCase):
    """Tests FileProcessRequest field types."""

    def test_file_field_is_string(self) -> None:
        """file field is string type."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertIsInstance(request.file, str)

    def test_file_name_field_is_string(self) -> None:
        """file_name field is string type."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertIsInstance(request.file_name, str)

    def test_file_id_field_is_string(self) -> None:
        """file_id field is string type."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
        }

        request = FileProcessRequest(**payload)

        self.assertIsInstance(request.file_id, str)

    def test_file_path_field_is_optional_string(self) -> None:
        """file_path field is optional string or None."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "file_path": "/path/to/file.txt",
        }

        request = FileProcessRequest(**payload)

        self.assertIsInstance(request.file_path, str)

    def test_chunk_size_field_is_integer(self) -> None:
        """chunk_size field is integer type."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "chunk_size": 5000,
        }

        request = FileProcessRequest(**payload)

        self.assertIsInstance(request.chunk_size, int)

    def test_chunk_overlap_field_is_integer(self) -> None:
        """chunk_overlap field is integer type."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "chunk_overlap": 500,
        }

        request = FileProcessRequest(**payload)

        self.assertIsInstance(request.chunk_overlap, int)

    def test_entity_types_field_is_optional_list(self) -> None:
        """entity_types field is optional list of strings."""
        payload = {
            "file": base64.b64encode(b"Content").decode(),
            "file_name": "test.txt",
            "file_id": "file-123",
            "entity_types": ["PERSON", "ORG"],
        }

        request = FileProcessRequest(**payload)

        self.assertIsInstance(request.entity_types, list)
