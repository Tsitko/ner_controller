"""Unit tests for TextChunkerInterface."""

import path_setup

path_setup.add_src_path()


import unittest
from abc import ABC

from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.interfaces.text_chunker_interface import (
    TextChunkerInterface,
)


class DummyTextChunker(TextChunkerInterface):
    """Dummy implementation for testing the interface."""

    def split_text(self, text, chunk_size, chunk_overlap, start_id=0):
        """Dummy implementation that returns empty list."""
        return []


class TestTextChunkerInterface(unittest.TestCase):
    """Tests TextChunkerInterface contract."""

    def test_is_abstract_base_class(self) -> None:
        """TextChunkerInterface is an ABC and cannot be instantiated."""
        self.assertTrue(issubclass(TextChunkerInterface, ABC))

        with self.assertRaises(TypeError):
            TextChunkerInterface()

    def test_requires_split_text_method(self) -> None:
        """TextChunkerInterface requires split_text method."""
        chunker = DummyTextChunker()

        # Method exists
        self.assertTrue(hasattr(chunker, "split_text"))
        self.assertTrue(callable(chunker.split_text))

    def test_concrete_implementation_can_be_instantiated(self) -> None:
        """Concrete implementations of TextChunkerInterface can be created."""
        chunker = DummyTextChunker()

        self.assertIsInstance(chunker, TextChunkerInterface)

    def test_split_text_accepts_required_parameters(self) -> None:
        """split_text accepts text, chunk_size, and chunk_overlap parameters."""
        chunker = DummyTextChunker()

        # Should not raise
        result = chunker.split_text(
            text="Sample text",
            chunk_size=100,
            chunk_overlap=10,
        )

        self.assertIsInstance(result, list)

    def test_split_text_accepts_optional_start_id(self) -> None:
        """split_text accepts optional start_id parameter with default."""
        chunker = DummyTextChunker()

        # Without start_id
        result1 = chunker.split_text("text", 100, 10)
        self.assertIsInstance(result1, list)

        # With start_id
        result2 = chunker.split_text("text", 100, 10, start_id=5)
        self.assertIsInstance(result2, list)

    def test_split_text_returns_list_of_file_chunks(self) -> None:
        """split_text returns list of FileChunk objects."""
        chunker = DummyTextChunker()

        result = chunker.split_text("text", 100, 10)

        self.assertIsInstance(result, list)
        # All items must be FileChunk instances when properly implemented
        for item in result:
            if item is not None:
                self.assertIsInstance(item, FileChunk)

    def test_split_text_with_empty_string(self) -> None:
        """split_text handles empty string input."""
        chunker = DummyTextChunker()

        result = chunker.split_text("", 100, 10)

        self.assertIsInstance(result, list)

    def test_split_text_with_large_chunk_size(self) -> None:
        """split_text accepts chunk_size larger than text length."""
        chunker = DummyTextChunker()

        result = chunker.split_text(
            text="Short text",
            chunk_size=10000,
            chunk_overlap=0,
        )

        self.assertIsInstance(result, list)

    def test_split_text_with_zero_overlap(self) -> None:
        """split_text accepts zero overlap."""
        chunker = DummyTextChunker()

        result = chunker.split_text(
            text="Text without overlap",
            chunk_size=100,
            chunk_overlap=0,
        )

        self.assertIsInstance(result, list)

    def test_interface_defines_parameter_validation_requirements(self) -> None:
        """Interface documentation specifies parameter validation rules."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIsNotNone(doc)
        # Document that validation is required
        self.assertIn("ValueError", doc)

    def test_interface_defines_validation_rules(self) -> None:
        """Interface specifies chunk_size > 0 and chunk_overlap < chunk_size."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIsNotNone(doc)
        # Should mention the validation rules
        self.assertIn("chunk_size", doc)
        self.assertIn("chunk_overlap", doc)

    def test_interface_defines_return_value_structure(self) -> None:
        """Interface specifies that chunks have sequential IDs and empty entities/embeddings."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIsNotNone(doc)
        # Should mention chunk structure
        self.assertIn("FileChunk", doc)


class TestTextChunkerInterfaceDocumentation(unittest.TestCase):
    """Tests that interface contract is properly documented."""

    def test_method_has_docstring(self) -> None:
        """split_text method has descriptive docstring."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIsNotNone(doc)
        self.assertIn("chunk", doc.lower())

    def test_docstring_describes_text_parameter(self) -> None:
        """Docstring describes the text parameter."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIn("text", doc.lower())

    def test_docstring_describes_chunk_size_parameter(self) -> None:
        """Docstring describes the chunk_size parameter."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIn("chunk_size", doc.lower())

    def test_docstring_describes_chunk_overlap_parameter(self) -> None:
        """Docstring describes the chunk_overlap parameter."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIn("chunk_overlap", doc.lower())

    def test_docstring_describes_start_id_parameter(self) -> None:
        """Docstring describes the start_id parameter."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIn("start_id", doc.lower())

    def test_docstring_describes_return_value(self) -> None:
        """Docstring describes the return value."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIn("return", doc.lower())
        self.assertIn("FileChunk", doc)

    def test_docstring_describes_exceptions(self) -> None:
        """Docstring documents exceptions that may be raised."""
        doc = TextChunkerInterface.split_text.__doc__

        self.assertIn("raises", doc.lower())
        self.assertIn("ValueError", doc)

    def test_interface_class_has_docstring(self) -> None:
        """Interface class has descriptive docstring."""
        doc = TextChunkerInterface.__doc__

        self.assertIsNotNone(doc)
        self.assertIn("contract", doc.lower())


class TestTextChunkerInterfaceValidationRequirements(unittest.TestCase):
    """Tests validation requirements specified by the interface."""

    def test_chunk_size_must_be_positive(self) -> None:
        """Interface specifies chunk_size must be > 0."""
        # This is a contract requirement documented in the interface
        # Concrete implementations must enforce this
        doc = TextChunkerInterface.split_text.__doc__
        self.assertIn("chunk_size <= 0", doc)

    def test_chunk_overlap_must_be_non_negative(self) -> None:
        """Interface specifies chunk_overlap must be >= 0."""
        doc = TextChunkerInterface.split_text.__doc__
        self.assertIn("chunk_overlap < 0", doc)

    def test_chunk_overlap_must_be_less_than_chunk_size(self) -> None:
        """Interface specifies chunk_overlap must be < chunk_size."""
        doc = TextChunkerInterface.split_text.__doc__
        self.assertIn("chunk_overlap >= chunk_size", doc)
