"""Unit tests for EmbeddingGeneratorInterface."""

import path_setup

path_setup.add_src_path()


import unittest
from abc import ABC

from ner_controller.domain.interfaces.embedding_generator_interface import (
    EmbeddingGeneratorInterface,
)


class DummyEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Dummy implementation for testing the interface."""

    def generate_embeddings(self, texts):
        """Dummy implementation that returns None for each text."""
        return tuple(None for _ in texts)


class TestEmbeddingGeneratorInterface(unittest.TestCase):
    """Tests EmbeddingGeneratorInterface contract."""

    def test_is_abstract_base_class(self) -> None:
        """EmbeddingGeneratorInterface is an ABC and cannot be instantiated."""
        self.assertTrue(issubclass(EmbeddingGeneratorInterface, ABC))

        with self.assertRaises(TypeError):
            EmbeddingGeneratorInterface()

    def test_requires_generate_embeddings_method(self) -> None:
        """EmbeddingGeneratorInterface requires generate_embeddings method."""
        generator = DummyEmbeddingGenerator()

        # Method exists
        self.assertTrue(hasattr(generator, "generate_embeddings"))
        self.assertTrue(callable(generator.generate_embeddings))

    def test_concrete_implementation_can_be_instantiated(self) -> None:
        """Concrete implementations of EmbeddingGeneratorInterface can be created."""
        generator = DummyEmbeddingGenerator()

        self.assertIsInstance(generator, EmbeddingGeneratorInterface)

    def test_generate_embeddings_accepts_sequence_of_strings(self) -> None:
        """generate_embeddings accepts a sequence of strings."""
        generator = DummyEmbeddingGenerator()

        texts = ["text1", "text2", "text3"]
        result = generator.generate_embeddings(texts)

        # Should not raise and should return a sequence
        self.assertIsInstance(result, (list, tuple))

    def test_generate_embeddings_returns_sequence_of_sequences_or_none(self) -> None:
        """generate_embeddings returns sequence of embeddings or None."""
        generator = DummyEmbeddingGenerator()

        result = generator.generate_embeddings(["text1", "text2"])

        # Result is a sequence
        self.assertIsInstance(result, (list, tuple))
        self.assertEqual(len(result), 2)

        # Each element is either None or a sequence of floats
        for item in result:
            self.assertTrue(item is None or all(isinstance(x, float) for x in item))

    def test_generate_embeddings_with_empty_sequence(self) -> None:
        """generate_embeddings handles empty input sequence."""
        generator = DummyEmbeddingGenerator()

        result = generator.generate_embeddings([])

        self.assertEqual(len(result), 0)

    def test_generate_embeddings_with_single_text(self) -> None:
        """generate_embeddings handles single text input."""
        generator = DummyEmbeddingGenerator()

        result = generator.generate_embeddings(["single text"])

        self.assertEqual(len(result), 1)

    def test_interface_defines_return_type_annotation(self) -> None:
        """Interface has proper type annotations for documentation."""
        # Check that the method signature includes type hints
        import inspect

        sig = inspect.signature(EmbeddingGeneratorInterface.generate_embeddings)
        self.assertIsNotNone(sig.return_annotation)


class TestEmbeddingGeneratorInterfaceDocumentation(unittest.TestCase):
    """Tests that interface contract is properly documented."""

    def test_method_has_docstring(self) -> None:
        """generate_embeddings method has descriptive docstring."""
        doc = EmbeddingGeneratorInterface.generate_embeddings.__doc__

        self.assertIsNotNone(doc)
        self.assertIn("embeddings", doc.lower())

    def test_docstring_describes_parameters(self) -> None:
        """Docstring describes the texts parameter."""
        doc = EmbeddingGeneratorInterface.generate_embeddings.__doc__

        self.assertIn("texts", doc.lower())

    def test_docstring_describes_return_value(self) -> None:
        """Docstring describes the return value."""
        doc = EmbeddingGeneratorInterface.generate_embeddings.__doc__

        self.assertIn("return", doc.lower())
        self.assertIn("embedding", doc.lower())

    def test_docstring_describes_exceptions(self) -> None:
        """Docstring documents exceptions that may be raised."""
        doc = EmbeddingGeneratorInterface.generate_embeddings.__doc__

        self.assertIn("raises", doc.lower())

    def test_interface_class_has_docstring(self) -> None:
        """Interface class has descriptive docstring."""
        doc = EmbeddingGeneratorInterface.__doc__

        self.assertIsNotNone(doc)
        self.assertIn("contract", doc.lower())
