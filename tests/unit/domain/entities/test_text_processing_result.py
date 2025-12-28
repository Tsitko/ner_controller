"""Unit tests for TextProcessingResult."""

import path_setup

path_setup.add_src_path()


import unittest


class TestTextProcessingResultImmutability(unittest.TestCase):
    """Tests TextProcessingResult frozen dataclass immutability."""

    def test_frozen_dataclass_prevents_field_reassignment(self) -> None:
        """Cannot reassign fields after instantiation."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="Sample text",
            entities=("Alice", "Bob"),
            embedding=(0.1, 0.2, 0.3),
        )

        with self.assertRaises(Exception):  # FrozenInstanceError or similar
            result.text = "Modified text"

    def test_frozen_dataclass_prevents_field_addition(self) -> None:
        """Cannot add new fields after instantiation."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="Sample text",
            entities=(),
            embedding=(),
        )

        with self.assertRaises(Exception):  # FrozenInstanceError or similar
            result.new_field = "value"


class TestTextProcessingResultFieldTypes(unittest.TestCase):
    """Tests TextProcessingResult field type constraints."""

    def test_text_field_accepts_string(self) -> None:
        """text field accepts and stores string values."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="Test text",
            entities=(),
            embedding=(),
        )

        self.assertIsInstance(result.text, str)
        self.assertEqual(result.text, "Test text")

    def test_entities_field_accepts_tuple_of_strings(self) -> None:
        """entities field accepts and stores tuple of strings."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        entities = ("Alice", "Bob", "OpenAI")
        result = TextProcessingResult(
            text="Text",
            entities=entities,
            embedding=(),
        )

        self.assertIsInstance(result.entities, tuple)
        self.assertEqual(len(result.entities), 3)
        self.assertIn("Alice", result.entities)

    def test_embedding_field_accepts_tuple_of_floats(self) -> None:
        """embedding field accepts and stores tuple of floats."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        embedding = (0.1, 0.2, 0.3, 0.4)
        result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=embedding,
        )

        self.assertIsInstance(result.embedding, tuple)
        self.assertEqual(len(result.embedding), 4)
        self.assertEqual(result.embedding[0], 0.1)

    def test_empty_tuple_for_entities(self) -> None:
        """entities field accepts empty tuple."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="No entities",
            entities=(),
            embedding=(),
        )

        self.assertEqual(len(result.entities), 0)

    def test_empty_tuple_for_embedding(self) -> None:
        """embedding field accepts empty tuple."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="No embedding",
            entities=(),
            embedding=(),
        )

        self.assertEqual(len(result.embedding), 0)

    def test_unicode_in_text_field(self) -> None:
        """text field accepts Unicode characters."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        unicode_text = "Привет, мир! Hello world!"
        result = TextProcessingResult(
            text=unicode_text,
            entities=(),
            embedding=(),
        )

        self.assertEqual(result.text, unicode_text)

    def test_unicode_entities(self) -> None:
        """entities field accepts Unicode strings."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        entities = ("Алиса", "Париж", "Москва")
        result = TextProcessingResult(
            text="Text",
            entities=entities,
            embedding=(),
        )

        self.assertIn("Алиса", result.entities)

    def test_negative_floats_in_embedding(self) -> None:
        """embedding field accepts negative float values."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        embedding = (-0.5, -0.2, 0.1, 0.3)
        result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=embedding,
        )

        self.assertEqual(result.embedding[0], -0.5)

    def test_large_embedding_dimensions(self) -> None:
        """embedding field accepts large dimension vectors."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        embedding = tuple(float(i) * 0.01 for i in range(10000))
        result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=embedding,
        )

        self.assertEqual(len(result.embedding), 10000)


class TestTextProcessingResultEquality(unittest.TestCase):
    """Tests TextProcessingResult equality and hashing."""

    def test_two_results_with_same_values_are_equal(self) -> None:
        """Two instances with same field values are equal."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result1 = TextProcessingResult(
            text="Same text",
            entities=("Entity1",),
            embedding=(0.1,),
        )

        result2 = TextProcessingResult(
            text="Same text",
            entities=("Entity1",),
            embedding=(0.1,),
        )

        self.assertEqual(result1, result2)

    def test_two_results_with_different_values_not_equal(self) -> None:
        """Two instances with different field values are not equal."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result1 = TextProcessingResult(
            text="Text 1",
            entities=("Entity1",),
            embedding=(0.1,),
        )

        result2 = TextProcessingResult(
            text="Text 2",
            entities=("Entity1",),
            embedding=(0.1,),
        )

        self.assertNotEqual(result1, result2)

    def test_result_not_equal_to_other_type(self) -> None:
        """TextProcessingResult is not equal to other types."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=(),
        )

        self.assertNotEqual(result, "Text")
        self.assertNotEqual(result, {"text": "Text"})

    def test_result_with_different_entities_order(self) -> None:
        """Results with different entity order are not equal."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result1 = TextProcessingResult(
            text="Text",
            entities=("Entity1", "Entity2"),
            embedding=(),
        )

        result2 = TextProcessingResult(
            text="Text",
            entities=("Entity2", "Entity1"),
            embedding=(),
        )

        # Order matters in tuples
        self.assertNotEqual(result1, result2)


class TestTextProcessingResultStringRepresentation(unittest.TestCase):
    """Tests TextProcessingResult string representation."""

    def test_repr_contains_class_name(self) -> None:
        """String representation contains class name."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=(),
        )

        repr_str = repr(result)
        self.assertIn("TextProcessingResult", repr_str)

    def test_repr_contains_field_values(self) -> None:
        """String representation contains field values."""
        from ner_controller.domain.entities.text_processing_result import (
            TextProcessingResult,
        )

        result = TextProcessingResult(
            text="Sample",
            entities=("Entity1",),
            embedding=(0.1, 0.2),
        )

        repr_str = repr(result)
        self.assertIn("Sample", repr_str)
        self.assertIn("Entity1", repr_str)
