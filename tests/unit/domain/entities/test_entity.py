"""Unit tests for Entity type alias."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.entity import Entity


class TestEntity(unittest.TestCase):
    """Tests Entity type alias (str)."""

    def test_entity_is_string_type(self) -> None:
        """Entity is a type alias for str."""
        entity: Entity = "Alice"

        self.assertIsInstance(entity, str)
        self.assertEqual(entity, "Alice")

    def test_entity_with_unicode(self) -> None:
        """Entity accepts Unicode characters."""
        entity: Entity = "Алиса"

        self.assertIsInstance(entity, str)
        self.assertEqual(entity, "Алиса")

    def test_entity_with_special_characters(self) -> None:
        """Entity accepts special characters."""
        entity: Entity = "San Francisco"

        self.assertIsInstance(entity, str)
        self.assertEqual(entity, "San Francisco")

    def test_entity_with_empty_string(self) -> None:
        """Entity accepts empty string (though semantically invalid)."""
        entity: Entity = ""

        self.assertIsInstance(entity, str)
        self.assertEqual(entity, "")

    def test_entity_type_checking(self) -> None:
        """Entity type annotation works correctly."""
        def process_entity(e: Entity) -> str:
            return e.upper()

        entity: Entity = "OpenAI"
        result = process_entity(entity)

        self.assertEqual(result, "OPENAI")
