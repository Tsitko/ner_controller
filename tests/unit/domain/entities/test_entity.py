"""Unit tests for Entity."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.entity import Entity


class TestEntity(unittest.TestCase):
    """Tests Entity data container."""

    def test_fields(self) -> None:
        """Entity stores expected values."""
        entity = Entity(text="Alice", label="PERSON", start=0, end=5)

        self.assertEqual(entity.text, "Alice")
        self.assertEqual(entity.label, "PERSON")
        self.assertEqual(entity.start, 0)
        self.assertEqual(entity.end, 5)
