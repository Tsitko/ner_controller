"""Unit tests for EntityExtractorInterface."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface


class TestEntityExtractorInterface(unittest.TestCase):
    """Tests abstract behavior of EntityExtractorInterface."""

    def test_is_abstract(self) -> None:
        """EntityExtractorInterface cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            EntityExtractorInterface()  # type: ignore[abstract]
