"""Unit tests for GlinerEntityExtractorConfig."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)


class TestGlinerEntityExtractorConfig(unittest.TestCase):
    """Tests default values for GlinerEntityExtractorConfig."""

    def test_defaults(self) -> None:
        """GlinerEntityExtractorConfig provides expected defaults."""
        config = GlinerEntityExtractorConfig()

        self.assertEqual(config.model_name, "urchade/gliner_multi-v2.1")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 8)
