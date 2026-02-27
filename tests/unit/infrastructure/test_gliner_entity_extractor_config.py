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
        self.assertEqual(config.base_model_name, "microsoft/mdeberta-v3-base")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.prediction_threshold, 0.2)
        self.assertTrue(config.local_files_only)
        self.assertTrue(config.offline_mode)
        self.assertIn("HF_HUB_OFFLINE", config.offline_env_vars)
