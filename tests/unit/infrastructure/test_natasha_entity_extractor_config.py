"""Unit tests for NatashaEntityExtractorConfig."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.infrastructure.ner.configs.natasha_entity_extractor_config import (
    NatashaEntityExtractorConfig,
)


class TestNatashaEntityExtractorConfig(unittest.TestCase):
    """Tests default values for NatashaEntityExtractorConfig."""

    def test_defaults(self) -> None:
        """Config provides expected default paths."""
        config = NatashaEntityExtractorConfig()

        self.assertIn("navec_news_v1_1B_250K_300d_100q.tar", str(config.navec_model_path()))
        self.assertIn("slovnet_ner_news_v1.tar", str(config.ner_model_path()))
