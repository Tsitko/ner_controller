"""Unit tests for AppConfig."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.configs.app_config import AppConfig


class TestAppConfig(unittest.TestCase):
    """Tests default values for AppConfig."""

    def test_defaults(self) -> None:
        """AppConfig provides expected defaults."""
        config = AppConfig()

        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 1304)
        self.assertEqual(config.title, "LLM Hallucination Checker")
        self.assertEqual(config.docs_url, "/docs")
