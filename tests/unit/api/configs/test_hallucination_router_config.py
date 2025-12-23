"""Unit tests for HallucinationRouterConfig."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.api.configs.hallucination_router_config import HallucinationRouterConfig


class TestHallucinationRouterConfig(unittest.TestCase):
    """Tests default values for HallucinationRouterConfig."""

    def test_defaults(self) -> None:
        """HallucinationRouterConfig provides expected defaults."""
        config = HallucinationRouterConfig()

        self.assertEqual(config.prefix, "/hallucination")
        self.assertEqual(config.tags, ("hallucination",))
