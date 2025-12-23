"""Unit tests for ApplicationFactory."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock

from fastapi import FastAPI

from ner_controller.configs.app_config import AppConfig
from ner_controller.domain.services.hallucination_detection_service import (
    HallucinationDetectionService,
)
from ner_controller.main import ApplicationFactory


class TestApplicationFactory(unittest.TestCase):
    """Tests FastAPI app creation and wiring."""

    def test_create_app_builds_fastapi_application(self) -> None:
        """Factory creates a FastAPI app with expected metadata and routes."""
        service = Mock(spec=HallucinationDetectionService)
        factory = ApplicationFactory(AppConfig(), service=service)

        app = factory.create_app()

        self.assertIsInstance(app, FastAPI)
        self.assertEqual(app.title, "LLM Hallucination Checker")
        self.assertEqual(app.docs_url, "/docs")
        paths = {route.path for route in app.routes}
        self.assertIn("/hallucination/check", paths)
