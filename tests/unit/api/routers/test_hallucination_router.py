"""Unit tests for HallucinationRouter."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock

from fastapi import APIRouter

from ner_controller.api.configs.hallucination_router_config import HallucinationRouterConfig
from ner_controller.api.routers.hallucination_router import HallucinationRouter
from ner_controller.api.schemas.hallucination_check_request import HallucinationCheckRequest
from ner_controller.domain.entities.hallucination_detection_result import (
    HallucinationDetectionResult,
)
from ner_controller.domain.services.hallucination_detection_service import (
    HallucinationDetectionService,
)


class TestHallucinationRouter(unittest.TestCase):
    """Tests router behavior and endpoint wiring."""

    def test_handle_hallucination_check_returns_response_schema(self) -> None:
        """Handler maps domain results to API response schema."""
        service = Mock(spec=HallucinationDetectionService)
        service.detect.return_value = HallucinationDetectionResult(
            potential_hallucinations=["Bob"],
            missing_entities=["Alice"],
        )
        router = HallucinationRouter(HallucinationRouterConfig(), service)
        request_model = HallucinationCheckRequest(
            request="Prompt",
            response="Answer",
            entity_types=["PERSON"],
        )

        response = router.handle_hallucination_check(request_model)

        self.assertEqual(response.potential_hallucinations, ["Bob"])
        self.assertEqual(response.missing_entities, ["Alice"])

    def test_create_router_registers_post_endpoint(self) -> None:
        """Router exposes the hallucination check endpoint."""
        service = Mock(spec=HallucinationDetectionService)
        router_builder = HallucinationRouter(HallucinationRouterConfig(), service)

        router = router_builder.create_router()

        self.assertIsInstance(router, APIRouter)
        paths = {route.path for route in router.routes}
        self.assertIn("/check", paths)
        methods = next(route.methods for route in router.routes if route.path == "/check")
        self.assertIn("POST", methods)
