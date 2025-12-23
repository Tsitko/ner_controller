"""Integration tests for API router with real service."""

import path_setup

path_setup.add_src_path()


import unittest
from typing import Sequence

from fastapi import FastAPI
from fastapi.routing import APIRoute

from ner_controller.api.configs.hallucination_router_config import HallucinationRouterConfig
from ner_controller.api.routers.hallucination_router import HallucinationRouter
from ner_controller.api.schemas.hallucination_check_request import HallucinationCheckRequest
from ner_controller.domain.entities.entity import Entity
from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator
from ner_controller.domain.services.hallucination_detection_service import (
    HallucinationDetectionService,
)


class StaticEntityExtractor(EntityExtractorInterface):
    """Simple extractor that returns preconfigured entities by text."""

    def __init__(self, mapping: dict[str, Sequence[Entity]]) -> None:
        """Initialize extractor with text-to-entities mapping."""
        self._mapping = mapping

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[Entity]:
        """Return entities for the given text, ignoring entity types."""
        return list(self._mapping.get(text, []))


class TestHallucinationRouterIntegration(unittest.TestCase):
    """Integration tests for router wiring with FastAPI."""

    def test_router_handles_request(self) -> None:
        """POST endpoint returns expected response model."""
        request_entities = [
            Entity(text="Alice", label="PERSON", start=0, end=5),
        ]
        response_entities = [
            Entity(text="Bob", label="PERSON", start=0, end=3),
        ]
        extractor = StaticEntityExtractor({"Prompt": request_entities, "Answer": response_entities})
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())
        router_config = HallucinationRouterConfig()
        router = HallucinationRouter(router_config, service).create_router()

        app = FastAPI()
        app.include_router(router, prefix=router_config.prefix)

        route = next(
            route
            for route in app.routes
            if isinstance(route, APIRoute) and route.path == "/hallucination/check"
        )
        payload = HallucinationCheckRequest(
            request="Prompt",
            response="Answer",
            entities_types=["PERSON"],
        )

        response = route.endpoint(payload)

        self.assertEqual(response.potential_hallucinations, ["Bob"])
        self.assertEqual(response.missing_entities, ["Alice"])
