"""End-to-end test for hallucination detection use case."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastapi.routing import APIRoute

from ner_controller.api.schemas.hallucination_check_request import HallucinationCheckRequest
from ner_controller.configs.app_config import AppConfig
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator
from ner_controller.domain.services.hallucination_detection_service import (
    HallucinationDetectionService,
)
from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)
from ner_controller.infrastructure.ner.gliner_entity_extractor import GlinerEntityExtractor
from ner_controller.main import ApplicationFactory


class TestHallucinationE2E(unittest.TestCase):
    """E2E test covering request-to-response flow with real GLiNER."""

    def test_full_flow_returns_expected_result(self) -> None:
        """App processes a request and returns hallucination output."""
        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())
        service = HallucinationDetectionService(extractor, EntityDiffCalculator())
        app = ApplicationFactory(AppConfig(), service=service).create_app()

        request_text = "Alice went to Paris and met OpenAI researchers."
        response_text = "Bob visited Paris and joined Anthropic."
        entity_types = ["PERSON", "LOCATION", "ORGANIZATION"]

        expected = service.detect(request_text, response_text, entity_types)

        route = next(
            route
            for route in app.routes
            if isinstance(route, APIRoute) and route.path == "/hallucination/check"
        )
        payload = HallucinationCheckRequest(
            request=request_text,
            response=response_text,
            entities_types=entity_types,
        )

        response = route.endpoint(payload)

        print(
            "E2E input:",
            {
                "request": request_text,
                "response": response_text,
                "entities_types": entity_types,
            },
        )
        print(
            "E2E output:",
            {
                "potential_hallucinations": response.potential_hallucinations,
                "missing_entities": response.missing_entities,
            },
        )

        self.assertIsInstance(
            response.potential_hallucinations,
            list,
            msg=(
                "Expected list output for potential_hallucinations. "
                f"Input: request={request_text!r}, response={response_text!r}, "
                f"entities_types={entity_types}. Output={response.potential_hallucinations!r}"
            ),
        )
        self.assertIsInstance(
            response.missing_entities,
            list,
            msg=(
                "Expected list output for missing_entities. "
                f"Input: request={request_text!r}, response={response_text!r}, "
                f"entities_types={entity_types}. Output={response.missing_entities!r}"
            ),
        )
        self.assertEqual(
            response.potential_hallucinations,
            expected.potential_hallucinations,
            msg=(
                "Input request/response texts should yield identical hallucination results "
                "between the service and HTTP handler."
            ),
        )
        self.assertEqual(
            response.missing_entities,
            expected.missing_entities,
            msg=(
                "Input request/response texts should yield identical missing-entity results "
                "between the service and HTTP handler."
            ),
        )

    def test_gliner_extractor_returns_expected_entity_types(self) -> None:
        """GLiNER extracts entities for the requested labels."""
        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())
        text = "Alice went to Paris to meet OpenAI."
        entity_types = ["PERSON", "LOCATION", "ORGANIZATION"]

        entities = extractor.extract(text, entity_types)

        print(
            "NER input:",
            {"text": text, "entities_types": entity_types},
        )
        print(
            "NER output:",
            [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                }
                for entity in entities
            ],
        )

        self.assertTrue(
            entities,
            msg=(
                "Expected entities from NER. "
                f"Input text={text!r}, entity_types={entity_types}"
            ),
        )
        for entity in entities:
            self.assertIsInstance(
                entity.text,
                str,
                msg=f"Entity text must be str. Input text={text!r}, output={entity}",
            )
            self.assertIsInstance(
                entity.label,
                str,
                msg=f"Entity label must be str. Input text={text!r}, output={entity}",
            )
            self.assertIsInstance(
                entity.start,
                int,
                msg=f"Entity start must be int. Input text={text!r}, output={entity}",
            )
            self.assertIsInstance(
                entity.end,
                int,
                msg=f"Entity end must be int. Input text={text!r}, output={entity}",
            )
        labels = {entity.label for entity in entities}
        self.assertTrue(
            labels.issubset(set(entity_types)),
            msg=f"NER labels {labels} should be within {entity_types}",
        )
        entity_texts = {entity.text.casefold() for entity in entities}
        self.assertTrue(
            {"alice", "paris", "openai"} & entity_texts,
            msg=(
                "NER should extract at least one of the known entities from input. "
                f"Extracted: {entity_texts}"
            ),
        )
