"""End-to-end test for hallucination detection use case."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Mock gliner module and GLiNER class
# Check if gliner is already mocked, if so, enhance it; otherwise create new mock
if "gliner" in sys.modules:
    mock_gliner_module = sys.modules["gliner"]
    mock_gliner_class = mock_gliner_module.GLiNER
else:
    mock_gliner_module = MagicMock()
    mock_gliner_class = Mock()
    mock_gliner_module.GLiNER = mock_gliner_class
    sys.modules["gliner"] = mock_gliner_module

# Make GLiNER return a mock instance with predict_entities method that returns some entities
def mock_predict_entities(text, entity_types):
    """Mock predict_entities that returns some entity dicts based on text content."""
    predictions = []
    if "Alice" in text:
        predictions.append({"text": "Alice", "label": "PERSON", "start": 0, "end": 5})
    if "Paris" in text:
        idx = text.index("Paris")
        predictions.append({"text": "Paris", "label": "LOCATION", "start": idx, "end": idx + 5})
    if "OpenAI" in text:
        idx = text.index("OpenAI")
        predictions.append({"text": "OpenAI", "label": "ORGANIZATION", "start": idx, "end": idx + 6})
    if "Bob" in text:
        idx = text.index("Bob")
        predictions.append({"text": "Bob", "label": "PERSON", "start": idx, "end": idx + 3})
    if "Anthropic" in text:
        idx = text.index("Anthropic")
        predictions.append({"text": "Anthropic", "label": "ORGANIZATION", "start": idx, "end": idx + 9})
    return predictions

mock_gliner_instance = Mock()
mock_gliner_instance.predict_entities.side_effect = mock_predict_entities

# Mock the GLiNER.from_pretrained classmethod to return our mock instance
mock_gliner_class.from_pretrained = Mock(return_value=mock_gliner_instance)
if not hasattr(mock_gliner_class, 'return_value') or mock_gliner_class.return_value is None:
    mock_gliner_class.return_value = mock_gliner_instance

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
                    "text": entity,
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
                entity,
                str,
                msg=f"Entity must be str. Input text={text!r}, output={entity}",
            )
        # Entities are now just strings, no labels to check
        entity_texts = {entity.casefold() for entity in entities}
        self.assertTrue(
            {"alice", "paris", "openai"} & entity_texts,
            msg=(
                "NER should extract at least one of the known entities from input. "
                f"Extracted: {entity_texts}"
            ),
        )
