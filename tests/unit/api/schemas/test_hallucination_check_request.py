"""Unit tests for HallucinationCheckRequest."""

import path_setup

path_setup.add_src_path()


import unittest

from pydantic import ValidationError

from ner_controller.api.schemas.hallucination_check_request import HallucinationCheckRequest


class TestHallucinationCheckRequest(unittest.TestCase):
    """Tests input schema validation."""

    def test_valid_payload(self) -> None:
        """Schema accepts a valid request payload."""
        payload = {
            "request": "Prompt",
            "response": "Answer",
            "entity_types": ["PERSON"],
        }

        model = HallucinationCheckRequest(**payload)

        self.assertEqual(model.request, "Prompt")
        self.assertEqual(model.response, "Answer")
        self.assertEqual(model.entity_types, ["PERSON"])

    def test_accepts_legacy_entities_types_alias(self) -> None:
        """Schema accepts legacy entities_types alias."""
        payload = {
            "request": "Prompt",
            "response": "Answer",
            "entities_types": ["PERSON"],
        }

        model = HallucinationCheckRequest(**payload)

        self.assertEqual(model.entity_types, ["PERSON"])

    def test_missing_fields_raises_validation_error(self) -> None:
        """Schema rejects payloads with missing fields."""
        with self.assertRaises(ValidationError):
            HallucinationCheckRequest(request="Prompt")
