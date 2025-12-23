"""Unit tests for HallucinationCheckResponse."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.api.schemas.hallucination_check_response import HallucinationCheckResponse


class TestHallucinationCheckResponse(unittest.TestCase):
    """Tests output schema behavior."""

    def test_payload_fields(self) -> None:
        """Schema stores output lists."""
        model = HallucinationCheckResponse(
            potential_hallucinations=["Bob"],
            missing_entities=["Alice"],
        )

        self.assertEqual(model.potential_hallucinations, ["Bob"])
        self.assertEqual(model.missing_entities, ["Alice"])
