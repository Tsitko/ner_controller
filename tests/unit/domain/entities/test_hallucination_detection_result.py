"""Unit tests for HallucinationDetectionResult."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.hallucination_detection_result import (
    HallucinationDetectionResult,
)


class TestHallucinationDetectionResult(unittest.TestCase):
    """Tests HallucinationDetectionResult data container."""

    def test_fields(self) -> None:
        """HallucinationDetectionResult stores expected values."""
        result = HallucinationDetectionResult(
            potential_hallucinations=["Delta"],
            missing_entities=["Gamma"],
        )

        self.assertEqual(result.potential_hallucinations, ["Delta"])
        self.assertEqual(result.missing_entities, ["Gamma"])
