"""Unit tests for EntityDiffCalculator."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.entity import Entity
from ner_controller.domain.services.entity_diff_calculator import EntityDiffCalculator


class TestEntityDiffCalculator(unittest.TestCase):
    """Tests entity diff calculation behavior."""

    def setUp(self) -> None:
        """Set up shared calculator instance."""
        self._calculator = EntityDiffCalculator()

    def test_calculate_detects_missing_and_hallucinated_entities(self) -> None:
        """Calculator returns response-only and request-only entities."""
        request_entities = [
            Entity(text="Alice", label="PERSON", start=0, end=5),
            Entity(text="Paris", label="LOCATION", start=10, end=15),
        ]
        response_entities = [
            Entity(text="Bob", label="PERSON", start=0, end=3),
            Entity(text="Paris", label="LOCATION", start=10, end=15),
            Entity(text="Zoo", label="ORGANIZATION", start=20, end=23),
            Entity(text="Bob", label="PERSON", start=30, end=33),
        ]

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, ["Bob", "Zoo"])
        self.assertEqual(result.missing_entities, ["Alice"])

    def test_calculate_is_case_insensitive_and_strips_whitespace(self) -> None:
        """Calculator ignores case and surrounding whitespace when matching entities."""
        request_entities = [Entity(text=" Alice ", label="PERSON", start=0, end=7)]
        response_entities = [Entity(text="alice", label="PERSON", start=0, end=5)]

        result = self._calculator.calculate(request_entities, response_entities)

        self.assertEqual(result.potential_hallucinations, [])
        self.assertEqual(result.missing_entities, [])
