"""Unit tests for EntityDiffResult."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.entity_diff_result import EntityDiffResult


class TestEntityDiffResult(unittest.TestCase):
    """Tests EntityDiffResult data container."""

    def test_fields(self) -> None:
        """EntityDiffResult stores expected values."""
        result = EntityDiffResult(
            potential_hallucinations=["Bob"],
            missing_entities=["Paris"],
        )

        self.assertEqual(result.potential_hallucinations, ["Bob"])
        self.assertEqual(result.missing_entities, ["Paris"])
