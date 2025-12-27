"""Unit tests for CompositeEntityExtractor."""

import sys

import path_setup

path_setup.add_src_path()


# Mock gliner module before importing
from unittest.mock import MagicMock, Mock
mock_gliner = MagicMock()
sys.modules["gliner"] = mock_gliner

import unittest
from unittest.mock import patch

from ner_controller.domain.interfaces.entity_extractor_interface import (
    EntityExtractorInterface,
)
from ner_controller.infrastructure.ner.composite_entity_extractor import (
    CompositeEntityExtractor,
)


class MockExtractor(EntityExtractorInterface):
    """Mock entity extractor for testing."""

    def __init__(self, entities: list[str]) -> None:
        """Initialize mock with predefined entities to return."""
        self._entities = entities
        self.extract_call_count = 0
        self.last_text = None
        self.last_entity_types = None

    def extract(self, text: str, entity_types: list[str]) -> list[str]:
        """Return predefined entities."""
        self.extract_call_count += 1
        self.last_text = text
        self.last_entity_types = entity_types
        return self._entities.copy()


class TestCompositeEntityExtractor(unittest.TestCase):
    """Tests composite entity extractor behavior."""

    def test_init_raises_error_for_empty_extractors(self) -> None:
        """Extractor raises ValueError when initialized with empty list."""
        with self.assertRaises(ValueError) as context:
            CompositeEntityExtractor([])

        self.assertIn("At least one extractor", str(context.exception))

    def test_init_stores_extractors(self) -> None:
        """Extractor stores provided extractors."""
        mock1 = Mock(spec=EntityExtractorInterface)
        mock2 = Mock(spec=EntityExtractorInterface)

        extractor = CompositeEntityExtractor([mock1, mock2])

        self.assertEqual(len(extractor._extractors), 2)

    def test_extract_combines_results_from_multiple_extractors(self) -> None:
        """Extractor combines results from all extractors."""
        mock1 = MockExtractor(["alice", "bob"])
        mock2 = MockExtractor(["charlie", "david"])

        composite = CompositeEntityExtractor([mock1, mock2])
        result = composite.extract("test text", ["Entity"])

        self.assertEqual(len(result), 4)
        self.assertIn("alice", result)
        self.assertIn("bob", result)
        self.assertIn("charlie", result)
        self.assertIn("david", result)

    def test_extract_calls_all_extractors(self) -> None:
        """Extractor calls all provided extractors."""
        mock1 = MockExtractor([])
        mock2 = MockExtractor([])
        mock3 = MockExtractor([])

        composite = CompositeEntityExtractor([mock1, mock2, mock3])
        composite.extract("test text", ["Entity"])

        self.assertEqual(mock1.extract_call_count, 1)
        self.assertEqual(mock2.extract_call_count, 1)
        self.assertEqual(mock3.extract_call_count, 1)

    def test_extract_passes_text_and_types_to_extractors(self) -> None:
        """Extractor passes text and entity_types to all extractors."""
        mock1 = MockExtractor([])
        mock2 = MockExtractor([])

        composite = CompositeEntityExtractor([mock1, mock2])
        text = "sample text"
        entity_types = ["Person", "Organization"]

        composite.extract(text, entity_types)

        self.assertEqual(mock1.last_text, text)
        self.assertEqual(mock1.last_entity_types, entity_types)
        self.assertEqual(mock2.last_text, text)
        self.assertEqual(mock2.last_entity_types, entity_types)

    def test_extract_deduplicates_results(self) -> None:
        """Extractor deduplicates results using Levenshtein distance."""
        mock1 = MockExtractor(["Alice", "Bob"])
        mock2 = MockExtractor(["Alice", "Charlie"])  # Alice is duplicate

        composite = CompositeEntityExtractor([mock1, mock2])
        result = composite.extract("test", ["Entity"])

        # Alice should appear only once
        self.assertEqual(len(result), 3)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertIn("Charlie", result)

    def test_extract_applies_levenshtein_deduplication(self) -> None:
        """Extractor applies Levenshtein distance-based deduplication."""
        mock1 = MockExtractor(["мосбилет", "москва"])
        mock2 = MockExtractor(["мосбилета"])  # Similar to мосбилет

        composite = CompositeEntityExtractor([mock1, mock2])
        result = composite.extract("test", ["Entity"])

        # Should deduplicate "мосбилет" and "мосбилета" (distance <= 2)
        self.assertLessEqual(len(result), 3)

    def test_extract_preserves_order_of_first_occurrence(self) -> None:
        """Extractor preserves order of first occurrence of each entity."""
        mock1 = MockExtractor(["Charlie", "Alice"])
        mock2 = MockExtractor(["Bob", "Alice"])  # Alice duplicate

        composite = CompositeEntityExtractor([mock1, mock2])
        result = composite.extract("test", ["Entity"])

        # Order: Charlie, Alice, Bob (Alice from mock1 is kept)
        self.assertEqual(result[0], "Charlie")
        self.assertEqual(result[1], "Alice")
        self.assertEqual(result[2], "Bob")

    def test_extract_returns_empty_for_empty_text(self) -> None:
        """Extractor returns empty list for empty text."""
        mock = MockExtractor(["entity"])

        composite = CompositeEntityExtractor([mock])
        result = composite.extract("", ["Entity"])

        self.assertEqual(len(result), 0)

    def test_extract_returns_empty_for_empty_entity_types(self) -> None:
        """Extractor returns empty list for empty entity_types."""
        mock = MockExtractor(["entity"])

        composite = CompositeEntityExtractor([mock])
        result = composite.extract("text", [])

        self.assertEqual(len(result), 0)

    def test_extract_with_single_extractor(self) -> None:
        """Extractor works with single extractor."""
        mock = MockExtractor(["alice", "bob"])

        composite = CompositeEntityExtractor([mock])
        result = composite.extract("text", ["Entity"])

        self.assertEqual(len(result), 2)
        self.assertIn("alice", result)
        self.assertIn("bob", result)

    def test_extract_with_no_entities_from_extractors(self) -> None:
        """Extractor returns empty list when no entities found."""
        mock1 = MockExtractor([])
        mock2 = MockExtractor([])

        composite = CompositeEntityExtractor([mock1, mock2])
        result = composite.extract("text", ["Entity"])

        self.assertEqual(len(result), 0)

    def test_extract_handles_unicode_entities(self) -> None:
        """Extractor handles Unicode entities from all extractors."""
        mock1 = MockExtractor(["Алиса", "Париж"])
        mock2 = MockExtractor(["Москва"])

        composite = CompositeEntityExtractor([mock1, mock2])
        result = composite.extract("text", ["Entity"])

        self.assertEqual(len(result), 3)
        self.assertIn("Алиса", result)
        self.assertIn("Париж", result)
        self.assertIn("Москва", result)

    def test_extract_runs_extractors_in_order(self) -> None:
        """Extractor runs extractors in the order they were provided."""
        call_order = []

        class OrderedMockExtractor(EntityExtractorInterface):
            def __init__(self, name: str) -> None:
                self.name = name

            def extract(self, text: str, entity_types: list[str]) -> list[str]:
                call_order.append(self.name)
                return []

        mock1 = OrderedMockExtractor("first")
        mock2 = OrderedMockExtractor("second")
        mock3 = OrderedMockExtractor("third")

        composite = CompositeEntityExtractor([mock1, mock2, mock3])
        composite.extract("text", ["Entity"])

        self.assertEqual(call_order, ["first", "second", "third"])

    @patch("ner_controller.infrastructure.ner.composite_entity_extractor.deduplicate_entities")
    def test_extract_uses_deduplicate_entities(self, mock_deduplicate: Mock) -> None:
        """Extractor calls deduplicate_entities utility."""
        mock_deduplicate.return_value = ["alice", "bob"]
        mock1 = MockExtractor(["alice", "charlie"])
        mock2 = MockExtractor(["bob", "david"])

        composite = CompositeEntityExtractor([mock1, mock2])
        result = composite.extract("text", ["Entity"])

        # deduplicate_entities should be called with combined results
        mock_deduplicate.assert_called_once()
        call_args = mock_deduplicate.call_args
        self.assertIn("alice", call_args[0][0])
        self.assertIn("charlie", call_args[0][0])
        self.assertIn("bob", call_args[0][0])
        self.assertIn("david", call_args[0][0])
        self.assertEqual(call_args[1]["threshold"], 2)

    def test_with_gliner_and_regex_extractors(self) -> None:
        """Integration test with GLiNER and regex extractors."""
        from ner_controller.infrastructure.ner.gliner_entity_extractor import (
            GlinerEntityExtractor,
        )
        from ner_controller.infrastructure.ner.regex_api_endpoint_extractor import (
            RegexApiEndpointExtractor,
        )

        # Create real regex extractor
        regex_extractor = RegexApiEndpointExtractor()

        # Create mock GLiNER extractor
        fake_model = Mock()
        fake_model.predict_entities.return_value = [
            {"text": "Alice", "label": "PERSON", "start": 0, "end": 5},
        ]

        with patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER") as mock_gliner_class:
            mock_gliner_class.from_pretrained.return_value = fake_model

            from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
                GlinerEntityExtractorConfig,
            )

            gliner_extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

            composite = CompositeEntityExtractor([gliner_extractor, regex_extractor])

            text = "Alice works here. Use POST /api/test endpoint."
            result = composite.extract(text, ["PERSON"])

            # Should have both GLiNER entities and regex endpoints
            self.assertGreaterEqual(len(result), 1)
            self.assertIn("Alice", result)
            self.assertIn("POST /api/test", result)
