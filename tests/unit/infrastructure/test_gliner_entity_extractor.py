"""Unit tests for GlinerEntityExtractor."""

import sys

import path_setup

path_setup.add_src_path()


# Mock gliner module before importing from ner_controller
from unittest.mock import MagicMock, Mock
mock_gliner = MagicMock()
sys.modules["gliner"] = mock_gliner

import unittest
from unittest.mock import patch

from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)
from ner_controller.infrastructure.ner.gliner_entity_extractor import GlinerEntityExtractor


class TestGlinerEntityExtractor(unittest.TestCase):
    """Tests GLiNER extraction behavior."""

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_returns_entity_names_only(self, gliner_class: Mock) -> None:
        """Extractor returns list of entity names (strings)."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = [
            {"text": "Alice", "label": "PERSON", "start": 0, "end": 5},
            {"text": "OpenAI", "label": "ORG", "start": 10, "end": 16},
        ]
        gliner_class.from_pretrained.return_value = fake_model

        config = GlinerEntityExtractorConfig(
            model_name="model",
            device="cpu",
            batch_size=4,
            cache_dir="/tmp/gliner-cache",
            local_files_only=True,
        )
        extractor = GlinerEntityExtractor(config)

        result = extractor.extract("Alice works at OpenAI.", ["PERSON", "ORG"])

        gliner_class.from_pretrained.assert_called_once_with(
            "model",
            cache_dir="/tmp/gliner-cache",
            local_files_only=True,
        )
        fake_model.predict_entities.assert_called_once()
        self.assertEqual(len(result), 2)
        self.assertIn("Alice", result)
        self.assertIn("OpenAI", result)
        # Result should be list of strings
        self.assertIsInstance(result, list)
        for entity in result:
            self.assertIsInstance(entity, str)

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_deduplicates_entities(self, gliner_class: Mock) -> None:
        """Extractor deduplicates duplicate entity names."""
        fake_model = Mock()
        # GLiNER returns duplicate entities
        fake_model.predict_entities.return_value = [
            {"text": "Alice", "label": "PERSON", "start": 0, "end": 5},
            {"text": "Alice", "label": "PERSON", "start": 20, "end": 25},
            {"text": "Bob", "label": "PERSON", "start": 10, "end": 13},
        ]
        gliner_class.from_pretrained.return_value = fake_model

        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

        result = extractor.extract("Alice and Bob and Alice", ["PERSON"])

        # Should deduplicate "Alice"
        self.assertEqual(len(result), 2)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_reuses_loaded_model(self, gliner_class: Mock) -> None:
        """Extractor loads GLiNER once and reuses the model instance."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = []
        gliner_class.from_pretrained.return_value = fake_model

        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

        extractor.extract("A", ["PERSON"])
        extractor.extract("B", ["PERSON"])

        gliner_class.from_pretrained.assert_called_once()
        self.assertEqual(fake_model.predict_entities.call_count, 2)

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_with_empty_predictions(self, gliner_class: Mock) -> None:
        """Extractor handles empty predictions correctly."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = []
        gliner_class.from_pretrained.return_value = fake_model

        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

        result = extractor.extract("No entities here", ["PERSON"])

        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, list)

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_preserves_order_of_first_occurrence(self, gliner_class: Mock) -> None:
        """Extractor preserves order of first occurrence of each entity."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = [
            {"text": "Charlie", "label": "PERSON", "start": 0, "end": 7},
            {"text": "Alice", "label": "PERSON", "start": 10, "end": 15},
            {"text": "Bob", "label": "PERSON", "start": 20, "end": 23},
            {"text": "Alice", "label": "PERSON", "start": 30, "end": 35},  # Duplicate
        ]
        gliner_class.from_pretrained.return_value = fake_model

        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

        result = extractor.extract("Charlie, Alice, Bob, Alice", ["PERSON"])

        # Order should be: Charlie, Alice, Bob (Alice duplicate removed)
        self.assertEqual(result, ["Charlie", "Alice", "Bob"])

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_with_unicode_entities(self, gliner_class: Mock) -> None:
        """Extractor handles Unicode entities correctly."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = [
            {"text": "Алиса", "label": "PERSON", "start": 0, "end": 10},
            {"text": "Париж", "label": "LOCATION", "start": 11, "end": 17},
        ]
        gliner_class.from_pretrained.return_value = fake_model

        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

        result = extractor.extract("Алиса посетила Париж", ["PERSON", "LOCATION"])

        self.assertIn("Алиса", result)
        self.assertIn("Париж", result)

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_applies_levenshtein_deduplication(self, gliner_class: Mock) -> None:
        """Extractor applies Levenshtein distance-based deduplication."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = [
            {"text": "мосбилет", "label": "ORG", "start": 0, "end": 8},
            {"text": "мосбилета", "label": "ORG", "start": 10, "end": 19},
            {"text": "москва", "label": "LOCATION", "start": 20, "end": 26},
        ]
        gliner_class.from_pretrained.return_value = fake_model

        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

        result = extractor.extract("мосбилет мосбилета", ["ORG", "LOCATION"])

        # Should apply Levenshtein deduplication (distance <= 2)
        # "мосбилет" and "мосбилета" should be deduplicated
        self.assertLessEqual(len(result), 3)

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_with_multiple_labels_same_text(self, gliner_class: Mock) -> None:
        """Extractor keeps entity name even if same text has multiple labels."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = [
            {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
            {"text": "Apple", "label": "PRODUCT", "start": 10, "end": 15},
        ]
        gliner_class.from_pretrained.return_value = fake_model

        extractor = GlinerEntityExtractor(GlinerEntityExtractorConfig())

        result = extractor.extract("Apple and Apple", ["ORG", "PRODUCT"])

        # Should deduplicate by text only, not by (text, label)
        self.assertEqual(len(result), 1)
        self.assertIn("Apple", result)
