"""Unit tests for GlinerEntityExtractor."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock, patch

from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)
from ner_controller.infrastructure.ner.gliner_entity_extractor import GlinerEntityExtractor


class TestGlinerEntityExtractor(unittest.TestCase):
    """Tests GLiNER extraction behavior."""

    @patch("ner_controller.infrastructure.ner.gliner_entity_extractor.GLiNER")
    def test_extract_maps_predictions_to_entities(self, gliner_class: Mock) -> None:
        """Extractor loads model and maps GLiNER outputs to Entity objects."""
        fake_model = Mock()
        fake_model.predict_entities.return_value = [
            {"text": "Alice", "label": "PERSON", "start": 0, "end": 5}
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

        result = extractor.extract("Alice", ["PERSON"])

        gliner_class.from_pretrained.assert_called_once_with(
            "model",
            cache_dir="/tmp/gliner-cache",
            local_files_only=True,
        )
        fake_model.predict_entities.assert_called_once()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "Alice")
        self.assertEqual(result[0].label, "PERSON")
        self.assertEqual(result[0].start, 0)
        self.assertEqual(result[0].end, 5)

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
