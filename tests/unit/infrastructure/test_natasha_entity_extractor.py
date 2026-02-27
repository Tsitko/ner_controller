"""Unit tests for NatashaEntityExtractor."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock, patch

from ner_controller.infrastructure.ner.configs.natasha_entity_extractor_config import (
    NatashaEntityExtractorConfig,
)
from ner_controller.infrastructure.ner.natasha_entity_extractor import NatashaEntityExtractor


class _FakeSpan:
    """Simple span object for tests."""

    def __init__(self, start: int, stop: int, type_name: str) -> None:
        self.start = start
        self.stop = stop
        self.type = type_name


class _FakeMarkup:
    """Simple markup object for tests."""

    def __init__(self, spans) -> None:
        self.spans = spans


class TestNatashaEntityExtractor(unittest.TestCase):
    """Tests Natasha extraction behavior."""

    def setUp(self) -> None:
        """Set up extractor instance."""
        self.extractor = NatashaEntityExtractor(NatashaEntityExtractorConfig())

    def test_extract_returns_empty_for_irrelevant_labels(self) -> None:
        """Extractor returns empty if requested labels are unsupported by Natasha."""
        with patch.object(self.extractor, "_load_ner_model") as load_model:
            result = self.extractor.extract("text", ["Product", "Date"])
        self.assertEqual(result, [])
        load_model.assert_not_called()

    def test_extract_filters_by_requested_labels(self) -> None:
        """Extractor keeps only spans matching requested mapped labels."""
        text = "Анна Иванова работает в Яндексе в Москве."
        fake_ner = Mock()
        fake_ner.return_value = _FakeMarkup(
            [
                _FakeSpan(0, 12, "PER"),
                _FakeSpan(24, 32, "ORG"),
                _FakeSpan(35, 41, "LOC"),
            ]
        )

        with patch.object(self.extractor, "_load_ner_model", return_value=fake_ner):
            result = self.extractor.extract(text, ["Person", "Organization"])

        self.assertIn("Анна Иванова", result)
        self.assertIn("Яндексе", result)
        self.assertNotIn("Москве", result)

    def test_extract_deduplicates_entities(self) -> None:
        """Extractor deduplicates repeated spans."""
        text = "Анна Иванова и Анна Иванова."
        fake_ner = Mock()
        fake_ner.return_value = _FakeMarkup(
            [
                _FakeSpan(0, 12, "PER"),
                _FakeSpan(15, 27, "PER"),
            ]
        )

        with patch.object(self.extractor, "_load_ner_model", return_value=fake_ner):
            result = self.extractor.extract(text, ["PERSON"])

        self.assertEqual(result, ["Анна Иванова"])

    def test_load_ner_model_raises_for_missing_files(self) -> None:
        """Extractor fails fast when Natasha model files are absent."""
        config = NatashaEntityExtractorConfig(
            cache_dir="/tmp/definitely_missing_natasha_models",
            navec_model_filename="missing_navec.tar",
            ner_model_filename="missing_slovnet.tar",
        )
        extractor = NatashaEntityExtractor(config)

        with self.assertRaises(FileNotFoundError):
            extractor._load_ner_model()

    def test_extract_disables_itself_after_missing_models(self) -> None:
        """Extractor disables itself after first missing-model error."""
        config = NatashaEntityExtractorConfig(
            cache_dir="/tmp/definitely_missing_natasha_models",
            navec_model_filename="missing_navec.tar",
            ner_model_filename="missing_slovnet.tar",
        )
        extractor = NatashaEntityExtractor(config)

        with patch.object(extractor, "_load_ner_model", side_effect=FileNotFoundError("missing")) as loader:
            first = extractor.extract("Анна Иванова", ["PERSON"])
            second = extractor.extract("Анна Иванова", ["PERSON"])

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(loader.call_count, 1)
