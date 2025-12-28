"""Unit tests for TextProcessingService."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock, patch

from ner_controller.domain.entities.text_processing_result import TextProcessingResult
from ner_controller.domain.interfaces.embedding_generator_interface import (
    EmbeddingGeneratorInterface,
)
from ner_controller.domain.interfaces.entity_extractor_interface import (
    EntityExtractorInterface,
)
from ner_controller.domain.services.text_processing_service import (
    DEFAULT_ENTITY_TYPES,
    TextProcessingService,
)


class TestTextProcessingServiceInitialization(unittest.TestCase):
    """Tests TextProcessingService initialization and dependencies."""

    def test_initialize_with_all_dependencies(self) -> None:
        """Service stores all dependencies correctly."""
        entity_extractor = Mock(spec=EntityExtractorInterface)
        embedding_generator = Mock(spec=EmbeddingGeneratorInterface)

        service = TextProcessingService(
            entity_extractor=entity_extractor,
            embedding_generator=embedding_generator,
        )

        self.assertIs(service._entity_extractor, entity_extractor)
        self.assertIs(service._embedding_generator, embedding_generator)

    def test_dependencies_are_required(self) -> None:
        """All dependencies must be provided (no defaults)."""
        with self.assertRaises(TypeError):
            TextProcessingService()  # type: ignore

        with self.assertRaises(TypeError):
            TextProcessingService(
                entity_extractor=Mock(spec=EntityExtractorInterface)
            )  # type: ignore


class TestTextProcessingServiceProcessText(unittest.TestCase):
    """Tests TextProcessingService.process_text method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)

        self.service = TextProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
        )

    def test_process_text_with_default_entity_types(self) -> None:
        """Service uses DEFAULT_ENTITY_TYPES when entity_types is None."""
        text = "Alice works at OpenAI."

        self.entity_extractor.extract.return_value = ["Alice", "OpenAI"]
        self.embedding_generator.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3]
        ]

        result = self.service.process_text(text=text, entity_types=None)

        # Verify entity extractor was called with default types
        call_args = self.entity_extractor.extract.call_args
        self.assertEqual(call_args[0][1], DEFAULT_ENTITY_TYPES)

        # Verify result structure
        self.assertIsInstance(result, TextProcessingResult)
        self.assertEqual(result.text, text)
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.embedding), 3)

    def test_process_text_with_custom_entity_types(self) -> None:
        """Service uses custom entity types when provided."""
        text = "Alice works at OpenAI."
        custom_types = ["PERSON", "ORG"]

        self.entity_extractor.extract.return_value = ["Alice"]
        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2]]

        result = self.service.process_text(text=text, entity_types=custom_types)

        # Verify entity extractor was called with custom types
        self.entity_extractor.extract.assert_called_with(text, custom_types)

        # Verify result structure
        self.assertEqual(len(result.entities), 1)
        self.assertIn("Alice", result.entities)

    def test_process_text_returns_text_processing_result(self) -> None:
        """Service returns TextProcessingResult with correct structure."""
        text = "Alice and Bob"

        self.entity_extractor.extract.return_value = ["Alice", "Bob"]
        self.embedding_generator.generate_embeddings.return_value = [
            [0.1, 0.2, 0.3]
        ]

        result = self.service.process_text(text=text)

        self.assertIsInstance(result, TextProcessingResult)
        self.assertEqual(result.text, text)
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.embedding), 3)

    def test_process_text_with_empty_entities(self) -> None:
        """Service handles case when no entities are extracted."""
        text = "Just plain text with no entities."

        self.entity_extractor.extract.return_value = []
        self.embedding_generator.generate_embeddings.return_value = [[0.1]]

        result = self.service.process_text(text=text)

        self.assertEqual(len(result.entities), 0)
        self.assertEqual(len(result.embedding), 1)

    def test_process_text_normalizes_whitespace(self) -> None:
        """Service strips leading/trailing whitespace from text."""
        text = "  Alice works here.  "

        self.entity_extractor.extract.return_value = ["Alice"]
        self.embedding_generator.generate_embeddings.return_value = [[0.1]]

        result = self.service.process_text(text=text)

        # Text should be normalized (whitespace stripped)
        self.assertEqual(result.text, "Alice works here.")
        # Entity extractor should receive normalized text
        self.entity_extractor.extract.assert_called_with(
            "Alice works here.", unittest.mock.ANY
        )

    def test_process_text_with_unicode_content(self) -> None:
        """Service handles Unicode text correctly."""
        text = "Алиса работает в OpenAI в Париже."

        self.entity_extractor.extract.return_value = ["Алиса", "OpenAI", "Париж"]
        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2]]

        result = self.service.process_text(text=text)

        self.assertEqual(result.text, text)
        self.assertIn("Алиса", result.entities)


class TestTextProcessingServiceValidateAndNormalizeText(unittest.TestCase):
    """Tests TextProcessingService._validate_and_normalize_text method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)

        self.service = TextProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
        )

    def test_validate_normal_text_returns_stripped(self) -> None:
        """Normal text is returned stripped of whitespace."""
        text = "  Normal text  "
        result = self.service._validate_and_normalize_text(text)

        self.assertEqual(result, "Normal text")

    def test_validate_empty_string_raises_value_error(self) -> None:
        """Empty string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.service._validate_and_normalize_text("")

        self.assertIn("empty", str(context.exception).lower())

    def test_validate_whitespace_only_raises_value_error(self) -> None:
        """String with only whitespace raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.service._validate_and_normalize_text("   \t\n  ")

        self.assertIn("whitespace", str(context.exception).lower())

    def test_validate_unicode_text(self) -> None:
        """Unicode text is normalized correctly."""
        text = "  Текст с пробелами  "
        result = self.service._validate_and_normalize_text(text)

        self.assertEqual(result, "Текст с пробелами")

    def test_validate_text_with_internal_whitespace_preserved(self) -> None:
        """Internal whitespace is preserved."""
        text = "  Text   with   internal   whitespace  "
        result = self.service._validate_and_normalize_text(text)

        self.assertEqual(result, "Text   with   internal   whitespace")


class TestTextProcessingServiceExtractEntities(unittest.TestCase):
    """Tests TextProcessingService._extract_entities method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)

        self.service = TextProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
        )

    def test_extract_entities_calls_extractor(self) -> None:
        """Calls entity extractor with correct parameters."""
        text = "Alice is here"
        entity_types = ["PERSON", "ORG"]

        self.entity_extractor.extract.return_value = ["Alice"]

        result = self.service._extract_entities(text, entity_types)

        self.entity_extractor.extract.assert_called_once_with(text, entity_types)
        self.assertEqual(len(result), 1)
        self.assertIn("Alice", result)

    def test_extract_entities_returns_list(self) -> None:
        """Returns list of entity strings."""
        text = "Alice and Bob"

        self.entity_extractor.extract.return_value = ("Alice", "Bob")

        result = self.service._extract_entities(text, ["PERSON"])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_extract_entities_with_no_results(self) -> None:
        """Returns empty list when no entities found."""
        text = "No entities here"

        self.entity_extractor.extract.return_value = []

        result = self.service._extract_entities(text, ["PERSON"])

        self.assertEqual(len(result), 0)

    def test_extract_entities_propagates_exception(self) -> None:
        """Propagates exceptions from entity extractor."""
        text = "Text"

        self.entity_extractor.extract.side_effect = RuntimeError("NER failed")

        with self.assertRaises(RuntimeError) as context:
            self.service._extract_entities(text, ["PERSON"])

        self.assertIn("NER failed", str(context.exception))


class TestTextProcessingServiceGenerateEmbedding(unittest.TestCase):
    """Tests TextProcessingService._generate_embedding method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.entity_extractor = Mock(spec=EntityExtractorInterface)
        self.embedding_generator = Mock(spec=EmbeddingGeneratorInterface)

        self.service = TextProcessingService(
            entity_extractor=self.entity_extractor,
            embedding_generator=self.embedding_generator,
        )

    def test_generate_embedding_calls_generator(self) -> None:
        """Calls embedding generator with text list."""
        text = "Sample text"

        self.embedding_generator.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        result = self.service._generate_embedding(text)

        # Verify generator was called with list containing the text
        self.embedding_generator.generate_embeddings.assert_called_once_with([text])

        # Verify result
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], 0.1)

    def test_generate_embedding_returns_list(self) -> None:
        """Returns list of floats."""
        text = "Text"

        self.embedding_generator.generate_embeddings.return_value = [(0.1, 0.2)]

        result = self.service._generate_embedding(text)

        self.assertIsInstance(result, list)
        self.assertEqual(result, [0.1, 0.2])

    def test_generate_embedding_with_none_result_raises_error(self) -> None:
        """Raises EmbeddingGenerationError when generator returns None."""
        from ner_controller.infrastructure.embedding.ollama_embedding_generator import (
            EmbeddingGenerationError,
        )

        text = "Text"
        self.embedding_generator.generate_embeddings.return_value = [None]

        with self.assertRaises(EmbeddingGenerationError) as context:
            self.service._generate_embedding(text)

        self.assertIn("Failed to generate embedding", str(context.exception))

    def test_generate_embedding_with_empty_result_raises_error(self) -> None:
        """Raises EmbeddingGenerationError when generator returns empty list."""
        from ner_controller.infrastructure.embedding.ollama_embedding_generator import (
            EmbeddingGenerationError,
        )

        text = "Text"
        self.embedding_generator.generate_embeddings.return_value = []

        with self.assertRaises(EmbeddingGenerationError) as context:
            self.service._generate_embedding(text)

        self.assertIn("Failed to generate embedding", str(context.exception))

    def test_generate_embedding_with_unicode_text(self) -> None:
        """Handles Unicode text correctly."""
        text = "Привет, мир!"

        self.embedding_generator.generate_embeddings.return_value = [[0.1]]

        result = self.service._generate_embedding(text)

        self.assertEqual(len(result), 1)

    def test_generate_embedding_large_dimensions(self) -> None:
        """Handles large embedding dimensions."""
        text = "Text"
        large_embedding = [float(i) * 0.01 for i in range(10000)]

        self.embedding_generator.generate_embeddings.return_value = [large_embedding]

        result = self.service._generate_embedding(text)

        self.assertEqual(len(result), 10000)


class TestTextProcessingServiceConstants(unittest.TestCase):
    """Tests default constants."""

    def test_default_entity_types_has_21_types(self) -> None:
        """DEFAULT_ENTITY_TYPES contains 21 entity types."""
        self.assertEqual(len(DEFAULT_ENTITY_TYPES), 21)

    def test_default_entity_types_contains_common_types(self) -> None:
        """DEFAULT_ENTITY_TYPES contains expected common entity types."""
        self.assertIn("Person", DEFAULT_ENTITY_TYPES)
        self.assertIn("Organization", DEFAULT_ENTITY_TYPES)
        self.assertIn("Location", DEFAULT_ENTITY_TYPES)
        self.assertIn("Event", DEFAULT_ENTITY_TYPES)
        self.assertIn("Product", DEFAULT_ENTITY_TYPES)

    def test_default_entity_types_all_strings(self) -> None:
        """All items in DEFAULT_ENTITY_TYPES are strings."""
        for entity_type in DEFAULT_ENTITY_TYPES:
            self.assertIsInstance(entity_type, str)
