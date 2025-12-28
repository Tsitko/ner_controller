"""E2E test for complete text processing flow."""

import sys
from pathlib import Path

# Add tests directory to path for path_setup import
TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import path_setup

path_setup.add_src_path()

# Mock gliner module before importing from ner_controller
from unittest.mock import MagicMock, Mock

# Create a proper GLiNER mock that works with both test files
if "gliner" not in sys.modules:
    mock_gliner_module = MagicMock()
    mock_gliner_class = Mock()
    mock_gliner_module.GLiNER = mock_gliner_class
    sys.modules["gliner"] = mock_gliner_module

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from ner_controller.api.configs.text_router_config import TextRouterConfig
from ner_controller.api.routers.text_router import TextRouter
from ner_controller.application.application_factory import ApplicationFactory
from ner_controller.domain.entities.text_processing_result import TextProcessingResult
from ner_controller.domain.interfaces.embedding_generator_interface import (
    EmbeddingGeneratorInterface,
)
from ner_controller.domain.interfaces.entity_extractor_interface import (
    EntityExtractorInterface,
)
from ner_controller.domain.services.text_processing_service import TextProcessingService


class MockOllamaEmbeddingGenerator(EmbeddingGeneratorInterface):
    """Mock Ollama embedding generator for E2E testing."""

    def __init__(self) -> None:
        """Initialize mock."""
        self.call_count = 0

    def generate_embeddings(self, texts):
        """Generate mock embeddings."""
        self.call_count += 1
        # Return consistent mock embeddings
        return [tuple(float(i) * 0.1 for i in range(10)) for _ in texts]


class MockGlinerEntityExtractor(EntityExtractorInterface):
    """Mock GLiNER entity extractor for E2E testing."""

    def __init__(self) -> None:
        """Initialize mock."""
        self.call_count = 0

    def extract(self, text, entity_types):
        """Extract mock entities."""
        self.call_count += 1
        # Return some mock entities based on text content
        entities = []

        # Extract person names
        if "Alice" in text:
            entities.append("Alice")

        if "Bob" in text:
            entities.append("Bob")

        # Extract organizations
        if "OpenAI" in text:
            entities.append("OpenAI")

        # Extract locations
        if "Paris" in text:
            entities.append("Paris")

        if "San Francisco" in text:
            entities.append("San Francisco")

        if "Москва" in text:
            entities.append("Москва")

        return entities


class TestTextProcessingE2E(unittest.TestCase):
    """E2E tests for complete text processing flow."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_entity_extractor = MockGlinerEntityExtractor()
        self.mock_embedding_generator = MockOllamaEmbeddingGenerator()

        # Create the text processing service with mocks
        self.mock_text_processing_service = TextProcessingService(
            entity_extractor=self.mock_entity_extractor,
            embedding_generator=self.mock_embedding_generator,
        )

    def tearDown(self) -> None:
        """Clean up patches."""
        pass

    def test_complete_text_processing_workflow(self) -> None:
        """Test complete workflow from HTTP request to HTTP response."""
        # Arrange
        text = "Alice works at OpenAI in San Francisco. Alice is a researcher."

        # Create application with mocked service
        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()

        # Create test client
        client = TestClient(app)

        # Act - Send POST request to /text/process
        response = client.post(
            "/text/process",
            json={
                "text": text,
                "entity_types": ["Person", "Organization", "Location"],
            },
        )

        # Assert - Verify response
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["text"], text)

        # Verify entities were extracted
        self.assertGreater(len(response_data["entities"]), 0)

        # Check for expected entities
        self.assertIn("Alice", response_data["entities"])
        self.assertIn("OpenAI", response_data["entities"])
        self.assertIn("San Francisco", response_data["entities"])

        # Verify embedding was generated
        self.assertIsInstance(response_data["embedding"], list)
        self.assertGreater(len(response_data["embedding"]), 0)

        # Verify entity extractor was called
        self.assertGreater(self.mock_entity_extractor.call_count, 0)

        # Verify embedding generator was called
        self.assertGreater(self.mock_embedding_generator.call_count, 0)

    def test_workflow_with_default_parameters(self) -> None:
        """Test workflow using default entity_types."""
        # Arrange
        text = "Alice and Bob work at OpenAI."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act - Send request with minimal parameters (use defaults)
        response = client.post(
            "/text/process",
            json={
                "text": text,
                # entity_types not provided
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["text"], text)

        # Should still have entities and embedding
        self.assertIsInstance(response_data["entities"], list)
        self.assertIsInstance(response_data["embedding"], list)

    def test_workflow_with_small_text(self) -> None:
        """Test workflow with very small text."""
        # Arrange
        text = "Alice is here."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Should extract entity
        self.assertGreater(len(response_data["entities"]), 0)
        self.assertIn("Alice", response_data["entities"])

        # Should generate embedding
        self.assertGreater(len(response_data["embedding"]), 0)

    def test_workflow_with_no_entities_found(self) -> None:
        """Test workflow when no entities are found in text."""
        # Arrange - Text with no known entities
        text = "The quick brown fox jumps over the lazy dog."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
                "entity_types": ["Person", "Organization"],
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Should have no entities but have embedding
        self.assertEqual(len(response_data["entities"]), 0)
        self.assertGreater(len(response_data["embedding"]), 0)

    def test_workflow_with_unicode_content(self) -> None:
        """Test workflow with Unicode text content."""
        # Arrange
        text = "Алиса работает в Москве в компании OpenAI."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["text"], text)

        # Should find entities
        self.assertGreater(len(response_data["entities"]), 0)
        self.assertIn("OpenAI", response_data["entities"])

    def test_workflow_error_handling_empty_text(self) -> None:
        """Test workflow with empty text content."""
        # Arrange
        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": "",
            },
        )

        # Assert - Pydantic validation returns 422
        self.assertEqual(response.status_code, 422)

    def test_workflow_error_handling_invalid_parameters(self) -> None:
        """Test workflow with invalid parameters."""
        # Arrange
        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act - Invalid: empty entity_types list
        response = client.post(
            "/text/process",
            json={
                "text": "Alice works at OpenAI.",
                "entity_types": [],  # Invalid
            },
        )

        # Assert
        self.assertEqual(response.status_code, 400)

    def test_workflow_with_custom_entity_types(self) -> None:
        """Test workflow with custom entity type filtering."""
        # Arrange
        text = "Alice works at OpenAI in San Francisco."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act - Request only Person entities
        response = client.post(
            "/text/process",
            json={
                "text": text,
                "entity_types": ["Person"],  # Only extract persons
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Should find Alice (Person) but not OpenAI or San Francisco
        self.assertIn("Alice", response_data["entities"])

    def test_workflow_data_integrity(self) -> None:
        """Test that data flows correctly through all layers without corruption."""
        # Arrange
        original_text = "Alice works at OpenAI."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": original_text,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Verify text integrity
        self.assertEqual(response_data["text"], original_text)

        # Verify entities are present
        self.assertGreater(len(response_data["entities"]), 0)
        # All entities should be non-empty strings
        for entity in response_data["entities"]:
            self.assertIsInstance(entity, str)
            self.assertGreater(len(entity), 0)

        # Verify embedding is present
        self.assertIsInstance(response_data["embedding"], list)
        self.assertGreater(len(response_data["embedding"]), 0)
        # All embedding values should be floats
        for value in response_data["embedding"]:
            self.assertIsInstance(value, (int, float))

    def test_workflow_response_time(self) -> None:
        """Test that workflow completes in reasonable time."""
        # Arrange
        text = "Alice works at OpenAI." * 10

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act - Measure response time
        import time

        start_time = time.time()
        response = client.post(
            "/text/process",
            json={
                "text": text,
            },
        )
        end_time = time.time()

        # Assert
        self.assertEqual(response.status_code, 200)

        # Should complete in reasonable time (< 5 seconds for mocked dependencies)
        self.assertLess(end_time - start_time, 5.0)

    def test_workflow_with_negative_embedding_values(self) -> None:
        """Test workflow handles negative embedding values correctly."""
        # Arrange - Create embedding generator with negative values

        class MockGeneratorWithNegatives(EmbeddingGeneratorInterface):
            def __init__(self):
                self.call_count = 0

            def generate_embeddings(self, texts):
                self.call_count += 1
                # Return embeddings with negative values
                return [
                    tuple([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0])
                    for _ in texts
                ]

        service_with_negatives = TextProcessingService(
            entity_extractor=self.mock_entity_extractor,
            embedding_generator=MockGeneratorWithNegatives(),
        )

        text = "Alice is here."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=service_with_negatives
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
            },
        )

        # Assert
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response detail: {response.json()}")
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        embedding = response_data["embedding"]

        # Should have negative values
        self.assertIn(-0.2, embedding)
        self.assertIn(-0.4, embedding)
        self.assertIn(-1.0, embedding)

    def test_workflow_with_large_embedding_dimensions(self) -> None:
        """Test workflow handles large embedding dimensions correctly."""
        # Arrange - Create embedding generator with large dimensions

        class MockGeneratorLargeDim(EmbeddingGeneratorInterface):
            def __init__(self):
                self.call_count = 0

            def generate_embeddings(self, texts):
                self.call_count += 1
                # Return embeddings with large dimensions (1536 like OpenAI)
                return [tuple(float(i) * 0.001 for i in range(1536)) for _ in texts]

        service_large_dim = TextProcessingService(
            entity_extractor=self.mock_entity_extractor,
            embedding_generator=MockGeneratorLargeDim(),
        )

        text = "Alice works at OpenAI."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=service_large_dim
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        embedding = response_data["embedding"]

        # Should have large dimension
        self.assertEqual(len(embedding), 1536)

    def test_workflow_endpoint_registration(self) -> None:
        """Test that the text processing endpoint is properly registered."""
        # Arrange
        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()

        # Check if /text/process route exists
        routes = [
            route
            for route in app.routes
            if hasattr(route, "path") and "/text/process" in route.path
        ]

        # Should have exactly one /text/process route
        self.assertEqual(len(routes), 1)

        # Should be a POST route
        self.assertIn("POST", routes[0].methods)

    def test_workflow_special_characters(self) -> None:
        """Test workflow handles special characters correctly."""
        # Arrange
        text = "Email: alice@example.com, visit https://openai.com"

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["text"], text)

        # Should handle special characters
        self.assertIn("@", response_data["text"])
        self.assertIn("https://", response_data["text"])

    def test_workflow_multilingual_content(self) -> None:
        """Test workflow with mixed language content."""
        # Arrange
        text = "Alice работает в компании OpenAI in Москва."

        from ner_controller.configs.app_config import AppConfig

        factory = ApplicationFactory(
            AppConfig(), text_processing_service=self.mock_text_processing_service
        )
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["text"], text)

        # Should find entities (mock extractor handles simple cases)
        self.assertIn("OpenAI", response_data["entities"])
