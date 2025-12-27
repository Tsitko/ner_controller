"""E2E test for complete file processing flow."""

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

import base64
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from ner_controller.api.configs.file_router_config import FileRouterConfig
from ner_controller.api.routers.file_router import FileRouter
from ner_controller.application.application_factory import ApplicationFactory
from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult
from ner_controller.domain.interfaces.embedding_generator_interface import (
    EmbeddingGeneratorInterface,
)
from ner_controller.domain.interfaces.entity_extractor_interface import (
    EntityExtractorInterface,
)
from ner_controller.domain.interfaces.text_chunker_interface import (
    TextChunkerInterface,
)
from ner_controller.domain.services.file_processing_service import FileProcessingService


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

        if "Alice" in text:
            entities.append("Alice")

        if "OpenAI" in text:
            entities.append("OpenAI")

        if "San Francisco" in text:
            entities.append("San Francisco")

        return entities


class MockTextChunker(TextChunkerInterface):
    """Mock text chunker for E2E testing."""

    def split_text(self, text, chunk_size, chunk_overlap, start_id=0):
        """Split text into mock chunks."""
        if not text:
            return []

        chunks = []
        chunk_id = start_id
        position = 0

        while position < len(text):
            end_pos = min(position + chunk_size, len(text))
            chunk_text = text[position:end_pos]

            chunks.append(
                FileChunk(
                    id=chunk_id,
                    text=chunk_text,
                    entities=(),
                    embedding=None,
                )
            )

            position += (chunk_size - chunk_overlap)
            chunk_id += 1

            if end_pos >= len(text):
                break

        return chunks


class TestFileProcessingE2E(unittest.TestCase):
    """E2E tests for complete file processing flow."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create mock dependencies
        self.mock_entity_extractor = MockGlinerEntityExtractor()
        self.mock_embedding_generator = MockOllamaEmbeddingGenerator()
        self.mock_text_chunker = MockTextChunker()

        # Create the file processing service with mocks
        self.mock_file_processing_service = FileProcessingService(
            entity_extractor=self.mock_entity_extractor,
            embedding_generator=self.mock_embedding_generator,
            text_chunker=self.mock_text_chunker,
        )

    def tearDown(self) -> None:
        """Clean up patches."""
        pass

    def test_complete_file_processing_workflow(self) -> None:
        """Test complete workflow from HTTP request to HTTP response."""
        # Arrange
        original_text = "Alice works at OpenAI in San Francisco. Alice is a researcher."
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        # Create application with mocked service
        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()

        # Create test client
        client = TestClient(app)

        # Act - Send POST request to /file/process
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test_document.txt",
                "file_id": "test-file-001",
                "chunk_size": 50,
                "chunk_overlap": 10,
                "entity_types": ["PERSON", "ORG", "LOCATION"],
            },
        )

        # Assert - Verify response
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["file_id"], "test-file-001")

        # Verify entities were extracted
        self.assertGreater(len(response_data["entities"]), 0)

        # Check for expected entities (entities are now just strings)
        self.assertIn("Alice", response_data["entities"])
        self.assertIn("OpenAI", response_data["entities"])
        self.assertIn("San Francisco", response_data["entities"])

        # Verify chunks were created
        self.assertGreater(len(response_data["chanks"]), 0)

        # Verify chunk structure
        for chunk in response_data["chanks"]:
            self.assertIn("id", chunk)
            self.assertIn("text", chunk)
            self.assertIn("entities", chunk)
            self.assertIn("embedding", chunk)

        # Verify embeddings were generated (should not be None)
        chunks_with_embeddings = [
            c for c in response_data["chanks"] if c["embedding"] is not None
        ]
        self.assertEqual(len(chunks_with_embeddings), len(response_data["chanks"]))

        # Verify entity extractor was called
        self.assertGreater(self.mock_entity_extractor.call_count, 0)

        # Verify embedding generator was called
        self.assertGreater(self.mock_embedding_generator.call_count, 0)

    def test_workflow_with_default_parameters(self) -> None:
        """Test workflow using default chunk_size, chunk_overlap, and entity_types."""
        # Arrange
        original_text = "Alice and Bob work at OpenAI."
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act - Send request with minimal parameters (use defaults)
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "minimal.txt",
                "file_id": "minimal-file",
                # chunk_size, chunk_overlap, entity_types not provided
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["file_id"], "minimal-file")

        # Should still have chunks and entities
        self.assertIsInstance(response_data["chanks"], list)
        self.assertIsInstance(response_data["entities"], list)

    def test_workflow_with_small_file(self) -> None:
        """Test workflow with very small file (single chunk)."""
        # Arrange
        original_text = "Alice is here."
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "small.txt",
                "file_id": "small-file",
                "chunk_size": 1000,
                "chunk_overlap": 100,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Should create single chunk
        self.assertEqual(len(response_data["chanks"]), 1)
        self.assertEqual(response_data["chanks"][0]["text"], original_text)

        # Should extract entity
        self.assertGreater(len(response_data["entities"]), 0)
        # entities is now a list of strings, not objects
        self.assertIn("Alice", response_data["entities"])

    def test_workflow_with_large_file(self) -> None:
        """Test workflow with larger file (multiple chunks)."""
        # Arrange - Create text that will span multiple chunks
        original_text = "Alice works at OpenAI. " * 20  # Repeat to create long text
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "large.txt",
                "file_id": "large-file",
                "chunk_size": 100,  # Force multiple chunks
                "chunk_overlap": 20,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Should create multiple chunks
        self.assertGreater(len(response_data["chanks"]), 1)

        # All chunks should have embeddings
        for chunk in response_data["chanks"]:
            self.assertIsNotNone(chunk["embedding"])

        # Chunks should have sequential IDs
        for i, chunk in enumerate(response_data["chanks"]):
            self.assertEqual(chunk["id"], i)

    def test_workflow_with_no_entities_found(self) -> None:
        """Test workflow when no entities are found in text."""
        # Arrange - Text with no known entities
        original_text = "The quick brown fox jumps over the lazy dog."
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "no-entities.txt",
                "file_id": "no-entities-file",
                "entity_types": ["PERSON", "ORG"],
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Should have chunks but no entities
        self.assertGreater(len(response_data["chanks"]), 0)
        self.assertEqual(len(response_data["entities"]), 0)

    def test_workflow_with_unicode_content(self) -> None:
        """Test workflow with Unicode text content."""
        # Arrange
        original_text = "Алиса работает в OpenAI в Сан-Франциско."
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "unicode.txt",
                "file_id": "unicode-file",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["file_id"], "unicode-file")

        # Should have chunks with text
        self.assertGreater(len(response_data["chanks"]), 0)

    def test_workflow_error_handling_invalid_base64(self) -> None:
        """Test workflow with invalid base64 content."""
        # Arrange - Invalid base64
        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": "This is not valid base64!!!",
                "file_name": "invalid.txt",
                "file_id": "invalid-file",
            },
        )

        # Assert - Should return 400 error
        self.assertEqual(response.status_code, 400)

    def test_workflow_error_handling_invalid_parameters(self) -> None:
        """Test workflow with invalid chunk parameters."""
        # Arrange
        encoded_file = base64.b64encode(b"Content").decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act - Invalid: chunk_overlap >= chunk_size
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "test",
                "chunk_size": 100,
                "chunk_overlap": 100,  # Invalid
            },
        )

        # Assert
        self.assertEqual(response.status_code, 400)

    def test_workflow_with_custom_entity_types(self) -> None:
        """Test workflow with custom entity type filtering."""
        # Arrange
        original_text = "Alice works at OpenAI in San Francisco."
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act - Request only PERSON entities
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "filtered.txt",
                "file_id": "filtered-file",
                "entity_types": ["PERSON"],  # Only extract persons
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # entities is now a list of strings, check if Alice is present
        self.assertIn("Alice", response_data["entities"])

    def test_workflow_data_integrity(self) -> None:
        """Test that data flows correctly through all layers without corruption."""
        # Arrange
        original_text = "Alice works at OpenAI."
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "integrity.txt",
                "file_id": "integrity-file",
                "chunk_size": 50,
                "chunk_overlap": 10,
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Verify file_id integrity
        self.assertEqual(response_data["file_id"], "integrity-file")

        # Verify chunk text contains original text
        all_chunk_text = " ".join(chunk["text"] for chunk in response_data["chanks"])
        self.assertIn("Alice", all_chunk_text)
        self.assertIn("OpenAI", all_chunk_text)

        # Verify entities are present (entities are now just strings)
        self.assertGreater(len(response_data["entities"]), 0)
        # All entities should be non-empty strings
        for entity in response_data["entities"]:
            self.assertIsInstance(entity, str)
            self.assertGreater(len(entity), 0)

    def test_workflow_response_time(self) -> None:
        """Test that workflow completes in reasonable time."""
        # Arrange
        original_text = "Alice works at OpenAI." * 10
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        from ner_controller.configs.app_config import AppConfig
        factory = ApplicationFactory(AppConfig(), file_processing_service=self.mock_file_processing_service)
        app = factory.create_app()
        client = TestClient(app)

        # Act - Measure response time
        import time

        start_time = time.time()
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "performance.txt",
                "file_id": "performance-file",
            },
        )
        end_time = time.time()

        # Assert
        self.assertEqual(response.status_code, 200)

        # Should complete in reasonable time (< 5 seconds for mocked dependencies)
        self.assertLess(end_time - start_time, 5.0)
