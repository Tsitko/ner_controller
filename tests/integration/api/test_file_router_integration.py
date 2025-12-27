"""Integration tests for FileRouter with FileProcessingService."""

import path_setup

path_setup.add_src_path()


import base64
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from ner_controller.api.configs.file_router_config import FileRouterConfig
from ner_controller.api.routers.file_router import FileRouter
from ner_controller.api.schemas.file_process_request import FileProcessRequest
from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult


class MockFileProcessingService:
    """Mock service for integration testing."""

    def __init__(self, result: FileProcessingResult) -> None:
        """Initialize with predefined result."""
        self._result = result
        self.last_call_args = None

    def process_file(self, **kwargs):
        """Store call args and return predefined result."""
        self.last_call_args = kwargs
        return self._result


class TestFileRouterIntegration(unittest.TestCase):
    """Integration tests for FileRouter with service."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = FileRouterConfig(prefix="/file", tags=("file-processing",))

    def test_full_request_response_cycle(self) -> None:
        """Router processes request and returns response correctly."""
        # Arrange
        original_text = "Alice works at OpenAI."
        encoded_file = base64.b64encode(original_text.encode()).decode()

        # Create mock result
        entity1 = "Alice"
        entity2 = "OpenAI"

        chunk = FileChunk(
            id=0,
            text=original_text,
            entities=(entity1, entity2),
            embedding=(0.1, 0.2, 0.3),
        )

        mock_result = FileProcessingResult(
            file_id="file-123",
            entities=(entity1, entity2),
            chunks=(chunk,),
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        # Create FastAPI app and test client
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "file-123",
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "entity_types": ["PERSON", "ORG"],
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["file_id"], "file-123")
        self.assertEqual(len(response_data["entities"]), 2)
        self.assertEqual(len(response_data["chanks"]), 1)

        # Verify entity data
        self.assertEqual(response_data["entities"][0], "Alice")
        self.assertEqual(response_data["entities"][1], "OpenAI")

        # Verify chunk data
        self.assertEqual(response_data["chanks"][0]["id"], 0)
        self.assertEqual(response_data["chanks"][0]["text"], original_text)
        self.assertEqual(len(response_data["chanks"][0]["entities"]), 2)
        self.assertEqual(response_data["chanks"][0]["embedding"], [0.1, 0.2, 0.3])

        # Verify service was called correctly
        self.assertIsNotNone(mock_service.last_call_args)
        self.assertEqual(
            mock_service.last_call_args["file_base64"], encoded_file
        )
        self.assertEqual(mock_service.last_call_args["file_id"], "file-123")
        self.assertEqual(
            mock_service.last_call_args["entity_types"], ["PERSON", "ORG"]
        )
        self.assertEqual(mock_service.last_call_args["chunk_size"], 1000)
        self.assertEqual(mock_service.last_call_args["chunk_overlap"], 100)

    def test_request_with_default_parameters(self) -> None:
        """Router uses default parameters when not provided in request."""
        # Arrange
        encoded_file = base64.b64encode(b"Content").decode()

        mock_result = FileProcessingResult(
            file_id="file-456",
            entities=(),
            chunks=(),
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act - request without chunk_size, chunk_overlap, entity_types
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "file-456",
                # chunk_size, chunk_overlap, entity_types not provided
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        # Verify service received defaults
        from ner_controller.domain.services.file_processing_service import (
            DEFAULT_CHUNK_OVERLAP,
            DEFAULT_CHUNK_SIZE,
        )

        self.assertEqual(
            mock_service.last_call_args["chunk_size"], DEFAULT_CHUNK_SIZE
        )
        self.assertEqual(
            mock_service.last_call_args["chunk_overlap"], DEFAULT_CHUNK_OVERLAP
        )
        self.assertIsNone(mock_service.last_call_args["entity_types"])

    def test_request_with_validation_error(self) -> None:
        """Router returns 400 for invalid request parameters."""
        # Arrange
        encoded_file = base64.b64encode(b"Content").decode()

        mock_result = FileProcessingResult(
            file_id="file-error",
            entities=(),
            chunks=(),
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act - request with invalid chunk_overlap >= chunk_size
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "file-error",
                "chunk_size": 100,
                "chunk_overlap": 100,  # Invalid: overlap == size
            },
        )

        # Assert
        self.assertEqual(response.status_code, 400)

    def test_request_with_empty_file(self) -> None:
        """Router returns 400 for empty file content."""
        # Arrange
        mock_result = FileProcessingResult(
            file_id="empty",
            entities=(),
            chunks=(),
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": "",
                "file_name": "empty.txt",
                "file_id": "empty",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 400)

    def test_request_with_service_error(self) -> None:
        """Router returns 500 when service raises unexpected error."""
        # Arrange - service that raises error

        class ErrorService:
            def process_file(self, **kwargs):
                raise RuntimeError("Service failed")

        router = FileRouter(
            config=self.config, service=ErrorService()  # type: ignore
        )

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        encoded_file = base64.b64encode(b"Content").decode()

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "error-file",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertIn("Processing failed", response.json()["detail"])

    def test_request_with_value_error_from_service(self) -> None:
        """Router returns 400 when service raises ValueError."""
        # Arrange - service that raises ValueError

        class ValueErrorService:
            def process_file(self, **kwargs):
                raise ValueError("Invalid base64 content")

        router = FileRouter(
            config=self.config, service=ValueErrorService()  # type: ignore
        )

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        encoded_file = base64.b64encode(b"Content").decode()

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "error-file",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid base64 content", response.json()["detail"])

    def test_request_with_multiple_chunks_response(self) -> None:
        """Router correctly serializes response with multiple chunks."""
        # Arrange
        encoded_file = base64.b64encode(b"Large text").decode()

        entity1 = "Entity1"
        entity2 = "Entity2"

        chunks = (
            FileChunk(
                id=0,
                text="First chunk",
                entities=(entity1,),
                embedding=(0.1, 0.2),
            ),
            FileChunk(
                id=1,
                text="Second chunk",
                entities=(entity2,),
                embedding=(0.3, 0.4),
            ),
        )

        mock_result = FileProcessingResult(
            file_id="multi-chunk",
            entities=(entity1, entity2),
            chunks=chunks,
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "multi-chunk",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(len(response_data["chanks"]), 2)

        # Verify first chunk
        self.assertEqual(response_data["chanks"][0]["id"], 0)
        self.assertEqual(response_data["chanks"][0]["text"], "First chunk")
        self.assertEqual(len(response_data["chanks"][0]["entities"]), 1)
        self.assertEqual(response_data["chanks"][0]["entities"][0], "Entity1")
        self.assertEqual(response_data["chanks"][0]["embedding"], [0.1, 0.2])

        # Verify second chunk
        self.assertEqual(response_data["chanks"][1]["id"], 1)
        self.assertEqual(response_data["chanks"][1]["text"], "Second chunk")
        self.assertEqual(len(response_data["chanks"][1]["entities"]), 1)
        self.assertEqual(response_data["chanks"][1]["entities"][0], "Entity2")
        self.assertEqual(response_data["chanks"][1]["embedding"], [0.3, 0.4])

    def test_request_with_chunks_having_no_embeddings(self) -> None:
        """Router handles chunks with None embeddings correctly."""
        # Arrange
        encoded_file = base64.b64encode(b"Text").decode()

        chunks = (
            FileChunk(
                id=0,
                text="With embedding",
                entities=(),
                embedding=(0.1, 0.2),
            ),
            FileChunk(
                id=1,
                text="Without embedding",
                entities=(),
                embedding=None,
            ),
        )

        mock_result = FileProcessingResult(
            file_id="partial-embeddings",
            entities=(),
            chunks=chunks,
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "partial-embeddings",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["chanks"][0]["embedding"], [0.1, 0.2])
        self.assertIsNone(response_data["chanks"][1]["embedding"])

    def test_request_with_unicode_content(self) -> None:
        """Router handles Unicode file content correctly."""
        # Arrange
        original_text = "Привет, мир! Hello world!"
        encoded_file = base64.b64encode(original_text.encode("utf-8")).decode()

        mock_result = FileProcessingResult(
            file_id="unicode-file",
            entities=(),
            chunks=(),
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
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
        self.assertEqual(response.json()["file_id"], "unicode-file")

    def test_endpoint_path_and_method(self) -> None:
        """Router registers endpoint at correct path with correct method."""
        # Arrange
        mock_result = FileProcessingResult(
            file_id="test",
            entities=(),
            chunks=(),
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        api_router = router.create_router()
        app.include_router(api_router)

        # Check routes
        # When using prefix, the route.path includes the full path
        process_routes = [
            route for route in api_router.routes if hasattr(route, "path") and route.path.endswith("/process")
        ]

        self.assertEqual(len(process_routes), 1)
        self.assertEqual(process_routes[0].methods, {"POST"})

    def test_response_schema_validation(self) -> None:
        """Response matches FileProcessResponse schema structure."""
        # Arrange
        encoded_file = base64.b64encode(b"Content").decode()

        entity = "Alice"

        mock_result = FileProcessingResult(
            file_id="schema-test",
            entities=(entity,),
            chunks=(
                FileChunk(
                    id=0,
                    text="Text",
                    entities=(entity,),
                    embedding=(0.1, 0.2, 0.3),
                ),
            ),
        )

        mock_service = MockFileProcessingService(result=mock_result)
        router = FileRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/file/process",
            json={
                "file": encoded_file,
                "file_name": "test.txt",
                "file_id": "schema-test",
            },
        )

        # Assert - validate response structure
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Required fields
        self.assertIn("file_id", data)
        self.assertIn("entities", data)
        self.assertIn("chanks", data)

        # Entity structure
        if len(data["entities"]) > 0:
            # Entities are now just strings
            self.assertIsInstance(data["entities"][0], str)

        # Chunk structure
        if len(data["chanks"]) > 0:
            chunk_data = data["chanks"][0]
            self.assertIn("id", chunk_data)
            self.assertIn("text", chunk_data)
            self.assertIn("entities", chunk_data)
            self.assertIn("embedding", chunk_data)
