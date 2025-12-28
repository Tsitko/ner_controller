"""Integration tests for TextRouter with TextProcessingService."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from ner_controller.api.configs.text_router_config import TextRouterConfig
from ner_controller.api.routers.text_router import TextRouter
from ner_controller.api.schemas.text_process_request import TextProcessRequest
from ner_controller.domain.entities.text_processing_result import TextProcessingResult


class MockTextProcessingService:
    """Mock service for integration testing."""

    def __init__(self, result: TextProcessingResult) -> None:
        """Initialize with predefined result."""
        self._result = result
        self.last_call_args = None

    def process_text(self, **kwargs):
        """Store call args and return predefined result."""
        self.last_call_args = kwargs
        return self._result


class TestTextRouterIntegration(unittest.TestCase):
    """Integration tests for TextRouter with service."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextRouterConfig(prefix="/text", tags=("text-processing",))

    def test_full_request_response_cycle(self) -> None:
        """Router processes request and returns response correctly."""
        # Arrange
        text = "Alice visited Paris and met with OpenAI researchers."

        # Create mock result
        mock_result = TextProcessingResult(
            text=text,
            entities=("Alice", "Paris", "OpenAI", "researchers"),
            embedding=(0.1, 0.2, 0.3, 0.4),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        # Create FastAPI app and test client
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": text,
                "entity_types": ["Person", "Location", "Organization"],
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(response_data["text"], text)
        self.assertEqual(len(response_data["entities"]), 4)
        self.assertEqual(len(response_data["embedding"]), 4)

        # Verify entity data
        self.assertIn("Alice", response_data["entities"])
        self.assertIn("Paris", response_data["entities"])
        self.assertIn("OpenAI", response_data["entities"])

        # Verify embedding data
        self.assertEqual(response_data["embedding"], [0.1, 0.2, 0.3, 0.4])

        # Verify service was called correctly
        self.assertIsNotNone(mock_service.last_call_args)
        self.assertEqual(mock_service.last_call_args["text"], text)
        self.assertEqual(
            mock_service.last_call_args["entity_types"],
            ["Person", "Location", "Organization"],
        )

    def test_request_with_default_parameters(self) -> None:
        """Router uses default entity_types when not provided in request."""
        # Arrange
        mock_result = TextProcessingResult(
            text="Alice works at OpenAI.",
            entities=("Alice", "OpenAI"),
            embedding=(0.1, 0.2),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act - request without entity_types
        response = client.post(
            "/text/process",
            json={
                "text": "Alice works at OpenAI.",
                # entity_types not provided
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        # Verify service received None for entity_types
        self.assertIsNone(mock_service.last_call_args["entity_types"])

    def test_request_with_validation_error(self) -> None:
        """Router returns 400 for invalid request parameters."""
        # Arrange
        mock_result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=(),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act - request with empty entity_types list
        response = client.post(
            "/text/process",
            json={
                "text": "Text",
                "entity_types": [],  # Invalid: empty list
            },
        )

        # Assert
        self.assertEqual(response.status_code, 400)

    def test_request_with_empty_text(self) -> None:
        """Router returns 422 for empty text content (Pydantic validation)."""
        # Arrange
        mock_result = TextProcessingResult(
            text="",
            entities=(),
            embedding=(),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": "",  # Invalid: empty text
            },
        )

        # Assert - Pydantic validation returns 422
        self.assertEqual(response.status_code, 422)

    def test_request_with_service_error(self) -> None:
        """Router returns 500 when service raises unexpected error."""
        # Arrange - service that raises error

        class ErrorService:
            def process_text(self, **kwargs):
                raise RuntimeError("Service failed")

        router = TextRouter(
            config=self.config, service=ErrorService()  # type: ignore
        )

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": "Text",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertIn("Processing failed", response.json()["detail"])

    def test_request_with_value_error_from_service(self) -> None:
        """Router returns 400 when service raises ValueError."""
        # Arrange - service that raises ValueError

        class ValueErrorService:
            def process_text(self, **kwargs):
                raise ValueError("Processing failed: invalid data")

        router = TextRouter(
            config=self.config, service=ValueErrorService()  # type: ignore
        )

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": "Valid text",  # Valid text, but service raises ValueError
            },
        )

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertIn("Processing failed", response.json()["detail"])

    def test_request_with_no_entities_found(self) -> None:
        """Router handles case when no entities are found."""
        # Arrange
        mock_result = TextProcessingResult(
            text="The quick brown fox jumps over the lazy dog.",
            entities=(),  # No entities
            embedding=(0.1, 0.2),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": "The quick brown fox jumps over the lazy dog.",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(len(response_data["entities"]), 0)
        self.assertEqual(len(response_data["embedding"]), 2)

    def test_request_with_unicode_content(self) -> None:
        """Router handles Unicode text content correctly."""
        # Arrange
        text = "Алиса работает в Париже. Hello world!"

        mock_result = TextProcessingResult(
            text=text,
            entities=("Алиса", "Париж"),
            embedding=(0.1, 0.2),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
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
        self.assertIn("Алиса", response_data["entities"])
        self.assertIn("Париж", response_data["entities"])

    def test_request_with_large_embedding(self) -> None:
        """Router handles large embedding dimensions correctly."""
        # Arrange
        large_embedding = tuple(float(i) * 0.01 for i in range(10000))

        mock_result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=large_embedding,
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": "Text",
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()
        self.assertEqual(len(response_data["embedding"]), 10000)

    def test_endpoint_path_and_method(self) -> None:
        """Router registers endpoint at correct path with correct method."""
        # Arrange
        mock_result = TextProcessingResult(
            text="test",
            entities=(),
            embedding=(),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        api_router = router.create_router()
        app.include_router(api_router)

        # Check routes
        process_routes = [
            route
            for route in api_router.routes
            if hasattr(route, "path") and route.path.endswith("/process")
        ]

        self.assertEqual(len(process_routes), 1)
        self.assertEqual(process_routes[0].methods, {"POST"})

    def test_response_schema_validation(self) -> None:
        """Response matches TextProcessResponse schema structure."""
        # Arrange
        mock_result = TextProcessingResult(
            text="Alice visited Paris.",
            entities=("Alice", "Paris"),
            embedding=(0.1, 0.2, 0.3),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": "Alice visited Paris.",
            },
        )

        # Assert - validate response structure
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Required fields
        self.assertIn("text", data)
        self.assertIn("entities", data)
        self.assertIn("embedding", data)

        # Field types
        self.assertIsInstance(data["text"], str)
        self.assertIsInstance(data["entities"], list)
        self.assertIsInstance(data["embedding"], list)

        # Entities are strings
        if len(data["entities"]) > 0:
            for entity in data["entities"]:
                self.assertIsInstance(entity, str)

    def test_request_with_custom_entity_types(self) -> None:
        """Router processes request with custom entity type filtering."""
        # Arrange
        mock_result = TextProcessingResult(
            text="Alice works at OpenAI in San Francisco.",
            entities=("Alice",),  # Only Person extracted
            embedding=(0.1,),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act - Request only Person entities
        response = client.post(
            "/text/process",
            json={
                "text": "Alice works at OpenAI in San Francisco.",
                "entity_types": ["Person"],  # Only extract persons
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        # Verify service received the custom entity types
        self.assertEqual(
            mock_service.last_call_args["entity_types"], ["Person"]
        )

        response_data = response.json()
        self.assertIn("Alice", response_data["entities"])

    def test_request_with_special_characters_in_text(self) -> None:
        """Router handles text with special characters correctly."""
        # Arrange
        text = "Email: test@example.com, visit http://example.com!"

        mock_result = TextProcessingResult(
            text=text,
            entities=(),
            embedding=(0.1,),
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
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

    def test_data_integrity_through_pipeline(self) -> None:
        """Test that data flows correctly through all layers without corruption."""
        # Arrange
        original_text = "Alice visited Paris and met with Bob."
        entities = ("Alice", "Paris", "Bob")
        embedding = (0.123, -0.456, 0.789)

        mock_result = TextProcessingResult(
            text=original_text,
            entities=entities,
            embedding=embedding,
        )

        mock_service = MockTextProcessingService(result=mock_result)
        router = TextRouter(config=self.config, service=mock_service)

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router.create_router())
        client = TestClient(app)

        # Act
        response = client.post(
            "/text/process",
            json={
                "text": original_text,
                "entity_types": ["Person", "Location"],
            },
        )

        # Assert
        self.assertEqual(response.status_code, 200)

        response_data = response.json()

        # Verify text integrity
        self.assertEqual(response_data["text"], original_text)

        # Verify entities integrity
        self.assertEqual(len(response_data["entities"]), 3)
        self.assertIn("Alice", response_data["entities"])
        self.assertIn("Paris", response_data["entities"])
        self.assertIn("Bob", response_data["entities"])

        # Verify embedding integrity (including negative values)
        self.assertEqual(len(response_data["embedding"]), 3)
        self.assertEqual(response_data["embedding"][0], 0.123)
        self.assertEqual(response_data["embedding"][1], -0.456)
        self.assertEqual(response_data["embedding"][2], 0.789)
