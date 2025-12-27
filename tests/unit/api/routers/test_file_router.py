"""Unit tests for FileRouter."""

import path_setup

path_setup.add_src_path()


import base64
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from ner_controller.api.configs.file_router_config import FileRouterConfig
from ner_controller.api.routers.file_router import FileRouter
from ner_controller.api.schemas.file_process_request import FileProcessRequest
from ner_controller.api.schemas.file_process_response import FileProcessResponse
from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult
from ner_controller.domain.services.file_processing_service import FileProcessingService


class TestFileRouterInitialization(unittest.TestCase):
    """Tests FileRouter initialization."""

    def test_initialize_with_config_and_service(self) -> None:
        """Router stores config and service correctly."""
        config = FileRouterConfig(prefix="/file", tags=("file-processing",))
        service = Mock(spec=FileProcessingService)

        router = FileRouter(config=config, service=service)

        self.assertEqual(router._config, config)
        self.assertIs(router._service, service)

    def test_config_and_service_are_required(self) -> None:
        """Both config and service must be provided."""
        with self.assertRaises(TypeError):
            FileRouter()

        with self.assertRaises(TypeError):
            FileRouter(config=FileRouterConfig())

        with self.assertRaises(TypeError):
            FileRouter(service=Mock(spec=FileProcessingService))


class TestFileRouterCreateRouter(unittest.TestCase):
    """Tests FileRouter.create_router method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = FileRouterConfig(prefix="/file", tags=("file-processing",))
        self.service = Mock(spec=FileProcessingService)
        self.router = FileRouter(config=self.config, service=self.service)

    def test_create_router_returns_api_router(self) -> None:
        """create_router returns FastAPI APIRouter instance."""
        from fastapi import APIRouter

        api_router = self.router.create_router()

        self.assertIsInstance(api_router, APIRouter)

    def test_create_router_sets_tags_from_config(self) -> None:
        """Router uses tags from configuration."""
        api_router = self.router.create_router()

        self.assertEqual(api_router.tags, ["file-processing"])

    def test_create_router_registers_process_endpoint(self) -> None:
        """Router registers POST /process endpoint."""
        api_router = self.router.create_router()

        # Check routes are registered
        # When using prefix, the route.path includes the full path
        routes = [route for route in api_router.routes if route.path.endswith("/process")]
        self.assertEqual(len(routes), 1)

        route = routes[0]
        self.assertEqual(route.methods, {"POST"})

    def test_create_router_endpoint_response_model(self) -> None:
        """Endpoint has correct response model set."""
        api_router = self.router.create_router()

        routes = [route for route in api_router.routes if route.path.endswith("/process")]
        self.assertEqual(len(routes), 1)
        route = routes[0]

        # Response model should be FileProcessResponse
        self.assertEqual(route.response_model, FileProcessResponse)


class TestFileRouterHandleFileProcess(unittest.TestCase):
    """Tests FileRouter.handle_file_process method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = FileRouterConfig(prefix="/file", tags=("file-processing",))
        self.service = Mock(spec=FileProcessingService)
        self.router = FileRouter(config=self.config, service=self.service)

    def test_handle_file_process_calls_service_with_correct_parameters(self) -> None:
        """Endpoint calls service.process_file with request parameters."""
        original_text = "Test content"
        encoded_file = base64.b64encode(original_text.encode()).decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            chunk_size=5000,
            chunk_overlap=500,
            entity_types=["PERSON", "ORG"],
        )

        # Mock service response
        mock_result = FileProcessingResult(
            file_id="file-123",
            entities=(),
            chunks=(),
        )
        self.service.process_file.return_value = mock_result

        with patch.object(self.router, "_validate_request"):
            with patch.object(
                self.router, "_convert_to_response", return_value=Mock()
            ) as mock_convert:
                self.router.handle_file_process(request)

                # Verify service was called with correct parameters
                self.service.process_file.assert_called_once_with(
                    file_base64=encoded_file,
                    file_id="file-123",
                    entity_types=["PERSON", "ORG"],
                    chunk_size=5000,
                    chunk_overlap=500,
                )

    def test_handle_file_process_returns_response(self) -> None:
        """Endpoint returns FileProcessResponse from conversion."""
        original_text = "Test"
        encoded_file = base64.b64encode(original_text.encode()).decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
        )

        mock_result = FileProcessingResult(
            file_id="file-123",
            entities=(),
            chunks=(),
        )
        self.service.process_file.return_value = mock_result

        mock_response = Mock(spec=FileProcessResponse)
        mock_response.file_id = "file-123"

        with patch.object(self.router, "_validate_request"):
            with patch.object(
                self.router, "_convert_to_response", return_value=mock_response
            ):
                response = self.router.handle_file_process(request)

                self.assertEqual(response.file_id, "file-123")

    def test_handle_file_process_raises_400_for_value_error(self) -> None:
        """Raises HTTPException 400 when service raises ValueError."""
        encoded_file = base64.b64encode(b"Content").decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
        )

        self.service.process_file.side_effect = ValueError("Invalid base64")

        with patch.object(self.router, "_validate_request"):
            with self.assertRaises(HTTPException) as context:
                self.router.handle_file_process(request)

            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("Invalid base64", context.exception.detail)

    def test_handle_file_process_raises_500_for_generic_exception(self) -> None:
        """Raises HTTPException 500 for unexpected errors."""
        encoded_file = base64.b64encode(b"Content").decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
        )

        self.service.process_file.side_effect = RuntimeError("Unexpected error")

        with patch.object(self.router, "_validate_request"):
            with self.assertRaises(HTTPException) as context:
                self.router.handle_file_process(request)

            self.assertEqual(context.exception.status_code, 500)
            self.assertIn("Processing failed", context.exception.detail)

    def test_handle_file_process_with_default_parameters(self) -> None:
        """Endpoint passes default chunk_size and overlap when not in request."""
        from ner_controller.domain.services.file_processing_service import (
            DEFAULT_CHUNK_OVERLAP,
            DEFAULT_CHUNK_SIZE,
        )

        encoded_file = base64.b64encode(b"Content").decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            # chunk_size and chunk_overlap use defaults (3000, 300)
        )

        mock_result = FileProcessingResult(
            file_id="file-123",
            entities=(),
            chunks=(),
        )
        self.service.process_file.return_value = mock_result

        with patch.object(self.router, "_validate_request"):
            with patch.object(
                self.router, "_convert_to_response", return_value=Mock()
            ):
                self.router.handle_file_process(request)

                # Verify defaults were passed
                call_args = self.service.process_file.call_args
                self.assertEqual(call_args[1]["chunk_size"], DEFAULT_CHUNK_SIZE)
                self.assertEqual(
                    call_args[1]["chunk_overlap"], DEFAULT_CHUNK_OVERLAP
                )


class TestFileRouterValidateRequest(unittest.TestCase):
    """Tests FileRouter._validate_request method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = FileRouterConfig(prefix="/file", tags=("file-processing",))
        self.service = Mock(spec=FileProcessingService)
        self.router = FileRouter(config=self.config, service=self.service)

    def test_validate_request_accepts_valid_parameters(self) -> None:
        """Accepts request with valid parameters."""
        encoded_file = base64.b64encode(b"Content").decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            chunk_size=1000,
            chunk_overlap=100,
        )

        # Should not raise
        self.router._validate_request(request)

    def test_validate_request_rejects_zero_chunk_size(self) -> None:
        """Raises HTTPException 400 when chunk_size is 0."""
        encoded_file = base64.b64encode(b"Content").decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            chunk_size=0,
            chunk_overlap=0,
        )

        with self.assertRaises(HTTPException) as context:
            self.router._validate_request(request)

        self.assertEqual(context.exception.status_code, 400)

    def test_validate_request_rejects_negative_chunk_size(self) -> None:
        """Raises HTTPException 400 when chunk_size is negative."""
        encoded_file = base64.b64encode(b"Content").decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            chunk_size=-100,
            chunk_overlap=0,
        )

        with self.assertRaises(HTTPException) as context:
            self.router._validate_request(request)

        self.assertEqual(context.exception.status_code, 400)

    def test_validate_request_rejects_negative_chunk_overlap(self) -> None:
        """Raises HTTPException 400 when chunk_overlap is negative."""
        encoded_file = base64.b64encode(b"Content").decode()

        request = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            chunk_size=1000,
            chunk_overlap=-10,
        )

        with self.assertRaises(HTTPException) as context:
            self.router._validate_request(request)

        self.assertEqual(context.exception.status_code, 400)

    def test_validate_request_rejects_overlap_greater_or_equal_size(self) -> None:
        """Raises HTTPException 400 when chunk_overlap >= chunk_size."""
        encoded_file = base64.b64encode(b"Content").decode()

        # overlap == size
        request1 = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            chunk_size=500,
            chunk_overlap=500,
        )

        with self.assertRaises(HTTPException) as context:
            self.router._validate_request(request1)
        self.assertEqual(context.exception.status_code, 400)

        # overlap > size
        request2 = FileProcessRequest(
            file=encoded_file,
            file_name="test.txt",
            file_id="file-123",
            chunk_size=500,
            chunk_overlap=600,
        )

        with self.assertRaises(HTTPException) as context:
            self.router._validate_request(request2)
        self.assertEqual(context.exception.status_code, 400)

    def test_validate_request_rejects_empty_file(self) -> None:
        """Raises HTTPException 400 when file field is empty."""
        request = FileProcessRequest(
            file="",
            file_name="empty.txt",
            file_id="file-empty",
        )

        with self.assertRaises(HTTPException) as context:
            self.router._validate_request(request)

        self.assertEqual(context.exception.status_code, 400)


class TestFileRouterConvertToResponse(unittest.TestCase):
    """Tests FileRouter._convert_to_response method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = FileRouterConfig(prefix="/file", tags=("file-processing",))
        self.service = Mock(spec=FileProcessingService)
        self.router = FileRouter(config=self.config, service=self.service)

    def test_convert_to_response_maps_all_fields(self) -> None:
        """Converts FileProcessingResult to FileProcessResponse correctly."""
        entity1 = "Alice"
        entity2 = "Bob"

        chunk1 = FileChunk(
            id=0,
            text="First chunk",
            entities=(entity1,),
            embedding=(0.1, 0.2),
        )

        chunk2 = FileChunk(
            id=1,
            text="Second chunk",
            entities=(entity2,),
            embedding=(0.3, 0.4),
        )

        result = FileProcessingResult(
            file_id="file-123",
            entities=(entity1, entity2),
            chunks=(chunk1, chunk2),
        )

        response = self.router._convert_to_response(result)

        self.assertIsInstance(response, FileProcessResponse)
        self.assertEqual(response.file_id, "file-123")
        self.assertEqual(len(response.entities), 2)
        self.assertEqual(len(response.chanks), 2)

    def test_convert_to_response_with_empty_result(self) -> None:
        """Converts empty FileProcessingResult correctly."""
        result = FileProcessingResult(
            file_id="empty-file",
            entities=(),
            chunks=(),
        )

        response = self.router._convert_to_response(result)

        self.assertEqual(response.file_id, "empty-file")
        self.assertEqual(len(response.entities), 0)
        self.assertEqual(len(response.chanks), 0)

    def test_convert_to_response_preserves_chunk_order(self) -> None:
        """Preserves chunk order from result to response."""
        chunks = tuple(
            FileChunk(
                id=i,
                text=f"Chunk {i}",
                entities=(),
                embedding=None,
            )
            for i in range(5)
        )

        result = FileProcessingResult(
            file_id="file-123",
            entities=(),
            chunks=chunks,
        )

        response = self.router._convert_to_response(result)

        for i, chunk_schema in enumerate(response.chanks):
            self.assertEqual(chunk_schema.id, i)




class TestFileRouterChunkToSchema(unittest.TestCase):
    """Tests FileRouter._chunk_to_schema method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = FileRouterConfig(prefix="/file", tags=("file-processing",))
        self.service = Mock(spec=FileProcessingService)
        self.router = FileRouter(config=self.config, service=self.service)

    def test_chunk_to_schema_maps_all_fields(self) -> None:
        """Converts FileChunk to ChunkSchema with all fields."""
        entity = "Alice"

        chunk = FileChunk(
            id=1,
            text="Sample text",
            entities=(entity,),
            embedding=(0.1, 0.2, 0.3),
        )

        schema = self.router._chunk_to_schema(chunk)

        self.assertEqual(schema.id, 1)
        self.assertEqual(schema.text, "Sample text")
        self.assertEqual(len(schema.entities), 1)
        self.assertEqual(schema.embedding, [0.1, 0.2, 0.3])

    def test_chunk_to_schema_with_no_entities(self) -> None:
        """Handles chunks with no entities."""
        chunk = FileChunk(
            id=0,
            text="No entities",
            entities=(),
            embedding=None,
        )

        schema = self.router._chunk_to_schema(chunk)

        self.assertEqual(len(schema.entities), 0)
        self.assertIsNone(schema.embedding)

    def test_chunk_to_schema_with_none_embedding(self) -> None:
        """Handles chunks with None embedding."""
        chunk = FileChunk(
            id=0,
            text="Text",
            entities=(),
            embedding=None,
        )

        schema = self.router._chunk_to_schema(chunk)

        self.assertIsNone(schema.embedding)

    def test_chunk_to_schema_converts_tuple_embedding_to_list(self) -> None:
        """Converts tuple embedding to list for schema."""
        chunk = FileChunk(
            id=0,
            text="Text",
            entities=(),
            embedding=(0.1, 0.2, 0.3),
        )

        schema = self.router._chunk_to_schema(chunk)

        self.assertIsInstance(schema.embedding, list)
        self.assertEqual(schema.embedding, [0.1, 0.2, 0.3])

    def test_chunk_to_schema_with_string_entities(self) -> None:
        """Handles string entities correctly."""
        entities = ("Alice", "Bob")

        chunk = FileChunk(
            id=0,
            text="Alice and Bob",
            entities=entities,
            embedding=None,
        )

        schema = self.router._chunk_to_schema(chunk)

        self.assertEqual(len(schema.entities), 2)
        self.assertEqual(schema.entities[0], "Alice")
        self.assertEqual(schema.entities[1], "Bob")
