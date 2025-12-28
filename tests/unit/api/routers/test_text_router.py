"""Unit tests for TextRouter."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from ner_controller.api.configs.text_router_config import TextRouterConfig
from ner_controller.api.routers.text_router import TextRouter
from ner_controller.api.schemas.text_process_request import TextProcessRequest
from ner_controller.api.schemas.text_process_response import TextProcessResponse
from ner_controller.domain.entities.text_processing_result import TextProcessingResult
from ner_controller.domain.services.text_processing_service import TextProcessingService


class TestTextRouterInitialization(unittest.TestCase):
    """Tests TextRouter initialization."""

    def test_initialize_with_config_and_service(self) -> None:
        """Router stores config and service correctly."""
        config = TextRouterConfig(prefix="/text", tags=("text-processing",))
        service = Mock(spec=TextProcessingService)

        router = TextRouter(config=config, service=service)

        self.assertEqual(router._config, config)
        self.assertIs(router._service, service)

    def test_config_and_service_are_required(self) -> None:
        """Both config and service must be provided."""
        with self.assertRaises(TypeError):
            TextRouter()  # type: ignore

        with self.assertRaises(TypeError):
            TextRouter(config=TextRouterConfig())  # type: ignore

        with self.assertRaises(TypeError):
            TextRouter(service=Mock(spec=TextProcessingService))  # type: ignore


class TestTextRouterCreateRouter(unittest.TestCase):
    """Tests TextRouter.create_router method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextRouterConfig(prefix="/text", tags=("text-processing",))
        self.service = Mock(spec=TextProcessingService)
        self.router = TextRouter(config=self.config, service=self.service)

    def test_create_router_returns_api_router(self) -> None:
        """create_router returns FastAPI APIRouter instance."""
        from fastapi import APIRouter

        api_router = self.router.create_router()

        self.assertIsInstance(api_router, APIRouter)

    def test_create_router_sets_tags_from_config(self) -> None:
        """Router uses tags from configuration."""
        api_router = self.router.create_router()

        self.assertEqual(api_router.tags, ["text-processing"])

    def test_create_router_registers_process_endpoint(self) -> None:
        """Router registers POST /process endpoint."""
        api_router = self.router.create_router()

        # Check routes are registered
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

        # Response model should be TextProcessResponse
        self.assertEqual(route.response_model, TextProcessResponse)


class TestTextRouterHandleTextProcess(unittest.TestCase):
    """Tests TextRouter.handle_text_process method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextRouterConfig(prefix="/text", tags=("text-processing",))
        self.service = Mock(spec=TextProcessingService)
        self.router = TextRouter(config=self.config, service=self.service)

    def test_handle_text_process_calls_service_with_correct_parameters(self) -> None:
        """Endpoint calls service.process_text with request parameters."""
        request = TextProcessRequest(
            text="Alice visited Paris.",
            entity_types=["Person", "Location"],
        )

        # Mock service response
        mock_result = TextProcessingResult(
            text="Alice visited Paris.",
            entities=("Alice", "Paris"),
            embedding=(0.1, 0.2, 0.3),
        )
        self.service.process_text.return_value = mock_result

        with patch.object(self.router, "_validate_request"):
            with patch.object(
                self.router, "_convert_to_response", return_value=Mock()
            ) as mock_convert:
                self.router.handle_text_process(request)

                # Verify service was called with correct parameters
                self.service.process_text.assert_called_once_with(
                    text="Alice visited Paris.",
                    entity_types=["Person", "Location"],
                )

    def test_handle_text_process_returns_response(self) -> None:
        """Endpoint returns TextProcessResponse from conversion."""
        request = TextProcessRequest(text="Alice works at OpenAI.")

        mock_result = TextProcessingResult(
            text="Alice works at OpenAI.",
            entities=("Alice", "OpenAI"),
            embedding=(0.1, 0.2),
        )
        self.service.process_text.return_value = mock_result

        mock_response = Mock(spec=TextProcessResponse)
        mock_response.text = "Alice works at OpenAI."

        with patch.object(self.router, "_validate_request"):
            with patch.object(
                self.router, "_convert_to_response", return_value=mock_response
            ):
                response = self.router.handle_text_process(request)

                self.assertEqual(response.text, "Alice works at OpenAI.")

    def test_handle_text_process_with_default_entity_types(self) -> None:
        """Endpoint passes None for entity_types when not in request."""
        request = TextProcessRequest(text="Text")

        mock_result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=(),
        )
        self.service.process_text.return_value = mock_result

        with patch.object(self.router, "_validate_request"):
            with patch.object(
                self.router, "_convert_to_response", return_value=Mock()
            ):
                self.router.handle_text_process(request)

                # Verify None was passed for entity_types
                call_args = self.service.process_text.call_args
                self.assertIsNone(call_args[1]["entity_types"])

    def test_handle_text_process_raises_400_for_value_error(self) -> None:
        """Raises HTTPException 400 when service raises ValueError."""
        request = TextProcessRequest(text="Test text")

        self.service.process_text.side_effect = ValueError("Text cannot be empty")

        with patch.object(self.router, "_validate_request"):
            with self.assertRaises(HTTPException) as context:
                self.router.handle_text_process(request)

            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("Text cannot be empty", context.exception.detail)

    def test_handle_text_process_raises_500_for_generic_exception(self) -> None:
        """Raises HTTPException 500 for unexpected errors."""
        request = TextProcessRequest(text="Text")

        self.service.process_text.side_effect = RuntimeError("Unexpected error")

        with patch.object(self.router, "_validate_request"):
            with self.assertRaises(HTTPException) as context:
                self.router.handle_text_process(request)

            self.assertEqual(context.exception.status_code, 500)
            self.assertIn("Processing failed", context.exception.detail)

    def test_handle_text_process_with_embedding_generation_error(self) -> None:
        """Raises HTTPException 500 for EmbeddingGenerationError."""
        from ner_controller.infrastructure.embedding.ollama_embedding_generator import (
            EmbeddingGenerationError,
        )

        request = TextProcessRequest(text="Text")

        self.service.process_text.side_effect = EmbeddingGenerationError(
            "Ollama connection failed"
        )

        with patch.object(self.router, "_validate_request"):
            with self.assertRaises(HTTPException) as context:
                self.router.handle_text_process(request)

            self.assertEqual(context.exception.status_code, 500)
            self.assertIn("Ollama connection failed", context.exception.detail)


class TestTextRouterValidateRequest(unittest.TestCase):
    """Tests TextRouter._validate_request method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextRouterConfig(prefix="/text", tags=("text-processing",))
        self.service = Mock(spec=TextProcessingService)
        self.router = TextRouter(config=self.config, service=self.service)

    def test_validate_request_accepts_valid_request(self) -> None:
        """Accepts request with valid parameters."""
        request = TextProcessRequest(
            text="Alice visited Paris.",
            entity_types=["Person", "Location"],
        )

        # Should not raise
        self.router._validate_request(request)

    def test_validate_request_accepts_none_entity_types(self) -> None:
        """Accepts request with entity_types=None."""
        request = TextProcessRequest(
            text="Text",
            entity_types=None,
        )

        # Should not raise
        self.router._validate_request(request)

    def test_validate_request_rejects_empty_entity_types_list(self) -> None:
        """Raises HTTPException 400 when entity_types is empty list."""
        request = TextProcessRequest(
            text="Text",
            entity_types=[],
        )

        with self.assertRaises(HTTPException) as context:
            self.router._validate_request(request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("entity_types cannot be an empty list", context.exception.detail)


class TestTextRouterConvertToResponse(unittest.TestCase):
    """Tests TextRouter._convert_to_response method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextRouterConfig(prefix="/text", tags=("text-processing",))
        self.service = Mock(spec=TextProcessingService)
        self.router = TextRouter(config=self.config, service=self.service)

    def test_convert_to_response_maps_all_fields(self) -> None:
        """Converts TextProcessingResult to TextProcessResponse correctly."""
        result = TextProcessingResult(
            text="Alice visited Paris.",
            entities=("Alice", "Paris"),
            embedding=(0.1, 0.2, 0.3),
        )

        response = self.router._convert_to_response(result)

        self.assertIsInstance(response, TextProcessResponse)
        self.assertEqual(response.text, "Alice visited Paris.")
        self.assertEqual(len(response.entities), 2)
        self.assertEqual(len(response.embedding), 3)

    def test_convert_to_response_with_empty_result(self) -> None:
        """Converts TextProcessingResult with no entities correctly."""
        result = TextProcessingResult(
            text="No entities",
            entities=(),
            embedding=(0.1, 0.2),
        )

        response = self.router._convert_to_response(result)

        self.assertEqual(response.text, "No entities")
        self.assertEqual(len(response.entities), 0)
        self.assertEqual(len(response.embedding), 2)

    def test_convert_to_response_with_large_embedding(self) -> None:
        """Converts result with large embedding dimension."""
        large_embedding = tuple(float(i) * 0.01 for i in range(10000))

        result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=large_embedding,
        )

        response = self.router._convert_to_response(result)

        self.assertEqual(len(response.embedding), 10000)

    def test_convert_to_response_with_unicode_entities(self) -> None:
        """Converts result with Unicode entities correctly."""
        result = TextProcessingResult(
            text="Text",
            entities=("Алиса", "Париж", "Москва"),
            embedding=(),
        )

        response = self.router._convert_to_response(result)

        self.assertEqual(len(response.entities), 3)
        self.assertIn("Алиса", response.entities)

    def test_convert_to_response_preserves_entity_order(self) -> None:
        """Preserves entity order from result to response."""
        result = TextProcessingResult(
            text="Text",
            entities=("Entity1", "Entity2", "Entity3"),
            embedding=(),
        )

        response = self.router._convert_to_response(result)

        self.assertEqual(response.entities[0], "Entity1")
        self.assertEqual(response.entities[1], "Entity2")
        self.assertEqual(response.entities[2], "Entity3")

    def test_convert_to_response_converts_tuples_to_lists(self) -> None:
        """Converts tuple fields to lists for JSON serialization."""
        result = TextProcessingResult(
            text="Text",
            entities=("Alice", "Bob"),
            embedding=(0.1, 0.2),
        )

        response = self.router._convert_to_response(result)

        # Tuples should be converted to lists
        self.assertIsInstance(response.entities, list)
        self.assertIsInstance(response.embedding, list)

    def test_convert_to_response_with_negative_embedding_values(self) -> None:
        """Converts result with negative embedding values correctly."""
        result = TextProcessingResult(
            text="Text",
            entities=(),
            embedding=(-0.5, -0.2, 0.1, 0.3),
        )

        response = self.router._convert_to_response(result)

        self.assertEqual(response.embedding[0], -0.5)
        self.assertEqual(response.embedding[1], -0.2)


class TestTextRouterCompleteFlow(unittest.TestCase):
    """Tests complete router flow scenarios."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextRouterConfig(prefix="/text", tags=("text-processing",))
        self.service = Mock(spec=TextProcessingService)
        self.router = TextRouter(config=self.config, service=self.service)

    def test_full_flow_success(self) -> None:
        """Complete successful flow from request to response."""
        request = TextProcessRequest(
            text="Alice works at OpenAI in San Francisco.",
            entity_types=["Person", "Organization", "Location"],
        )

        mock_result = TextProcessingResult(
            text="Alice works at OpenAI in San Francisco.",
            entities=("Alice", "OpenAI", "San Francisco"),
            embedding=(0.1, 0.2, 0.3, 0.4),
        )
        self.service.process_text.return_value = mock_result

        response = self.router.handle_text_process(request)

        # Verify response structure
        self.assertIsInstance(response, TextProcessResponse)
        self.assertEqual(response.text, "Alice works at OpenAI in San Francisco.")
        self.assertEqual(len(response.entities), 3)
        self.assertEqual(len(response.embedding), 4)

        # Verify service was called
        self.service.process_text.assert_called_once()

    def test_flow_with_validation_error(self) -> None:
        """Flow with empty entity_types list raises 400."""
        request = TextProcessRequest(
            text="Text",
            entity_types=[],
        )

        with self.assertRaises(HTTPException) as context:
            self.router.handle_text_process(request)

        self.assertEqual(context.exception.status_code, 400)
        # Service should not be called
        self.service.process_text.assert_not_called()

    def test_flow_with_service_error(self) -> None:
        """Flow with service error raises 500."""
        request = TextProcessRequest(text="Text")

        self.service.process_text.side_effect = RuntimeError("Service error")

        with self.assertRaises(HTTPException) as context:
            self.router.handle_text_process(request)

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("Processing failed", context.exception.detail)
