"""HTTP router for text processing."""

from fastapi import APIRouter, HTTPException

from ner_controller.api.configs.text_router_config import TextRouterConfig
from ner_controller.api.schemas.text_process_request import TextProcessRequest
from ner_controller.api.schemas.text_process_response import TextProcessResponse
from ner_controller.domain.entities.text_processing_result import TextProcessingResult
from ner_controller.domain.services.text_processing_service import TextProcessingService


class TextRouter:
    """Creates routes for text processing endpoints."""

    def __init__(
        self,
        config: TextRouterConfig,
        service: TextProcessingService,
    ) -> None:
        """
        Initialize router with configuration and service.

        Args:
            config: Router configuration for prefix and tags.
            service: Text processing service for business logic.
        """
        self._config = config
        self._service = service

    def create_router(self) -> APIRouter:
        """
        Create an APIRouter with registered endpoints.

        Returns:
            Configured FastAPI APIRouter instance.
        """
        router = APIRouter(prefix=self._config.prefix, tags=list(self._config.tags))
        router.add_api_route(
            "/process",
            self.handle_text_process,
            methods=["POST"],
            response_model=TextProcessResponse,
        )
        return router

    def handle_text_process(
        self,
        request: TextProcessRequest,
    ) -> TextProcessResponse:
        """
        Handle a text processing request.

        Args:
            request: Text processing request with plain text content.

        Returns:
            TextProcessResponse with entities and embedding.

        Raises:
            HTTPException: If processing fails (400 for client errors, 500 for server errors).
        """
        # Validate request
        self._validate_request(request)

        try:
            # Process the text
            result = self._service.process_text(
                text=request.text,
                entity_types=request.entity_types,
            )

            # Convert domain result to response schema
            return self._convert_to_response(result)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            # Check if it's an EmbeddingGenerationError
            error_type = type(e).__name__
            if error_type == "EmbeddingGenerationError":
                raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    def _validate_request(self, request: TextProcessRequest) -> None:
        """
        Validate request parameters.

        Args:
            request: Request to validate.

        Raises:
            HTTPException: If validation fails.
        """
        # Pydantic already validates text is not empty (min_length=1)
        # Additional validation can be added here if needed
        if request.entity_types is not None and not request.entity_types:
            raise HTTPException(status_code=400, detail="entity_types cannot be an empty list")

    def _convert_to_response(self, result: TextProcessingResult) -> TextProcessResponse:
        """
        Convert domain result to API response schema.

        Args:
            result: Text processing result from domain service.

        Returns:
            TextProcessResponse for HTTP response.
        """
        return TextProcessResponse(
            text=result.text,
            entities=list(result.entities),
            embedding=list(result.embedding),
        )
