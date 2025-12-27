"""HTTP router for file processing."""

from fastapi import APIRouter, HTTPException

from ner_controller.api.configs.file_router_config import FileRouterConfig
from ner_controller.api.schemas.file_process_request import FileProcessRequest
from ner_controller.api.schemas.file_process_response import ChunkSchema, FileProcessResponse
from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.domain.entities.file_processing_result import FileProcessingResult
from ner_controller.domain.services.file_processing_service import FileProcessingService


class FileRouter:
    """Creates routes for file processing endpoints."""

    def __init__(
        self,
        config: FileRouterConfig,
        service: FileProcessingService,
    ) -> None:
        """
        Initialize router with configuration and service.

        Args:
            config: Router configuration for prefix and tags.
            service: File processing service for business logic.
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
            self.handle_file_process,
            methods=["POST"],
            response_model=FileProcessResponse,
        )
        return router

    def handle_file_process(
        self,
        request: FileProcessRequest,
    ) -> FileProcessResponse:
        """
        Handle a file processing request.

        Args:
            request: File processing request with base64-encoded content.

        Returns:
            FileProcessResponse with entities and chunked text with embeddings.

        Raises:
            HTTPException: If processing fails (400 for client errors, 500 for server errors).
        """
        # Validate parameters
        self._validate_request(request)

        try:
            # Process the file
            result = self._service.process_file(
                file_base64=request.file,
                file_id=request.file_id,
                entity_types=request.entity_types,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )

            # Convert domain entities to response schema
            return self._convert_to_response(result)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    def _validate_request(self, request: FileProcessRequest) -> None:
        """
        Validate request parameters.

        Args:
            request: Request to validate.

        Raises:
            HTTPException: If validation fails.
        """
        if request.chunk_size <= 0:
            raise HTTPException(status_code=400, detail=f"chunk_size must be > 0, got {request.chunk_size}")
        if request.chunk_overlap < 0:
            raise HTTPException(status_code=400, detail=f"chunk_overlap must be >= 0, got {request.chunk_overlap}")
        if request.chunk_overlap >= request.chunk_size:
            raise HTTPException(
                status_code=400,
                detail=f"chunk_overlap must be < chunk_size, got chunk_overlap={request.chunk_overlap}, chunk_size={request.chunk_size}",
            )
        if not request.file or not request.file.strip():
            raise HTTPException(status_code=400, detail="file content cannot be empty")

    def _convert_to_response(self, result: FileProcessingResult) -> FileProcessResponse:
        """
        Convert domain result to API response schema.

        Args:
            result: File processing result from domain service.

        Returns:
            FileProcessResponse for HTTP response.
        """
        chunk_schemas = [self._chunk_to_schema(chunk) for chunk in result.chunks]

        return FileProcessResponse(
            file_id=result.file_id,
            entities=list(result.entities),
            chanks=chunk_schemas,
        )

    def _chunk_to_schema(self, chunk: FileChunk) -> ChunkSchema:
        """Convert domain FileChunk to schema."""
        return ChunkSchema(
            id=chunk.id,
            text=chunk.text,
            entities=list(chunk.entities),
            embedding=list(chunk.embedding) if chunk.embedding is not None else None,
        )

