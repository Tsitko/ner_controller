"""HTTP router for hallucination checks."""

from fastapi import APIRouter, HTTPException

from ner_controller.api.configs.hallucination_router_config import HallucinationRouterConfig
from ner_controller.api.schemas.hallucination_check_request import HallucinationCheckRequest
from ner_controller.api.schemas.hallucination_check_response import HallucinationCheckResponse
from ner_controller.domain.services.hallucination_detection_service import HallucinationDetectionService


class HallucinationRouter:
    """Creates routes for hallucination detection endpoints."""

    def __init__(
        self,
        config: HallucinationRouterConfig,
        service: HallucinationDetectionService,
    ) -> None:
        """Initialize router with configuration and service."""
        self._config = config
        self._service = service

    def create_router(self) -> APIRouter:
        """Create an APIRouter with registered endpoints."""
        router = APIRouter(tags=list(self._config.tags))
        router.add_api_route(
            "/check",
            self.handle_hallucination_check,
            methods=["POST"],
            response_model=HallucinationCheckResponse,
        )
        return router

    def handle_hallucination_check(
        self,
        request: HallucinationCheckRequest,
    ) -> HallucinationCheckResponse:
        """Handle a hallucination check request."""
        if not request.entity_types:
            raise HTTPException(status_code=400, detail="entity_types must be provided.")

        result = self._service.detect(
            request_text=request.request,
            response_text=request.response,
            entity_types=request.entity_types,
        )
        return HallucinationCheckResponse(
            potential_hallucinations=result.potential_hallucinations,
            missing_entities=result.missing_entities,
        )
