"""Response schema for hallucination checks."""

from pydantic import BaseModel, Field


class HallucinationCheckResponse(BaseModel):
    """Output payload containing hallucination analysis results."""

    potential_hallucinations: list[str] = Field(
        ..., description="Entities found only in the response."
    )
    missing_entities: list[str] = Field(
        ..., description="Entities found only in the request."
    )
