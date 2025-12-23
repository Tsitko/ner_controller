"""Request schema for hallucination checks."""

from pydantic import BaseModel, Field


class HallucinationCheckRequest(BaseModel):
    """Input payload for hallucination detection."""

    request: str = Field(..., description="Full prompt and context sent to the LLM.")
    response: str = Field(..., description="Full response produced by the LLM.")
    entities_types: list[str] = Field(
        ..., description="List of entity type labels to extract."
    )
