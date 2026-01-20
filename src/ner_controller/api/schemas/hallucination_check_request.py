"""Request schema for hallucination checks."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class HallucinationCheckRequest(BaseModel):
    """Input payload for hallucination detection."""

    model_config = ConfigDict(populate_by_name=True)

    request: str = Field(..., description="Full prompt and context sent to the LLM.")
    response: str = Field(..., description="Full response produced by the LLM.")
    entity_types: list[str] = Field(
        ...,
        description=(
            "List of entity type labels to extract. "
            "Accepts `entity_types` (preferred) or legacy `entities_types`."
        ),
        validation_alias=AliasChoices("entity_types", "entities_types"),
    )
