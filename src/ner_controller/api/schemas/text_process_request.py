"""Request schema for text processing."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextProcessRequest(BaseModel):
    """
    Input payload for single text processing with NER and embeddings.

    Attributes:
        text: Plain text string to process.
        entity_types: List of entity types to extract. None uses default comprehensive list.
    """

    model_config = ConfigDict(strict=True, extra='forbid')

    text: str = Field(..., min_length=1, description="Plain text string to process.")
    entity_types: list[str] | None = Field(
        None,
        description="Entity types to extract. None uses default comprehensive list.",
    )

    @field_validator('text')
    @classmethod
    def text_must_not_be_whitespace_only(cls, v: str) -> str:
        """Validate that text is not only whitespace."""
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v
