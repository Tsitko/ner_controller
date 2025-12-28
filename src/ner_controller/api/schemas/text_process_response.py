"""Response schema for text processing."""

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class TextProcessResponse(BaseModel):
    """
    Result of processing a single text with NER and embeddings.

    Attributes:
        text: Original input text.
        entities: Deduplicated list of extracted entity strings.
        embedding: Embedding vector for the full text.
    """

    model_config = ConfigDict(strict=True)

    text: str = Field(..., description="Original input text.")
    entities: Sequence[str] = Field(default_factory=list, description="Extracted entities.")
    embedding: Sequence[float] = Field(..., description="Embedding vector for the text.")
