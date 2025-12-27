"""Response schema for file processing."""

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field


class ChunkSchema(BaseModel):
    """Text chunk with entities and embedding."""

    model_config = ConfigDict(strict=True)

    id: int = Field(..., description="Sequential chunk identifier.")
    text: str = Field(..., description="Chunk text content.")
    entities: Sequence[str] = Field(default_factory=list, description="Entities in this chunk.")
    embedding: Sequence[float] | None = Field(None, description="Embedding vector for the chunk.")


class FileProcessResponse(BaseModel):
    """
    Result of processing a file with NER and embeddings.

    Attributes:
        file_id: Unique identifier for the processed file.
        entities: All unique entities found across all chunks.
        chanks: List of text chunks with entities and embeddings.
    """

    model_config = ConfigDict(strict=True)

    file_id: str = Field(..., description="Unique identifier for the processed file.")
    entities: Sequence[str] = Field(default_factory=list, description="All unique entities.")
    chanks: Sequence[ChunkSchema] = Field(default_factory=list, description="Text chunks with data.")

