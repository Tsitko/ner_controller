"""Request schema for file processing."""

from pydantic import BaseModel, ConfigDict, Field


class FileProcessRequest(BaseModel):
    """
    Input payload for file processing with NER and embeddings.

    Attributes:
        file: Base64-encoded file content.
        file_name: Name of the file (for logging/metadata).
        file_id: Unique identifier for the file.
        file_path: Original path to the file (optional metadata).
        chunk_overlap: Number of characters to overlap between chunks. Default: 300.
        chunk_size: Maximum characters per chunk. Default: 3000.
        entity_types: List of entity types to extract. Default: comprehensive list.
    """

    model_config = ConfigDict(strict=True)

    file: str = Field(..., description="Base64-encoded file content.")
    file_name: str = Field(..., description="Name of the file.")
    file_id: str = Field(..., description="Unique identifier for the file.")
    file_path: str | None = Field(None, description="Original path to the file.")
    chunk_overlap: int = Field(300, description="Overlap characters between chunks.")
    chunk_size: int = Field(3000, description="Maximum characters per chunk.")
    entity_types: list[str] | None = Field(
        None,
        description="Entity types to extract. None uses default comprehensive list.",
    )
