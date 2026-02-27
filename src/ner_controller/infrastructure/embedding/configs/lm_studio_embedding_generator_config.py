"""Configuration for LM Studio embedding generator."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LmStudioEmbeddingGeneratorConfig:
    """Configuration values for LM Studio embedding service with OpenAI-compatible API."""

    base_url: str = "http://localhost:1234"
    model: str = "text-embedding-qwen3-embedding-8b"
    embedding_endpoint: str = "/v1/embeddings"
    batch_size: int = 4
    request_timeout: int = 180
