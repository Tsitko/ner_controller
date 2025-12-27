"""Configuration for Ollama embedding generator."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaEmbeddingGeneratorConfig:
    """Configuration values for Ollama embedding service."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen3-embedding:8b"
    embedding_endpoint: str = "/api/embed"
    batch_size: int = 20
    request_timeout: int = 60
