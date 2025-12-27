"""Ollama-based embedding generation implementation."""

import logging
from typing import Sequence

import httpx

from ner_controller.domain.interfaces.embedding_generator_interface import EmbeddingGeneratorInterface
from ner_controller.infrastructure.embedding.configs.ollama_embedding_generator_config import (
    OllamaEmbeddingGeneratorConfig,
)

logger = logging.getLogger(__name__)


class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."""

    pass


class OllamaEmbeddingGenerator(EmbeddingGeneratorInterface):
    """
    Generates text embeddings using Ollama API.

    Communicates with local Ollama instance to generate embeddings for text batches.
    """

    def __init__(self, config: OllamaEmbeddingGeneratorConfig) -> None:
        """
        Initialize generator with Ollama configuration.

        Args:
            config: Configuration for Ollama connection and model settings.
        """
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.request_timeout,
        )

    def generate_embeddings(self, texts: Sequence[str]) -> Sequence[Sequence[float] | None]:
        """
        Generate embeddings for a batch of texts using Ollama.

        Processes texts in batches according to configured batch_size.
        Failed embeddings return None instead of raising.

        Args:
            texts: Sequence of text strings to embed.

        Returns:
            Sequence of embedding vectors or None for failures.

        Raises:
            EmbeddingGenerationError: If the service fails completely.
        """
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self._config.batch_size):
            batch = tuple(texts[i : i + self._config.batch_size])
            batch_embeddings = self._send_batch_request(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _send_batch_request(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float] | None]:
        """
        Send a single batch request to Ollama.

        Args:
            texts: Batch of texts to embed.

        Returns:
            Sequence of embeddings or None for failures in the batch.

        Raises:
            EmbeddingGenerationError: If the request fails completely.
        """
        if not texts:
            return []

        payload = {
            "model": self._config.model,
            "input": list(texts),
        }

        try:
            response = self._client.post(self._config.embedding_endpoint, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return self._parse_response(response_data)
        except EmbeddingGenerationError:
            raise
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                logger.error(f"Ollama server error: {e}")
                return [None] * len(texts)
            else:
                raise EmbeddingGenerationError(f"Ollama request failed: {e}")
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise EmbeddingGenerationError(f"Cannot connect to Ollama service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}")
            return [None] * len(texts)

    def _parse_response(self, response_data: dict) -> Sequence[Sequence[float] | None]:
        """
        Parse Ollama API response to extract embeddings.

        Args:
            response_data: Raw JSON response from Ollama.

        Returns:
            Sequence of embedding vectors.

        Raises:
            EmbeddingGenerationError: If response format is invalid.
        """
        if not isinstance(response_data, dict):
            raise EmbeddingGenerationError(f"Invalid response type: {type(response_data)}")

        if "embeddings" not in response_data:
            raise EmbeddingGenerationError("Response missing 'embeddings' field")

        embeddings = response_data["embeddings"]
        if not isinstance(embeddings, list):
            raise EmbeddingGenerationError(f"Invalid embeddings type: {type(embeddings)}")

        # Validate each embedding is a list
        for emb in embeddings:
            if emb is not None and not isinstance(emb, list):
                raise EmbeddingGenerationError(f"Invalid embedding element type: {type(emb)}")

        # Convert each embedding list to tuple
        return [tuple(emb) if emb is not None else None for emb in embeddings]

    def __del__(self) -> None:
        """Close HTTP client on destruction."""
        if hasattr(self, "_client"):
            self._client.close()
