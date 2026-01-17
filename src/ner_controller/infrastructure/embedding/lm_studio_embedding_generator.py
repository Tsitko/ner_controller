"""LM Studio-based embedding generation implementation with OpenAI-compatible API."""

import logging
from typing import Sequence

import httpx

from ner_controller.domain.interfaces.embedding_generator_interface import EmbeddingGeneratorInterface
from ner_controller.infrastructure.embedding.configs.lm_studio_embedding_generator_config import (
    LmStudioEmbeddingGeneratorConfig,
)

logger = logging.getLogger(__name__)


class EmbeddingGenerationError(Exception):
    """Raised when embedding generation fails."""

    pass


class LmStudioEmbeddingGenerator(EmbeddingGeneratorInterface):
    """
    Generates text embeddings using LM Studio with OpenAI-compatible API.

    Communicates with LM Studio instance to generate embeddings for text batches.
    """

    def __init__(self, config: LmStudioEmbeddingGeneratorConfig) -> None:
        """
        Initialize generator with LM Studio configuration.

        Args:
            config: Configuration for LM Studio connection and model settings.
        """
        self._config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=config.request_timeout,
        )

    def generate_embeddings(self, texts: Sequence[str]) -> Sequence[Sequence[float] | None]:
        """
        Generate embeddings for a batch of texts using LM Studio.

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
        Send a single batch request to LM Studio.

        Args:
            texts: Batch of texts to embed.

        Returns:
            Sequence of embeddings or None for failures in the batch.

        Raises:
            EmbeddingGenerationError: If the request fails completely.
        """
        if not texts:
            return []

        # OpenAI-compatible API format
        payload = {
            "model": self._config.model,
            "input": list(texts) if len(texts) > 1 else texts[0],
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
                logger.error(f"LM Studio server error: {e}")
                return [None] * len(texts)
            else:
                raise EmbeddingGenerationError(f"LM Studio request failed: {e}")
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise EmbeddingGenerationError(f"Cannot connect to LM Studio service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}")
            return [None] * len(texts)

    def _parse_response(self, response_data: dict) -> Sequence[Sequence[float] | None]:
        """
        Parse OpenAI-compatible API response to extract embeddings.

        Args:
            response_data: Raw JSON response from LM Studio.

        Returns:
            Sequence of embedding vectors.

        Raises:
            EmbeddingGenerationError: If response format is invalid.
        """
        if not isinstance(response_data, dict):
            raise EmbeddingGenerationError(f"Invalid response type: {type(response_data)}")

        if "data" not in response_data:
            raise EmbeddingGenerationError("Response missing 'data' field")

        data = response_data["data"]
        if not isinstance(data, list):
            raise EmbeddingGenerationError(f"Invalid data type: {type(data)}")

        # Sort by index to ensure correct order
        data.sort(key=lambda x: x.get("index", 0))

        embeddings = []
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingGenerationError(f"Invalid data item type: {type(item)}")

            if "embedding" not in item:
                raise EmbeddingGenerationError("Data item missing 'embedding' field")

            embedding = item["embedding"]
            if not isinstance(embedding, list):
                raise EmbeddingGenerationError(f"Invalid embedding type: {type(embedding)}")

            if not all(isinstance(x, (int, float)) for x in embedding):
                raise EmbeddingGenerationError("Embedding contains non-numeric values")

            embeddings.append(tuple(embedding))

        return embeddings

    def __del__(self) -> None:
        """Close HTTP client on destruction."""
        if hasattr(self, "_client"):
            self._client.close()
