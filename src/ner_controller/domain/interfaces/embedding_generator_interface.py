"""Interface for text embedding generation."""

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingGeneratorInterface(ABC):
    """Contract for generating text embeddings."""

    @abstractmethod
    def generate_embeddings(self, texts: Sequence[str]) -> Sequence[Sequence[float] | None]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: Sequence of text strings to embed.

        Returns:
            Sequence of embedding vectors (each as a sequence of floats).
            Returns None for any text that fails to generate an embedding.

        Raises:
            EmbeddingGenerationError: If the embedding service fails completely.
        """
        raise NotImplementedError
