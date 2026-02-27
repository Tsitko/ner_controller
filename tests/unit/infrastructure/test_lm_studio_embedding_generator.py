"""Unit tests for LmStudioEmbeddingGenerator."""

import path_setup

path_setup.add_src_path()


import unittest
from unittest.mock import patch

from ner_controller.infrastructure.embedding.configs.lm_studio_embedding_generator_config import (
    LmStudioEmbeddingGeneratorConfig,
)
from ner_controller.infrastructure.embedding.lm_studio_embedding_generator import (
    EmbeddingConnectionError,
    EmbeddingGenerationError,
    EmbeddingTimeoutError,
    LmStudioEmbeddingGenerator,
)


class TestLmStudioEmbeddingGenerator(unittest.TestCase):
    """Tests fallback logic for LM Studio embedding generator."""

    def setUp(self) -> None:
        """Create generator instance."""
        self.generator = LmStudioEmbeddingGenerator(
            LmStudioEmbeddingGeneratorConfig(batch_size=4)
        )

    @patch.object(LmStudioEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_splits_failed_batch(self, mock_send_batch) -> None:
        """A failed batch is split into smaller batches and recovered."""
        mock_send_batch.side_effect = [
            EmbeddingGenerationError("bad payload"),
            [(0.1,), (0.2,)],
            [(0.3,), (0.4,)],
        ]

        result = self.generator.generate_embeddings(["t1", "t2", "t3", "t4"])

        self.assertEqual(result, [(0.1,), (0.2,), (0.3,), (0.4,)])
        self.assertEqual(mock_send_batch.call_count, 3)

    @patch.object(LmStudioEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_returns_none_for_single_text_failure(self, mock_send_batch) -> None:
        """A single text that still fails returns None."""
        mock_send_batch.side_effect = EmbeddingGenerationError("bad payload")

        result = self.generator.generate_embeddings(["t1"])

        self.assertEqual(result, [None])
        self.assertEqual(mock_send_batch.call_count, 1)

    @patch.object(LmStudioEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_does_not_split_timeout_batches(self, mock_send_batch) -> None:
        """Timeout batch is skipped as-is to avoid request storms."""
        mock_send_batch.side_effect = EmbeddingTimeoutError("timed out")

        result = self.generator.generate_embeddings(["a", "b", "c", "d"])

        self.assertEqual(result, [None, None, None, None])
        self.assertEqual(mock_send_batch.call_count, 1)

    @patch.object(LmStudioEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_does_not_split_connection_errors(self, mock_send_batch) -> None:
        """Connection error batch is skipped as-is to avoid request storms."""
        mock_send_batch.side_effect = EmbeddingConnectionError("connect error")

        result = self.generator.generate_embeddings(["a", "b"])

        self.assertEqual(result, [None, None])
        self.assertEqual(mock_send_batch.call_count, 1)

    @patch.object(LmStudioEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_respects_order(self, mock_send_batch) -> None:
        """Output order matches input order across fallback splits."""
        mock_send_batch.side_effect = [
            EmbeddingGenerationError("bad payload"),
            [(1.0,)],
            EmbeddingGenerationError("bad payload"),
            [(2.0,)],
            [(3.0,)],
        ]

        result = self.generator.generate_embeddings(["a", "b", "c"])

        self.assertEqual(result, [(1.0,), (2.0,), (3.0,)])
