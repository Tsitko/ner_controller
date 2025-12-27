"""Unit tests for OllamaEmbeddingGenerator."""

import path_setup

path_setup.add_src_path()


import json
import unittest
from unittest.mock import Mock, patch

import httpx

from ner_controller.infrastructure.embedding.configs.ollama_embedding_generator_config import (
    OllamaEmbeddingGeneratorConfig,
)
from ner_controller.infrastructure.embedding.ollama_embedding_generator import (
    EmbeddingGenerationError,
    OllamaEmbeddingGenerator,
)


class TestOllamaEmbeddingGeneratorInitialization(unittest.TestCase):
    """Tests OllamaEmbeddingGenerator initialization."""

    def test_initialize_with_config(self) -> None:
        """Generator stores configuration and creates httpx client."""
        config = OllamaEmbeddingGeneratorConfig(
            base_url="http://localhost:11434",
            model="qwen3-embedding:8b",
            batch_size=20,
            request_timeout=60,
        )

        generator = OllamaEmbeddingGenerator(config)

        self.assertEqual(generator._config, config)
        self.assertIsInstance(generator._client, httpx.Client)

    def test_httpx_client_uses_config_base_url(self) -> None:
        """HTTP client is configured with base_url from config."""
        config = OllamaEmbeddingGeneratorConfig(
            base_url="http://custom-ollama:8080",
            model="test-model",
        )

        generator = OllamaEmbeddingGenerator(config)

        self.assertEqual(generator._client.base_url, "http://custom-ollama:8080")

    def test_httpx_client_uses_config_timeout(self) -> None:
        """HTTP client is configured with timeout from config."""
        config = OllamaEmbeddingGeneratorConfig(
            base_url="http://localhost:11434",
            model="test-model",
            request_timeout=120,
        )

        generator = OllamaEmbeddingGenerator(config)

        self.assertEqual(generator._client.timeout.connect, 120)
        self.assertEqual(generator._config.request_timeout, 120)


class TestOllamaEmbeddingGeneratorGenerateEmbeddings(unittest.TestCase):
    """Tests OllamaEmbeddingGenerator.generate_embeddings method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = OllamaEmbeddingGeneratorConfig(
            base_url="http://localhost:11434",
            model="qwen3-embedding:8b",
            batch_size=2,  # Small batch for testing
        )
        self.generator = OllamaEmbeddingGenerator(self.config)

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_with_single_text(self, mock_send_batch) -> None:
        """Generates embedding for a single text."""
        mock_send_batch.return_value = [(0.1, 0.2, 0.3)]

        result = self.generator.generate_embeddings(["Single text"])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (0.1, 0.2, 0.3))
        mock_send_batch.assert_called_once_with(("Single text",))

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_with_multiple_texts(self, mock_send_batch) -> None:
        """Generates embeddings for multiple texts."""
        # Return 2 embeddings for first batch (Text 1, Text 2), 1 for second (Text 3)
        mock_send_batch.side_effect = [
            [(0.1, 0.2), (0.3, 0.4)],  # First batch of 2
            [(0.5, 0.6)],               # Second batch of 1
        ]

        result = self.generator.generate_embeddings(["Text 1", "Text 2", "Text 3"])

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], (0.1, 0.2))
        self.assertEqual(result[1], (0.3, 0.4))
        self.assertEqual(result[2], (0.5, 0.6))

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_with_empty_sequence(self, mock_send_batch) -> None:
        """Returns empty list for empty input."""
        result = self.generator.generate_embeddings([])

        self.assertEqual(result, [])
        mock_send_batch.assert_not_called()

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_splits_into_batches(self, mock_send_batch) -> None:
        """Splits texts into batches according to batch_size config."""
        # With batch_size=2, 5 texts should create 3 batches
        mock_send_batch.return_value = [
            (0.1,),
            (0.2,),
        ]

        result = self.generator.generate_embeddings(["T1", "T2", "T3", "T4", "T5"])

        # Should call _send_batch_request 3 times
        self.assertEqual(mock_send_batch.call_count, 3)

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_with_batch_failing_partially(
        self, mock_send_batch
    ) -> None:
        """Returns None for texts that fail to generate embeddings."""
        # Third text fails
        mock_send_batch.return_value = [
            (0.1, 0.2),
            (0.3, 0.4),
            None,
        ]

        result = self.generator.generate_embeddings(["Text 1", "Text 2", "Text 3"])

        self.assertEqual(result[0], (0.1, 0.2))
        self.assertEqual(result[1], (0.3, 0.4))
        self.assertIsNone(result[2])

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_raises_error_on_total_failure(
        self, mock_send_batch
    ) -> None:
        """Raises EmbeddingGenerationError when all texts fail."""
        mock_send_batch.side_effect = EmbeddingGenerationError("Total failure")

        with self.assertRaises(EmbeddingGenerationError):
            self.generator.generate_embeddings(["Text 1", "Text 2"])

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_generate_embeddings_preserves_order(self, mock_send_batch) -> None:
        """Maintains input order in output embeddings."""
        mock_send_batch.return_value = [
            (0.1,),
            (0.2,),
            (0.3,),
        ]

        result = self.generator.generate_embeddings(["A", "B", "C"])

        self.assertEqual(result[0], (0.1,))
        self.assertEqual(result[1], (0.2,))
        self.assertEqual(result[2], (0.3,))


class TestOllamaEmbeddingGeneratorSendBatchRequest(unittest.TestCase):
    """Tests OllamaEmbeddingGenerator._send_batch_request method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = OllamaEmbeddingGeneratorConfig(
            base_url="http://localhost:11434",
            model="qwen3-embedding:8b",
        )
        self.generator = OllamaEmbeddingGenerator(self.config)

    @patch("httpx.Client.post")
    def test_send_batch_request_sends_correct_format(self, mock_post) -> None:
        """Sends properly formatted request to Ollama API."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "qwen3-embedding:8b",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }
        mock_post.return_value = mock_response

        result = self.generator._send_batch_request(["Text 1", "Text 2"])

        # Verify request was made to correct endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], "/api/embed")

        # Verify request body structure
        request_body = call_args[1]["json"]
        self.assertEqual(request_body["model"], "qwen3-embedding:8b")
        self.assertEqual(request_body["input"], ["Text 1", "Text 2"])

    @patch("httpx.Client.post")
    def test_send_batch_request_parses_successful_response(
        self, mock_post
    ) -> None:
        """Parses successful Ollama response correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "qwen3-embedding:8b",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
        mock_post.return_value = mock_response

        result = self.generator._send_batch_request(["Text 1", "Text 2"])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (0.1, 0.2, 0.3))
        self.assertEqual(result[1], (0.4, 0.5, 0.6))

    @patch("httpx.Client.post")
    def test_send_batch_request_handles_timeout(self, mock_post) -> None:
        """Raises EmbeddingGenerationError on timeout."""
        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        with self.assertRaises(EmbeddingGenerationError):
            self.generator._send_batch_request(["Text 1"])

    @patch("httpx.Client.post")
    def test_send_batch_request_handles_connection_error(self, mock_post) -> None:
        """Raises EmbeddingGenerationError on connection failure."""
        mock_post.side_effect = httpx.ConnectError("Connection failed")

        with self.assertRaises(EmbeddingGenerationError):
            self.generator._send_batch_request(["Text 1"])

    @patch("httpx.Client.post")
    def test_send_batch_request_handles_http_500(self, mock_post) -> None:
        """Returns None for all texts when Ollama returns 500."""
        import httpx
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        # Make raise_for_status raise HTTPStatusError
        mock_post.return_value = mock_response
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=mock_response
        )

        result = self.generator._send_batch_request(["Text 1", "Text 2"])

        # All embeddings should be None
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    @patch("httpx.Client.post")
    def test_send_batch_request_handles_malformed_response(self, mock_post) -> None:
        """Raises EmbeddingGenerationError for malformed response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"invalid": "structure"}
        mock_post.return_value = mock_response

        with self.assertRaises(EmbeddingGenerationError):
            self.generator._send_batch_request(["Text 1"])

    @patch("httpx.Client.post")
    def test_send_batch_request_handles_missing_embeddings_key(
        self, mock_post
    ) -> None:
        """Raises EmbeddingGenerationError when response lacks embeddings key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model": "qwen3-embedding:8b"}
        mock_post.return_value = mock_response

        with self.assertRaises(EmbeddingGenerationError):
            self.generator._send_batch_request(["Text 1"])


class TestOllamaEmbeddingGeneratorParseResponse(unittest.TestCase):
    """Tests OllamaEmbeddingGenerator._parse_response method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = OllamaEmbeddingGeneratorConfig(
            base_url="http://localhost:11434",
            model="qwen3-embedding:8b",
        )
        self.generator = OllamaEmbeddingGenerator(self.config)

    def test_parse_valid_response(self) -> None:
        """Parses valid Ollama response correctly."""
        response_data = {
            "model": "qwen3-embedding:8b",
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }

        result = self.generator._parse_response(response_data)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (0.1, 0.2, 0.3))
        self.assertEqual(result[1], (0.4, 0.5, 0.6))

    def test_parse_response_with_single_embedding(self) -> None:
        """Parses response with single embedding vector."""
        response_data = {
            "model": "qwen3-embedding:8b",
            "embeddings": [[0.1, 0.2, 0.3]],
        }

        result = self.generator._parse_response(response_data)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (0.1, 0.2, 0.3))

    def test_parse_response_with_large_embeddings(self) -> None:
        """Parses response with high-dimensional embeddings."""
        large_embedding = list(float(i) * 0.01 for i in range(1000))
        response_data = {
            "model": "qwen3-embedding:8b",
            "embeddings": [large_embedding],
        }

        result = self.generator._parse_response(response_data)

        self.assertEqual(len(result[0]), 1000)

    def test_parse_response_raises_error_for_missing_embeddings(self) -> None:
        """Raises EmbeddingGenerationError when embeddings key is missing."""
        response_data = {"model": "qwen3-embedding:8b"}

        with self.assertRaises(EmbeddingGenerationError):
            self.generator._parse_response(response_data)

    def test_parse_response_raises_error_for_invalid_embeddings_type(
        self,
    ) -> None:
        """Raises EmbeddingGenerationError when embeddings is not a list."""
        response_data = {
            "model": "qwen3-embedding:8b",
            "embeddings": "not-a-list",
        }

        with self.assertRaises(EmbeddingGenerationError):
            self.generator._parse_response(response_data)

    def test_parse_response_raises_error_for_invalid_embedding_element(
        self,
    ) -> None:
        """Raises EmbeddingGenerationError when embedding is not a list."""
        response_data = {
            "model": "qwen3-embedding:8b",
            "embeddings": ["not-a-list"],
        }

        with self.assertRaises(EmbeddingGenerationError):
            self.generator._parse_response(response_data)

    def test_parse_response_converts_to_tuples(self) -> None:
        """Converts embedding lists to tuples for immutability."""
        response_data = {
            "model": "qwen3-embedding:8b",
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        result = self.generator._parse_response(response_data)

        self.assertIsInstance(result[0], tuple)
        self.assertIsInstance(result[1], tuple)


class TestOllamaEmbeddingGeneratorCleanup(unittest.TestCase):
    """Tests OllamaEmbeddingGenerator resource cleanup."""

    def test_del_closes_http_client(self) -> None:
        """Destructor closes HTTP client."""
        config = OllamaEmbeddingGeneratorConfig(
            base_url="http://localhost:11434",
            model="qwen3-embedding:8b",
        )
        generator = OllamaEmbeddingGenerator(config)

        client = generator._client

        # Trigger cleanup
        del generator

        # Client should be closed
        self.assertTrue(client.is_closed)


class TestEmbeddingGenerationError(unittest.TestCase):
    """Tests EmbeddingGenerationError exception."""

    def test_is_exception_subclass(self) -> None:
        """EmbeddingGenerationError is an Exception subclass."""
        self.assertTrue(issubclass(EmbeddingGenerationError, Exception))

    def test_can_be_raised_with_message(self) -> None:
        """Exception can be raised with custom message."""
        with self.assertRaises(EmbeddingGenerationError) as context:
            raise EmbeddingGenerationError("Custom error message")

        self.assertEqual(str(context.exception), "Custom error message")

    def test_can_be_raised_without_message(self) -> None:
        """Exception can be raised without custom message."""
        with self.assertRaises(EmbeddingGenerationError):
            raise EmbeddingGenerationError()

    def test_can_be_caught_as_generic_exception(self) -> None:
        """Exception can be caught as generic Exception."""
        try:
            raise EmbeddingGenerationError("Test")
        except Exception as e:
            self.assertIsInstance(e, EmbeddingGenerationError)


class TestOllamaEmbeddingGeneratorIntegrationBehavior(unittest.TestCase):
    """Tests integration behavior with batching and error handling."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = OllamaEmbeddingGeneratorConfig(
            base_url="http://localhost:11434",
            model="qwen3-embedding:8b",
            batch_size=2,
        )
        self.generator = OllamaEmbeddingGenerator(self.config)

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_large_text_batch_splits_correctly(self, mock_send_batch) -> None:
        """Large batch splits into correct number of sub-batches."""
        # batch_size=2, so 7 texts = 4 batches (2+2+2+1)
        mock_send_batch.side_effect = [
            [(0.1,), (0.2,)],  # First batch of 2
            [(0.3,), (0.4,)],  # Second batch of 2
            [(0.5,), (0.6,)],  # Third batch of 2
            [(0.7,)],           # Fourth batch of 1
        ]

        result = self.generator.generate_embeddings(
            [f"Text {i}" for i in range(7)]
        )

        self.assertEqual(mock_send_batch.call_count, 4)
        self.assertEqual(len(result), 7)

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_batch_size_boundary_condition(self, mock_send_batch) -> None:
        """Handles exact multiple of batch_size correctly."""
        # batch_size=2, exactly 4 texts = 2 batches
        mock_send_batch.return_value = [(0.1,), (0.2,)]

        result = self.generator.generate_embeddings(["T1", "T2", "T3", "T4"])

        self.assertEqual(mock_send_batch.call_count, 2)

    @patch.object(OllamaEmbeddingGenerator, "_send_batch_request")
    def test_partial_batch_failure_handling(self, mock_send_batch) -> None:
        """Handles failure of one batch among multiple."""
        # First batch succeeds, second fails
        mock_send_batch.side_effect = [
            [(0.1,), (0.2,)],  # Batch 1
            EmbeddingGenerationError("Batch 2 failed"),  # Batch 2
        ]

        with self.assertRaises(EmbeddingGenerationError):
            self.generator.generate_embeddings(["T1", "T2", "T3", "T4"])
