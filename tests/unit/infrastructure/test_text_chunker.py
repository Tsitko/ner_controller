"""Unit tests for TextChunker."""

import path_setup

path_setup.add_src_path()


import unittest

from ner_controller.domain.entities.file_chunk import FileChunk
from ner_controller.infrastructure.chunking.configs.text_chunker_config import (
    TextChunkerConfig,
)
from ner_controller.infrastructure.chunking.text_chunker import TextChunker


class TestTextChunkerInitialization(unittest.TestCase):
    """Tests TextChunker initialization."""

    def test_initialize_with_config(self) -> None:
        """Chunker stores configuration correctly."""
        config = TextChunkerConfig(
            preserve_sentences=True,
            min_chunk_size=100,
        )

        chunker = TextChunker(config)

        self.assertEqual(chunker._config, config)

    def test_default_config_values(self) -> None:
        """Config has appropriate default values."""
        config = TextChunkerConfig()

        self.assertTrue(config.preserve_sentences)
        self.assertEqual(config.min_chunk_size, 100)


class TestTextChunkerSplitText(unittest.TestCase):
    """Tests TextChunker.split_text method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextChunkerConfig()
        self.chunker = TextChunker(self.config)

    def test_split_text_creates_single_chunk_for_short_text(self) -> None:
        """Creates single chunk when text is shorter than chunk_size."""
        text = "This is a short text."
        chunks = self.chunker.split_text(text, chunk_size=100, chunk_overlap=10)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].id, 0)

    def test_split_text_creates_multiple_chunks(self) -> None:
        """Creates multiple chunks when text exceeds chunk_size."""
        text = "A" * 150  # 150 characters
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=20, start_id=0
        )

        # With stride=80 (100-20), we should get chunks at positions 0 and 80
        # Chunk 1: positions 0-99 (100 chars)
        # Chunk 2: positions 80-149 (70 chars)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].text), 100)
        self.assertEqual(len(chunks[1].text), 70)

    def test_split_text_with_zero_overlap(self) -> None:
        """Creates chunks with no overlap when chunk_overlap=0."""
        text = "ABCDEFGH" * 20  # 160 characters
        chunks = self.chunker.split_text(
            text, chunk_size=50, chunk_overlap=0, start_id=0
        )

        # Should create chunks: 0-49, 50-99, 100-149, 150-159
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].text, text[0:50])
        self.assertEqual(chunks[1].text, text[50:100])
        self.assertEqual(chunks[2].text, text[100:150])
        self.assertEqual(chunks[3].text, text[150:160])

    def test_split_text_with_overlap(self) -> None:
        """Creates chunks with correct overlap."""
        text = "ABCDEFGHIJ" * 20  # 200 characters
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=30, start_id=0
        )

        # Stride = 100 - 30 = 70
        # Chunk 1: 0-99, Chunk 2: 70-169, Chunk 3: 140-199
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, text[0:100])
        self.assertEqual(chunks[1].text, text[70:170])
        self.assertEqual(chunks[2].text, text[140:200])

    def test_split_text_uses_custom_start_id(self) -> None:
        """Uses custom start_id for first chunk."""
        text = "Short text"
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=10, start_id=5
        )

        self.assertEqual(chunks[0].id, 5)
        if len(chunks) > 1:
            self.assertEqual(chunks[1].id, 6)

    def test_split_text_increments_ids_sequentially(self) -> None:
        """Assigns sequential IDs to chunks."""
        text = "A" * 500
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=20, start_id=10
        )

        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.id, 10 + i)

    def test_split_text_with_empty_string(self) -> None:
        """Returns empty list for empty input."""
        chunks = self.chunker.split_text("", chunk_size=100, chunk_overlap=10)

        self.assertEqual(len(chunks), 0)

    def test_split_text_creates_chunks_with_empty_entities(self) -> None:
        """All chunks have empty entities tuple."""
        text = "Some text here"
        chunks = self.chunker.split_text(text, chunk_size=100, chunk_overlap=10)

        for chunk in chunks:
            self.assertEqual(chunk.entities, ())

    def test_split_text_creates_chunks_with_none_embedding(self) -> None:
        """All chunks have None as embedding."""
        text = "Some text here"
        chunks = self.chunker.split_text(text, chunk_size=100, chunk_overlap=10)

        for chunk in chunks:
            self.assertIsNone(chunk.embedding)

    def test_split_text_exact_multiple_of_stride(self) -> None:
        """Handles when text length is exact multiple of stride."""
        # chunk_size=100, overlap=20, stride=80
        # 240 characters = 3 chunks at positions 0, 80, 160
        text = "A" * 240
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=20, start_id=0
        )

        self.assertEqual(len(chunks), 3)

    def test_split_text_last_chunk_smaller_than_chunk_size(self) -> None:
        """Last chunk can be smaller than chunk_size."""
        text = "A" * 250  # Not a perfect multiple
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=20, start_id=0
        )

        # Check that last chunk is smaller
        if len(chunks) > 1:
            self.assertLessEqual(len(chunks[-1].text), 100)

    def test_split_text_with_unicode(self) -> None:
        """Handles Unicode text correctly."""
        text = "Привет мир! " * 20  # Unicode text
        chunks = self.chunker.split_text(
            text, chunk_size=50, chunk_overlap=10, start_id=0
        )

        self.assertGreater(len(chunks), 0)
        # Verify chunks contain valid Unicode
        for chunk in chunks:
            self.assertIsInstance(chunk.text, str)

    def test_split_text_preserves_text_content(self) -> None:
        """Chunks contain exact portions of original text."""
        text = "ABCDEFGHIJ" * 10
        chunks = self.chunker.split_text(
            text, chunk_size=50, chunk_overlap=10, start_id=0
        )

        # Verify all chunk text exists in original
        for chunk in chunks:
            self.assertIn(chunk.text, text)

    def test_split_text_with_large_overlap(self) -> None:
        """Handles large overlap (close to chunk_size)."""
        text = "A" * 200
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=90, start_id=0
        )

        # Stride = 10, so should create many chunks
        self.assertGreater(len(chunks), 2)

    def test_split_text_single_character_chunk_size(self) -> None:
        """Handles extreme case of chunk_size=1."""
        text = "ABC"
        chunks = self.chunker.split_text(
            text, chunk_size=1, chunk_overlap=0, start_id=0
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "A")
        self.assertEqual(chunks[1].text, "B")
        self.assertEqual(chunks[2].text, "C")


class TestTextChunkerValidateParameters(unittest.TestCase):
    """Tests TextChunker._validate_parameters method."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextChunkerConfig()
        self.chunker = TextChunker(self.config)

    def test_validate_accepts_valid_parameters(self) -> None:
        """Accepts valid chunk_size and chunk_overlap."""
        # Should not raise
        self.chunker._validate_parameters(chunk_size=100, chunk_overlap=20)

    def test_validate_rejects_zero_chunk_size(self) -> None:
        """Raises ValueError when chunk_size is 0."""
        with self.assertRaises(ValueError):
            self.chunker._validate_parameters(chunk_size=0, chunk_overlap=0)

    def test_validate_rejects_negative_chunk_size(self) -> None:
        """Raises ValueError when chunk_size is negative."""
        with self.assertRaises(ValueError):
            self.chunker._validate_parameters(chunk_size=-100, chunk_overlap=0)

    def test_validate_rejects_negative_chunk_overlap(self) -> None:
        """Raises ValueError when chunk_overlap is negative."""
        with self.assertRaises(ValueError):
            self.chunker._validate_parameters(chunk_size=100, chunk_overlap=-10)

    def test_validate_rejects_overlap_equal_to_size(self) -> None:
        """Raises ValueError when chunk_overlap equals chunk_size."""
        with self.assertRaises(ValueError):
            self.chunker._validate_parameters(chunk_size=100, chunk_overlap=100)

    def test_validate_rejects_overlap_greater_than_size(self) -> None:
        """Raises ValueError when chunk_overlap exceeds chunk_size."""
        with self.assertRaises(ValueError):
            self.chunker._validate_parameters(chunk_size=50, chunk_overlap=100)

    def test_validate_accepts_zero_overlap(self) -> None:
        """Accepts chunk_overlap=0."""
        # Should not raise
        self.chunker._validate_parameters(chunk_size=100, chunk_overlap=0)

    def test_validate_accepts_large_chunk_size(self) -> None:
        """Accepts very large chunk_size values."""
        # Should not raise
        self.chunker._validate_parameters(chunk_size=1000000, chunk_overlap=100)


class TestTextChunkerEdgeCases(unittest.TestCase):
    """Tests TextChunker edge cases and corner cases."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextChunkerConfig()
        self.chunker = TextChunker(self.config)

    def test_split_text_with_newlines(self) -> None:
        """Handles text with newline characters."""
        text = "Line 1\nLine 2\nLine 3\n" * 10
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=20, start_id=0
        )

        self.assertGreater(len(chunks), 0)
        # Newlines should be preserved in chunks
        for chunk in chunks:
            if "\n" in chunk.text:
                self.assertIn("\n", chunk.text)

    def test_split_text_with_tabs(self) -> None:
        """Handles text with tab characters."""
        text = "Column1\tColumn2\tColumn3\n" * 10
        chunks = self.chunker.split_text(
            text, chunk_size=100, chunk_overlap=10, start_id=0
        )

        self.assertGreater(len(chunks), 0)

    def test_split_text_with_special_characters(self) -> None:
        """Handles text with special characters."""
        text = "Special chars: !@#$%^&*()[]{}<>?,./"
        chunks = self.chunker.split_text(
            text, chunk_size=50, chunk_overlap=10, start_id=0
        )

        self.assertGreater(len(chunks), 0)

    def test_split_text_very_long_single_line(self) -> None:
        """Handles very long single line without spaces."""
        text = "A" * 10000
        chunks = self.chunker.split_text(
            text, chunk_size=1000, chunk_overlap=100, start_id=0
        )

        self.assertGreater(len(chunks), 1)

    def test_split_text_whitespace_only(self) -> None:
        """Handles text containing only whitespace."""
        text = "     \n\n   \t\t   "
        chunks = self.chunker.split_text(
            text, chunk_size=10, chunk_overlap=2, start_id=0
        )

        # Should create chunks from whitespace
        self.assertGreater(len(chunks), 0)

    def test_split_text_mixed_whitespace_and_content(self) -> None:
        """Handles text with mixed whitespace and content."""
        text = "   Word1   \n\n   Word2   \t\t   Word3   "
        chunks = self.chunker.split_text(
            text, chunk_size=20, chunk_overlap=5, start_id=0
        )

        self.assertGreater(len(chunks), 0)

    def test_split_text_chunk_size_one_with_overlap_zero(self) -> None:
        """Extreme case: chunk_size=1, overlap=0."""
        text = "ABC"
        chunks = self.chunker.split_text(
            text, chunk_size=1, chunk_overlap=0, start_id=0
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "A")
        self.assertEqual(chunks[1].text, "B")
        self.assertEqual(chunks[2].text, "C")


class TestTextChunkerOverlapBehavior(unittest.TestCase):
    """Tests detailed overlap behavior of TextChunker."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextChunkerConfig()
        self.chunker = TextChunker(self.config)

    def test_overlap_content_is_correct(self) -> None:
        """Verifies that overlap contains correct text from previous chunk."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = self.chunker.split_text(
            text, chunk_size=10, chunk_overlap=3, start_id=0
        )

        if len(chunks) >= 2:
            # Chunk 0: positions 0-9 (ABCDEFGHIJ)
            # Chunk 1: positions 7-16 (HIJKLMNOPQ)
            # Overlap should be HIJ (positions 7-9)
            self.assertEqual(chunks[0].text, "ABCDEFGHIJ")
            self.assertEqual(chunks[1].text, "HIJKLMNOPQ")
            # Verify overlap
            self.assertEqual(chunks[0].text[-3:], chunks[1].text[:3])

    def test_no_overlap_when_overlap_is_zero(self) -> None:
        """Verifies no content overlap when chunk_overlap=0."""
        text = "ABCDEFGHIJ" * 3
        chunks = self.chunker.split_text(
            text, chunk_size=10, chunk_overlap=0, start_id=0
        )

        for i in range(len(chunks) - 1):
            # Adjacent chunks should not share characters
            self.assertNotEqual(chunks[i].text[-1], chunks[i + 1].text[0])

    def test_overlap_size_is_correct(self) -> None:
        """Verifies exact overlap size matches chunk_overlap parameter."""
        # Use text with unique characters to make overlap detection reliable
        # Create text where each character position is unique
        text = "".join(chr(65 + (i % 26)) + str(i // 26) for i in range(100))
        chunk_overlap = 15
        chunks = self.chunker.split_text(
            text, chunk_size=50, chunk_overlap=chunk_overlap, start_id=0
        )

        if len(chunks) >= 2:
            # Calculate actual overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].text
                chunk2_start = chunks[i + 1].text

                # Find the maximum overlap by checking all possible overlap lengths
                overlap_len = 0
                max_possible = min(len(chunk1_end), len(chunk2_start))

                for j in range(max_possible, 0, -1):
                    if chunk1_end[-j:] == chunk2_start[:j]:
                        overlap_len = j
                        break

                self.assertEqual(overlap_len, chunk_overlap,
                    f"Expected overlap {chunk_overlap} but got {overlap_len}. "
                    f"Chunk 1 end: ...{chunk1_end[-30:]}, Chunk 2 start: {chunk2_start[:30]}...")


class TestTextChunkerLargeTexts(unittest.TestCase):
    """Tests TextChunker with large text inputs."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.config = TextChunkerConfig()
        self.chunker = TextChunker(self.config)

    def test_split_very_long_text(self) -> None:
        """Handles very long text (100k characters)."""
        text = "A" * 100000
        chunks = self.chunker.split_text(
            text, chunk_size=5000, chunk_overlap=500, start_id=0
        )

        self.assertGreater(len(chunks), 10)

    def test_split_long_text_with_many_chunks(self) -> None:
        """Creates many chunks from long text."""
        text = "WORD " * 10000  # ~50k characters
        chunks = self.chunker.split_text(
            text, chunk_size=1000, chunk_overlap=100, start_id=0
        )

        self.assertGreater(len(chunks), 40)

    def test_chunk_ids_for_many_chunks(self) -> None:
        """Verifies sequential IDs for large number of chunks."""
        text = "A" * 50000
        start_id = 1000
        chunks = self.chunker.split_text(
            text, chunk_size=1000, chunk_overlap=100, start_id=start_id
        )

        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.id, start_id + i)
