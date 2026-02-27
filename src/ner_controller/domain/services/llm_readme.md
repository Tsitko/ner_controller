# LLM Readme

## Purpose
- Domain services implementing use cases.

## Key Files
- See files in this folder for implementation details.

## Usage Notes
- Follow project naming and one-class-per-file rules.
- `FileProcessingService` applies NER on smaller sub-chunks (`NER_CHUNK_SIZE=1200`)
  with sentence-aware splitting and word-safe fallback for very long sentences.
- If embedding generation fails, chunks are still returned with `embedding = null`.

## Interfaces
- This folder contains internal modules used by higher-level layers.
