# LLM Readme

## Purpose
- Describe the architecture and service behavior for LLM-based assistance.

## Key Files
- `task.md`: requirements and architecture design notes.
- `task_text_processing.md`: text processing endpoint design.
- `AGENTS.md`: development workflow and rules.
- `src/ner_controller/main.py`: FastAPI wiring.

## Usage Notes
- Follow the architecture-test-implementation order defined in project rules.
- Keep one class per file and update README files after code changes.

## Interfaces

### POST /hallucination/check
Detect potential LLM hallucinations by comparing entities.
- Input: `request`, `response`, `entities_types`
- Output: `potential_hallucinations`, `missing_entities` arrays

### POST /file/process
Process a text file with NER and embeddings.
- Input: `file` (base64), `file_id`, `chunk_size`, `chunk_overlap`, `entity_types`
- Output: `file_id`, `entities`, `chanks` (chunks with embeddings)

### POST /text/process
Process a single text with NER and embedding (no chunking, no base64).
- Input: `text` (string), optional `entity_types` list
- Output: `text`, `entities` (deduplicated strings), `embedding` vector
- Default entity types: 23 types including Person, Organization, Location, etc.
