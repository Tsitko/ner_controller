# LLM Readme

## Purpose
- Describe the architecture and service behavior for LLM-based assistance.

## Key Files
- `task.md`: requirements and architecture design notes.
- `AGENTS.md`: development workflow and rules.
- `src/ner_controller/main.py`: FastAPI wiring.

## Usage Notes
- Follow the architecture-test-implementation order defined in project rules.
- Keep one class per file and update README files after code changes.

## Interfaces
- POST `/hallucination/check` with request, response, and entities_types.
- Returns potential_hallucinations and missing_entities arrays.
