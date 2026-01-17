# Task: Text Processing Endpoint

## Overview
Add a new endpoint to process a single text with NER and embedding generation (no chunking, no base64).

## Request Format
```json
{
  "text": "Alice visited Paris and met with OpenAI researchers.",
  "entity_types": ["Person", "Location", "Organization"]
}
```

If `entity_types` is not specified, use default entity types from `FileProcessingService.DEFAULT_ENTITY_TYPES`.

## Response Format
```json
{
  "text": "Alice visited Paris and met with OpenAI researchers.",
  "entities": ["Alice", "Paris", "OpenAI", "researchers"],
  "embedding": [0.1, 0.2, 0.3, ...]
}
```

## Requirements

### Endpoint
- Route: `POST /text/process`
- Input: plain text (string), optional entity_types list
- Output: original text, list of extracted entities (deduplicated strings), embedding vector

### Processing
1. Extract entities from the full text (no chunking)
2. Generate a single embedding for the full text
3. Return text, entities, and embedding

### Entity Extraction
- Reuse existing `CompositeEntityExtractor` (GLiNER + regex)
- Deduplicate using Levenshtein distance (threshold <= 2)
- Return as plain list of strings (names only)

### Embedding Generation
- Reuse existing `OllamaEmbeddingGenerator`
- Use `qwen3-embedding:8b` model via Ollama
- Return full embedding vector as list of floats

### Error Handling
- Empty text: HTTP 400
- Embedding generation failure: HTTP 500
- NER failure: HTTP 500

## Design Notes
- No need for `TextChunker` (single text processing)
- No need for base64 decoding (plain text input)
- Reuse `EntityExtractorInterface` and `EmbeddingGeneratorInterface`
- Create new service: `TextProcessingService`
- Follow existing architecture patterns from `FileProcessingService`

---

## Architecture Design

### Created Structure

```
src/ner_controller/
├── domain/
│   ├── entities/
│   │   └── text_processing_result.py        # NEW: Domain entity for result
│   ├── interfaces/
│   │   ├── embedding_generator_interface.py # EXISTING: Reused
│   │   └── entity_extractor_interface.py    # EXISTING: Reused
│   └── services/
│       ├── text_processing_service.py       # NEW: Core business logic
│       ├── file_processing_service.py       # EXISTING: Reference
│       └── levenshtein_utils.py             # EXISTING: Reused for deduplication
├── infrastructure/
│   └── ner/
│       ├── composite_entity_extractor.py    # EXISTING: Reused
│       └── ollama_embedding_generator.py    # EXISTING: Reused
├── api/
│   ├── configs/
│   │   └── text_router_config.py            # NEW: Router configuration
│   ├── schemas/
│   │   ├── text_process_request.py          # NEW: Request schema
│   │   └── text_process_response.py         # NEW: Response schema
│   └── routers/
│       └── text_router.py                   # NEW: HTTP endpoint handler
└── application/
    └── application_factory.py               # UPDATED: Wire new router
```

### Components Overview

#### 1. Domain Layer

**`TextProcessingResult`** (`domain/entities/text_processing_result.py`)
- Immutable dataclass (frozen)
- Fields:
  - `text: str` - Original input text
  - `entities: tuple[str, ...]` - Deduplicated entity list
  - `embedding: tuple[float, ...]` - Embedding vector
- Responsibility: Pure data container for processing results

**`TextProcessingService`** (`domain/services/text_processing_service.py`)
- Dependencies:
  - `EntityExtractorInterface` - Extracts entities from text
  - `EmbeddingGeneratorInterface` - Generates embeddings
- Methods:
  - `process_text(text: str, entity_types: Sequence[str] | None) -> TextProcessingResult`
  - `_validate_and_normalize_text(text: str) -> str`
  - `_extract_entities(text: str, entity_types: list[str]) -> list[str]`
  - `_generate_embedding(text: str) -> list[float]`
- Reuses `DEFAULT_ENTITY_TYPES` from `FileProcessingService`
- No chunking logic (simpler than FileProcessingService)

#### 2. API Layer

**`TextProcessRequest`** (`api/schemas/text_process_request.py`)
- Pydantic BaseModel with strict validation
- Fields:
  - `text: str` - Required, min_length=1
  - `entity_types: list[str] | None` - Optional
- Uses `ConfigDict(strict=True)` for validation

**`TextProcessResponse`** (`api/schemas/text_process_response.py`)
- Pydantic BaseModel
- Fields:
  - `text: str` - Original text
  - `entities: Sequence[str]` - Extracted entities
  - `embedding: Sequence[float]` - Embedding vector
- Uses `default_factory=list` for optional sequences

**`TextRouterConfig`** (`api/configs/text_router_config.py`)
- Frozen dataclass
- Fields:
  - `prefix: str = "/text"`
  - `tags: tuple[str, ...] = ("text-processing",)`

**`TextRouter`** (`api/routers/text_router.py`)
- Methods:
  - `create_router() -> APIRouter` - Builds FastAPI router
  - `handle_text_process(request: TextProcessRequest) -> TextProcessResponse` - Endpoint handler
  - `_validate_request(request: TextProcessRequest) -> None` - Request validation
  - `_convert_to_response(result: TextProcessingResult) -> TextProcessResponse` - Domain to API mapping
- Error mapping:
  - `ValueError` -> HTTP 400
  - `Exception` -> HTTP 500

#### 3. Application Layer

**`ApplicationFactory`** (Updated in `application/application_factory.py`)
- Added imports:
  - `TextRouterConfig`
  - `TextRouter`
  - `TextProcessingService`
- Added constructor parameter: `text_processing_service: Optional[TextProcessingService]`
- Added method: `_build_text_processing_service() -> TextProcessingService`
- Updated `create_app()` to instantiate and include text router
- Reuses existing `create_entity_extractor()` method

### Implementation Recommendations

#### 1. Implementation Order

1. **Start with domain entities** (text_processing_result.py)
   - Simple dataclass, no dependencies
   - Easy to test

2. **Implement request/response schemas** (text_process_request.py, text_process_response.py)
   - Pydantic models are straightforward
   - Can validate schema design early

3. **Implement TextProcessingService** (text_processing_service.py)
   - Core business logic
   - Use existing dependencies (entity extractor, embedding generator)
   - Focus on proper error handling

4. **Implement TextRouter** (text_router.py)
   - HTTP layer
   - Map domain exceptions to HTTP status codes
   - Test request validation

5. **Wire in ApplicationFactory** (application_factory.py)
   - Add dependency injection
   - Register router with FastAPI

6. **Integration testing**
   - Test complete flow end-to-end
   - Verify error handling

#### 2. Key Implementation Details

**TextProcessingService._extract_entities():**
```python
# Entities are already deduplicated by CompositeEntityExtractor
entities = self._entity_extractor.extract(text, entity_types)
return list(entities)
```

**TextProcessingService._generate_embedding():**
```python
# generate_embeddings returns Sequence[Sequence[float] | None]
embeddings = self._embedding_generator.generate_embeddings([text])
if not embeddings or embeddings[0] is None:
    raise EmbeddingGenerationError("Failed to generate embedding")
return list(embeddings[0])
```

**Error Handling in TextRouter:**
- Catch `ValueError` from service (validation errors) -> HTTP 400
- Catch `EmbeddingGenerationError` specifically -> HTTP 500 with clear message
- Catch generic `Exception` -> HTTP 500 with "Processing failed" message

#### 3. Required Dependencies

No new dependencies required. Uses existing:
- `fastapi` - Web framework
- `pydantic` - Schema validation
- `httpx` - HTTP client (via OllamaEmbeddingGenerator)

#### 4. Testing Strategy

**Unit Tests:**
- Test `TextProcessingService` with mocked dependencies
- Test entity extraction flow
- Test embedding generation flow
- Test validation logic

**Integration Tests:**
- Test complete endpoint flow
- Test error scenarios (empty text, invalid entity_types)
- Test with real Ollama and GLiNER services

**Test Cases:**
1. Valid request with default entity types
2. Valid request with custom entity types
3. Empty text (should return 400)
4. Text with no entities (should return empty list)
5. Text with entities requiring deduplication
6. Embedding generation failure handling

### Considerations

**Edge Cases:**
- Empty string after stripping: Should raise ValueError
- Entity types list is empty: Should raise HTTP 400
- Text with only whitespace: Should raise ValueError
- Embedding service unavailable: Should raise HTTP 500
- NER service fails: Should propagate exception

**Performance:**
- No chunking means single pass through NER model
- Single embedding generation (fast)
- No need for batch processing
- Response time should be < 2 seconds for typical texts

**Security:**
- Text length: Consider adding max_length constraint (e.g., 10000 chars)
- Input sanitization: Strip leading/trailing whitespace
- Entity types validation: Ensure they are valid strings

**API Design:**
- Follows existing patterns from FileRouter
- Consistent error response format
- Clear separation of concerns (domain vs API layer)
- Immutable domain entities (frozen dataclass)

**Future Extensions:**
- Could add text normalization options
- Could add entity confidence scores
- Could add support for multiple embedding models
- Could add caching for repeated texts
