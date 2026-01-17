# Text Processing Endpoint Architecture

## Dependency Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI App                             │
│                    (ApplicationFactory)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            |
                            | creates
                            v
┌─────────────────────────────────────────────────────────────────┐
│                       TextRouter                                │
│                    (API Layer)                                  │
├─────────────────────────────────────────────────────────────────┤
│  POST /text/process                                            │
│    - handle_text_process()                                      │
│    - Validates request                                          │
│    - Converts to/from domain models                             │
│    - Maps exceptions to HTTP status codes                       │
└───────────────────────────┬─────────────────────────────────────┘
                            |
                            | uses
                            v
┌─────────────────────────────────────────────────────────────────┐
│                  TextProcessingService                          │
│                   (Domain Layer)                                │
├─────────────────────────────────────────────────────────────────┤
│  process_text(text, entity_types)                              │
│    - Validates input                                            │
│    - Orchestrates NER + embeddings                              │
│    - Returns TextProcessingResult                               │
└───────────┬───────────────────────┬─────────────────────────────┘
            |                       |
            | extracts              | generates
            v                       v
┌──────────────────────┐   ┌──────────────────────────────────┐
│ CompositeEntity      │   │  OllamaEmbeddingGenerator        │
│ Extractor            │   │                                  │
├──────────────────────┤   ├──────────────────────────────────┤
│ - GLiNER model       │   │ - qwen3-embedding:8b model       │
│ - Regex extractor    │   │ - HTTP client to Ollama          │
│ - Levenshtein dedup  │   │ - Batch processing               │
└──────────────────────┘   └──────────────────────────────────┘
```

## Request/Response Flow

```
Client
  |
  | POST /text/process
  | { "text": "...", "entity_types": ["Person", "Location"] }
  v
┌────────────────────┐
│ TextRouter         │
│ - Validates JSON   │
│ - Checks text not  │
│   empty            │
└─────┬──────────────┘
      |
      | TextProcessRequest
      v
┌────────────────────┐
│ TextProcessing     │
│ Service            │
│ - Normalizes text  │
│ - Extracts entities│
│ - Generates embed  │
└─────┬──────────────┘
      |
      | TextProcessingResult
      v
┌────────────────────┐
│ TextRouter         │
│ - Converts to      │
│   response schema  │
└─────┬──────────────┘
      |
      | TextProcessResponse
      v
Client
  { "text": "...", "entities": [...], "embedding": [...] }
```

## Component Responsibilities

### API Layer (text_router.py)
- **Responsibility**: HTTP handling, validation, response formatting
- **Does NOT**: Business logic, data processing
- **Error Handling**: Maps domain exceptions to HTTP status codes

### Domain Layer (text_processing_service.py)
- **Responsibility**: Business logic, orchestration
- **Does NOT**: HTTP concerns, data access
- **Dependencies**: Interfaces only (EntityExtractor, EmbeddingGenerator)

### Infrastructure Layer (Existing)
- **CompositeEntityExtractor**: GLiNER + Regex extraction
- **OllamaEmbeddingGenerator**: Embedding generation via Ollama
- **Levenshtein Utils**: Entity deduplication

## Key Design Decisions

1. **No Chunking**: Unlike FileProcessingService, processes entire text as single unit
2. **Reuse Existing Infrastructure**: Leverages proven entity extractor and embedding generator
3. **Immutable Results**: Frozen dataclass ensures data integrity
4. **Interface-Based Design**: Service depends on interfaces, not concrete implementations
5. **Separation of Concerns**: Clear boundaries between API, domain, and infrastructure layers

## Error Handling Strategy

```
Input Validation Errors (ValueError)
  -> HTTP 400 Bad Request

Business Logic Errors (EmbeddingGenerationError)
  -> HTTP 500 Internal Server Error

Unexpected Errors (Exception)
  -> HTTP 500 Internal Server Error
  -> Logged with details
```
