# Task

Разработать новый эндоинт для получения всех ключевых сущностей из файла и разбивания его на чанки + вычисление эмбедингов для каждого чанка.

## Формат запроса

```json
{
  "file": "<base64_encoded_file_text>",
  "file_name": "name",
  "file_id": "uuid",
  "file_path": "/path/to/file",
  "chunk_overlap": 300,
  "chunk_size": 3000,
  "entity_types": ["Person", "Organization"]
}
```

Если entity_types не указан, то использовать все типы сущностей: ["Person", "Organization", "Location", "Event", "Product", "Service", "Technology", "Concept", "Time", "Money", "Quantity", "ClassName", "Library", "Framework", "Language", "Tool", "Methodology", "Standard", "Protocol", "API Endpoint"]
chunk_overlap по умолчанию 300, chunk_size по умолчанию 3000.

## Формат ответа

```json
{
  "file_id": "uuid",
  "entities": [],
  "chanks":[
    {
      "id": "int",
      "text": "string",
      "entities": [],
      "embedding": "list of float"
    }
  ]
}
```

## Модель для эмбеддингов

Для эмбеддингов использовать модель qwen3-embedding:8b доступную на локальной ollama. Батчи для эмбеддинга использовать по 20 элементов.

Адрес локальной ollama http://localhost:11434

Эндпоинт для эмбеддинга: http://localhost:11434/api/embed

Пример запроса:

```bash
curl -X POST http://localhost:11434/api/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-embed-text",
    "input": "Текст, для которого нужен эмбеддинг"
  }'
```

---

## Architecture Design

### Created Structure

```
src/ner_controller/
├── domain/
│   ├── entities/
│   │   ├── file_chunk.py                    # Text chunk with entities and embedding
│   │   └── file_processing_result.py        # Aggregated file processing result
│   ├── interfaces/
│   │   ├── embedding_generator_interface.py # Interface for embedding generation
│   │   └── text_chunker_interface.py        # Interface for text chunking
│   └── services/
│       └── file_processing_service.py       # Orchestrates file processing pipeline
├── infrastructure/
│   ├── embedding/
│   │   ├── configs/
│   │   │   └── ollama_embedding_generator_config.py
│   │   └── ollama_embedding_generator.py    # Ollama API client
│   └── chunking/
│       ├── configs/
│       │   └── text_chunker_config.py
│       └── text_chunker.py                  # Character-based chunking with overlap
└── api/
    ├── schemas/
    │   ├── file_process_request.py          # Request schema
    │   └── file_process_response.py         # Response schema with EntitySchema, ChunkSchema
    ├── routers/
    │   └── file_router.py                   # FastAPI router at /file/process
    └── configs/
        └── file_router_config.py            # Router configuration
```

### Components Overview

#### Domain Layer

**Entities:**
- `FileChunk` (frozen dataclass)
  - Purpose: Represents a text chunk with extracted entities and embedding
  - Fields: id (int), text (str), entities (tuple[Entity]), embedding (tuple[float] | None)

- `FileProcessingResult` (frozen dataclass)
  - Purpose: Aggregates results from file processing
  - Fields: file_id (str), entities (tuple[Entity]), chunks (tuple[FileChunk])

**Interfaces:**
- `EmbeddingGeneratorInterface` (ABC)
  - Method: `generate_embeddings(texts: Sequence[str]) -> Sequence[Sequence[float] | None]`
  - Purpose: Contract for generating text embeddings
  - Error handling: Returns None for individual failures, raises EmbeddingGenerationError for total failure

- `TextChunkerInterface` (ABC)
  - Method: `split_text(text, chunk_size, chunk_overlap, start_id) -> list[FileChunk]`
  - Purpose: Contract for splitting text into overlapping chunks
  - Validation: Raises ValueError if chunk_size <= 0 or chunk_overlap >= chunk_size

**Services:**
- `FileProcessingService`
  - Purpose: Orchestrates complete file processing pipeline
  - Dependencies: EntityExtractorInterface, EmbeddingGeneratorInterface, TextChunkerInterface
  - Constants: DEFAULT_ENTITY_TYPES (21 types), DEFAULT_CHUNK_SIZE (3000), DEFAULT_CHUNK_OVERLAP (300)
  - Public method: `process_file(file_base64, file_id, entity_types, chunk_size, chunk_overlap) -> FileProcessingResult`
  - Private methods:
    - `_decode_base64(encoded: str) -> str`: Decodes base64 content
    - `_extract_entities_from_chunks(chunks, entity_types) -> list[FileChunk]`: Extracts entities per chunk
    - `_generate_embeddings_for_chunks(chunks) -> list[FileChunk]`: Batch embedding generation
    - `_collect_all_entities(chunks) -> list`: Aggregates unique entities

#### Infrastructure Layer

**Embedding Generation:**
- `OllamaEmbeddingGenerator` (implements EmbeddingGeneratorInterface)
  - Purpose: Generate embeddings using local Ollama service
  - Dependencies: httpx.Client for HTTP requests
  - Configuration: OllamaEmbeddingGeneratorConfig (base_url, model, batch_size, timeout)
  - Public method: `generate_embeddings(texts) -> Sequence[Sequence[float] | None]`
  - Private methods:
    - `_send_batch_request(texts) -> Sequence[Sequence[float] | None]`: HTTP call to Ollama
    - `_parse_response(response_data) -> Sequence[Sequence[float] | None]`: Response parsing
  - Error handling: EmbeddingGenerationError exception class
  - Cleanup: __del__ closes httpx client

**Text Chunking:**
- `TextChunker` (implements TextChunkerInterface)
  - Purpose: Split text into character-based chunks with overlap
  - Algorithm: Stride = chunk_size - chunk_overlap, step through text
  - Configuration: TextChunkerConfig (preserve_sentences, min_chunk_size)
  - Validation: `_validate_parameters(chunk_size, chunk_overlap)`

#### API Layer

**Schemas:**
- `FileProcessRequest` (Pydantic BaseModel)
  - Fields: file (str), file_name (str), file_id (str), file_path (str | None)
  - Defaults: chunk_overlap=300, chunk_size=3000, entity_types=None

- `FileProcessResponse` (Pydantic BaseModel)
  - Fields: file_id (str), entities (Sequence[EntitySchema]), chanks (Sequence[ChunkSchema])

- `EntitySchema` (Pydantic BaseModel)
  - Fields: text (str), label (str), start (int), end (int)

- `ChunkSchema` (Pydantic BaseModel)
  - Fields: id (int), text (str), entities (Sequence[EntitySchema]), embedding (Sequence[float] | None)

**Router:**
- `FileRouter`
  - Purpose: FastAPI router for file processing endpoint
  - Configuration: FileRouterConfig (prefix="/file", tags=("file-processing",))
  - Endpoint: POST /file/process
  - Public methods:
    - `create_router() -> APIRouter`: Creates configured router
    - `handle_file_process(request) -> FileProcessResponse`: Endpoint handler
  - Private methods:
    - `_validate_request(request)`: Validates chunk_size, chunk_overlap, file content
    - `_convert_to_response(result) -> FileProcessResponse`: Domain to schema conversion
    - `_entity_to_schema(entity) -> EntitySchema`: Entity conversion
    - `_chunk_to_schema(chunk) -> ChunkSchema`: Chunk conversion

### Implementation Recommendations

#### 1. FileProcessingService Implementation Order

**Step 1: Base64 decoding**
- Use `base64.b64decode()` with validation
- Handle UnicodeDecodeError for non-UTF8 content
- Strip leading/trailing whitespace

**Step 2: Text chunking**
- Implement stride-based character chunking in TextChunker
- Ensure last chunk covers remaining text even if smaller than chunk_size
- Add parameter validation before processing

**Step 3: Entity extraction per chunk**
- Reuse existing GlinerEntityExtractor (already implements EntityExtractorInterface)
- Process chunks sequentially or in parallel (consider ThreadPoolExecutor for performance)
- Maintain chunk order and assign extracted entities to respective chunks

**Step 4: Embedding generation**
- Implement batching logic in OllamaEmbeddingGenerator
- Batch size: 20 texts per request (configurable)
- Use httpx for async-friendly HTTP client
- Implement retry logic for transient failures
- Handle partial failures: return None for failed embeddings, log errors

**Step 5: Entity aggregation**
- Collect all entities from all chunks
- Deduplicate by (text, label) pair
- Optionally merge positions across chunks

#### 2. Ollama Integration Details

**Request format:**
```python
POST http://localhost:11434/api/embed
{
    "model": "qwen3-embedding:8b",
    "input": ["text1", "text2", ...]  # Batch of up to 20 texts
}
```

**Expected response format:**
```python
{
    "model": "qwen3-embedding:8b",
    "embeddings": [
        [0.1, 0.2, 0.3, ...],  # Embedding for text1
        [0.4, 0.5, 0.6, ...]   # Embedding for text2
    ]
}
```

**Error handling:**
- httpx.TimeoutException: Log and retry (up to 3 attempts)
- httpx.ConnectError: Raise EmbeddingGenerationError (service unavailable)
- HTTP 500: Return None for batch, log error
- Malformed response: Raise EmbeddingGenerationError

#### 3. Testing Strategy

**Unit tests:**
- `FileProcessingService._decode_base64`: Test valid/invalid base64, empty strings
- `TextChunker.split_text`: Test edge cases (empty text, single chunk, overlap > size)
- `OllamaEmbeddingGenerator`: Mock httpx.Client for successful/failed requests
- `FileRouter._validate_request`: Test parameter validation

**Integration tests:**
- `FileProcessingService`: Mock Ollama, test with real GLiNER
- `OllamaEmbeddingGenerator`: Test against real Ollama instance (skip if unavailable)

**E2E test:**
- Full request/response cycle with real GLiNER and Ollama
- Use small test file (< 100 chars)
- Verify chunk count, entity extraction, embedding dimensions

#### 4. Performance Considerations

**Entity extraction:**
- GLiNER processes each chunk independently
- Consider batch processing if GLiNER supports it
- For large files: chunk count = (text_length) / (chunk_size - overlap)

**Embedding generation:**
- Batching reduces HTTP overhead
- 20 texts per batch = ~20-100 HTTP requests for typical files
- Ollama local service: expect 50-200ms per batch
- For large files: consider async processing (asyncio + httpx.AsyncClient)

**Memory:**
- Chunks stored in memory: ~chunk_size * chunk_count bytes
- Embeddings: embedding_dim * chunk_count * 4 bytes (float32)
- For 100KB file with 3000 char chunks: ~34 chunks, ~34KB text + embeddings

#### 5. Security Considerations

**Input validation:**
- Validate chunk_size and chunk_overlap ranges to prevent DoS
- Limit file size (add max_file_bytes to config, suggest 10MB)
- Sanitize file_name and file_path (prevent path traversal)

**Ollama integration:**
- Local service only (localhost:11434)
- No authentication required for local deployment
- Consider adding API key if deployed externally

**Error messages:**
- Don't expose internal details in HTTP responses
- Log full errors server-side
- Return generic error messages to clients

#### 6. Required Dependencies

Add to requirements.txt:
```
httpx>=0.25.0
```

Already present:
```
fastapi
pydantic
gliner
```

#### 7. Configuration Management

Environment variables (optional):
- `OLLAMA_BASE_URL`: Override default http://localhost:11434
- `OLLAMA_MODEL`: Override default qwen3-embedding:8b
- `OLLAMA_TIMEOUT`: Override default 60s timeout
- `MAX_FILE_SIZE`: Override file size limit

Implement in OllamaEmbeddingGeneratorConfig.__init__ using os.getenv().

### Considerations

**Edge cases:**
1. Empty file content: Return empty entities and empty chunks list
2. Single chunk smaller than chunk_size: Still create one chunk
3. No entities found in chunk: Return chunk with empty entities tuple
4. Embedding generation failure for some chunks: Set embedding=None for those chunks, continue
5. Invalid base64: Raise ValueError with clear message
6. Text shorter than chunk_size: Create single chunk with full text
7. chunk_overlap >= chunk_size: Validate and reject before processing

**Performance notes:**
- GLiNER model loading is cached in GlinerEntityExtractor (already implemented)
- Ollama model loading is handled by Ollama service (not in Python code)
- First request will be slow (model loading), subsequent requests fast
- Consider health check endpoint to verify Ollama availability

**Error handling strategy:**
- Input validation errors: HTTP 400 with specific field and issue
- Service unavailable (Ollama down): HTTP 503 with retry-after hint
- Processing errors: HTTP 500 with generic message, log details
- Partial failures (some chunks fail): Return partial results, log warnings

**Testing strategy:**
- Unit tests for each component with mocked dependencies
- Integration tests for service layer with real GLiNER, mocked Ollama
- E2E tests with both real services (use small test files)
- Parameterized tests for various chunk sizes and entity type combinations
- Performance tests with large files (1MB+) to validate batching
- Concurrent request tests to verify thread safety

**Potential challenges:**
1. Entity position offsets are relative to each chunk, not original file
   - Decision: Keep positions chunk-relative (simpler, matches requirements)
   - Alternative: Calculate absolute positions (add chunk offset)

2. Duplicate entities across chunks
   - Decision: Return all entities with positions, deduplication optional
   - Current design: Collects unique entities but keeps chunk-specific entities

3. Ollama batch processing order
   - Challenge: Ensure embeddings match chunk order
   - Solution: Process batches sequentially, preserve order within batch

4. GLiNER multilingual support
   - Already using gliner_multi-v2.1 model
   - Handles multiple languages without configuration

5. Large file handling
   - Current design: Full file in memory
   - Future improvement: Stream processing for files > 100MB

### Dependencies Flow

```
FileRouter (API)
  -> FileProcessingService (Domain)
      -> EntityExtractorInterface (Domain)
          <- GlinerEntityExtractor (Infrastructure)
      -> EmbeddingGeneratorInterface (Domain)
          <- OllamaEmbeddingGenerator (Infrastructure)
      -> TextChunkerInterface (Domain)
          <- TextChunker (Infrastructure)
```

All dependencies flow from API -> Domain -> Infrastructure, following DDD principles.

### File Paths Reference

**Domain Layer:**
- /home/denis/Projects/ner_controller/src/ner_controller/domain/entities/file_chunk.py
- /home/denis/Projects/ner_controller/src/ner_controller/domain/entities/file_processing_result.py
- /home/denis/Projects/ner_controller/src/ner_controller/domain/interfaces/embedding_generator_interface.py
- /home/denis/Projects/ner_controller/src/ner_controller/domain/interfaces/text_chunker_interface.py
- /home/denis/Projects/ner_controller/src/ner_controller/domain/services/file_processing_service.py

**Infrastructure Layer:**
- /home/denis/Projects/ner_controller/src/ner_controller/infrastructure/embedding/configs/ollama_embedding_generator_config.py
- /home/denis/Projects/ner_controller/src/ner_controller/infrastructure/embedding/ollama_embedding_generator.py
- /home/denis/Projects/ner_controller/src/ner_controller/infrastructure/chunking/configs/text_chunker_config.py
- /home/denis/Projects/ner_controller/src/ner_controller/infrastructure/chunking/text_chunker.py

**API Layer:**
- /home/denis/Projects/ner_controller/src/ner_controller/api/schemas/file_process_request.py
- /home/denis/Projects/ner_controller/src/ner_controller/api/schemas/file_process_response.py
- /home/denis/Projects/ner_controller/src/ner_controller/api/routers/file_router.py
- /home/denis/Projects/ner_controller/src/ner_controller/api/configs/file_router_config.py

