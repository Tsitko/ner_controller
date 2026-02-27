# NER Controller

Service for Named Entity Recognition (NER) and text processing using GLiNER, Natasha, regex patterns, and embedding models.

**Models:**
- GLiNER: `urchade/gliner_multi-v2.1` (multilingual NER)
- Natasha: `slovnet_ner_news_v1` + `navec_news_v1_1B_250K_300d_100q` (Russian NER)
- Regex: API endpoints pattern matching
- Embeddings: `text-embedding-qwen3-embedding-8b` via LM Studio (local)

## Features

### Entity Extraction
- **Hybrid approach**: Combines GLiNER ML model with regex-based extractors
- **Composite extractor**: Runs GLiNER first, then regex extractors for specialized patterns
- **Deduplication**: Levenshtein distance-based (threshold <= 2) with case-insensitive matching
- **Unique entities**: Returns deduplicated list of entity names (strings)

### Hallucination Detection
- FastAPI endpoint: `/hallucination/check`
- Compares entities between request and response
- Detects potential hallucinations and missing entities
- Configurable entity types

### File Processing
- FastAPI endpoint: `/file/process`
- Accepts base64-encoded text files
- Splits text into chunks with overlap
- Extracts entities from each chunk (GLiNER + regex)
- Generates embeddings via LM Studio for each chunk
- Returns all unique entities and chunked text with embeddings

### Text Processing
- FastAPI endpoint: `/text/process`
- Accepts plain text (no base64 encoding)
- Extracts entities from the full text (no chunking)
- Generates a single embedding for the full text
- Returns entities and embedding for the text

## Requirements
- Python 3.12
- venv in project root (`venv/`)
- LM Studio running with OpenAI-compatible API and `text-embedding-qwen3-embedding-8b` model

## Setup
```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Run
```bash
venv/bin/uvicorn ner_controller.main:app --host 0.0.0.0 --port 1304
```

OpenAPI docs are available at `http://localhost:1304/docs`.

## Model Download

### GLiNER Model
The GLiNER model (`urchade/gliner_multi-v2.1`), base model (`microsoft/mdeberta-v3-base`),
and Natasha models must be downloaded once and then used strictly from local cache.

To manually download the model:
```bash
venv/bin/python download_model.py
```

`download_model.py` downloads all models required for fully offline execution, including Natasha.

### LM Studio Embedding Model
Install and run LM Studio:
```bash
# Install LM Studio (https://lmstudio.ai)
# Load the text-embedding-qwen3-embedding-8b model
# Start the inference server with OpenAI-compatible API
# Default: http://localhost:1234 (or configure custom host/port in LmStudioEmbeddingGeneratorConfig)
```

The service connects to `http://localhost:1234/v1/embeddings` by default (OpenAI-compatible API).

## Autostart (WSL)
1) Enable systemd for WSL by adding this to `/etc/wsl.conf`:
```ini
[boot]
systemd=true
```
2) Restart WSL: `wsl.exe --shutdown` from Windows, then open WSL again.
3) Run the setup script:
```bash
./setup_supervisor.sh
```

## Test
```bash
# Run all tests (376 tests)
venv/bin/python -m unittest discover -s tests

# Run E2E test with real embeddings
venv/bin/python tests/e2e/test_file_processing_real.py
```

## API Endpoints

### POST /hallucination/check
Detect potential LLM hallucinations by comparing entities.

**Request:**
```json
{
  "request": "Alice went to Paris and met OpenAI researchers.",
  "response": "Bob visited Paris and joined Anthropic.",
  "entity_types": ["Person", "Location", "Organization"]
}
```

**Response:**
```json
{
  "potential_hallucinations": ["Bob", "Anthropic"],
  "missing_entities": ["Alice", "OpenAI researchers"]
}
```

**Note:** `entity_types` is the preferred field name. The legacy `entities_types` alias is still accepted.
**Note:** OpenAPI field descriptions explicitly mention the legacy `entities_types` alias.

### POST /file/process
Process a text file with NER and embeddings.

**Request:**
```json
{
  "file": "<base64_encoded_content>",
  "file_name": "document.txt",
  "file_id": "unique-uuid",
  "file_path": "/path/to/file",
  "chunk_size": 3000,
  "chunk_overlap": 300,
  "entity_types": ["Person", "Organization"]
}
```

**Response:**
```json
{
  "file_id": "unique-uuid",
  "entities": ["Alice", "OpenAI", "POST /api/users", "GET /billing/info"],
  "chanks": [
    {
      "id": 0,
      "text": "First chunk of text...",
      "entities": ["Alice", "OpenAI"],
      "embedding": [0.1, 0.2, ...]
    }
  ]
}
```

**Note:** Entity format has been simplified to plain strings (names only). Labels, positions, and other metadata are removed. Entities are automatically deduplicated using Levenshtein distance (threshold <= 2).
**Note:** `entity_types` is the preferred field name. The legacy `entities_types` alias is still accepted.

### POST /text/process
Process a single text with NER and embedding (no chunking, no base64).

**Request:**
```json
{
  "text": "Alice visited Paris and met with OpenAI researchers.",
  "entity_types": ["Person", "Location", "Organization"]
}
```

**Response:**
```json
{
  "text": "Alice visited Paris and met with OpenAI researchers.",
  "entities": ["Alice", "Paris", "OpenAI", "researchers"],
  "embedding": [0.1, 0.2, 0.3, ...]
}
```

**Note:** If `entity_types` is not specified, uses the default comprehensive list of 23 entity types.
**Note:** `entity_types` is the preferred field name. The legacy `entities_types` alias is still accepted.

## Entity Types

Default entity types (used if not specified):
- **General**: Person, Organization, Location, Event
- **Business**: Product, Service, Technology, Concept
- **Data**: Time, Money, Quantity
- **Code**: ClassName, Library, Framework, Language, Tool
- **Technical**: Methodology, Standard, Protocol, API Endpoint

### Entity Deduplication

Entities are automatically deduplicated using:
1. **Case-insensitive matching**: "Alice" == "alice"
2. **Levenshtein distance**: Similar strings within distance <= 2 are merged
   - Example: "мосбилет" and "мосбилета" → "мосбилет"

### Specialized Extractors

**API Endpoint Extractor (regex-based):**
- Finds patterns like: `POST /api/users`, `GET /billing/info`
- Runs after GLiNER to complement ML-based extraction
- Case-insensitive HTTP method matching
