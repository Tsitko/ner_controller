# Infrastructure Layer

External integrations such as GLiNER/Natasha-backed NER extractors and LM Studio embedding generation.

Notes:
- Regex API endpoint extractor matches HTTP methods followed by a leading-slash path.

## GLiNER Configuration

The service uses `urchade/gliner_multi-v2.1` model for multilingual NER.

### Model Loading Behavior

- **Offline-only**: GLiNER loads strictly from local cache and will not attempt
  any network access.
- **Base model cache**: GLiNER depends on the base transformer model
  (`microsoft/mdeberta-v3-base`). The service ensures tokenizer files are copied
  into the GLiNER snapshot directory so loading stays local.
- **Failure mode**: If required artifacts are missing locally, the service raises
  an error instead of downloading them.

### Configuration

See `ner/configs/gliner_entity_extractor_config.py` for all settings:
- `model_name`: HuggingFace model identifier
- `base_model_name`: Base transformer required by GLiNER tokenizer
- `device`: CPU or GPU device
- `cache_dir`: Custom cache directory (None = default HF cache)
- `local_files_only`: Restrict to cached models only
- `offline_mode`: Enforce offline-only loading
- `offline_env_vars`: Environment variables applied to disable network access

## Natasha Configuration

The service also uses Natasha (Slovnet NER) for additional Russian entity extraction.

See `ner/configs/natasha_entity_extractor_config.py`:
- `cache_dir`: Local directory with Natasha models
- `navec_model_filename`: Navec embedding model file
- `ner_model_filename`: Slovnet NER model file

## LM Studio Embedding Configuration

The service uses LM Studio with OpenAI-compatible API for text embeddings.
Embedding requests are serialized to avoid sending new requests before previous ones complete.

### Configuration

See `embedding/configs/lm_studio_embedding_generator_config.py` for all settings:
- `base_url`: LM Studio server URL (default: `http://localhost:1234`)
- `model`: Model name (default: `text-embedding-qwen3-embedding-8b`)
- `embedding_endpoint`: OpenAI-compatible endpoint (default: `/v1/embeddings`)
- `batch_size`: Number of texts per request (default: 4)
- `request_timeout`: Request timeout in seconds (default: 180)

### API Format

The generator uses OpenAI-compatible API format:

**Request:**
```json
{
  "model": "text-embedding-qwen3-embedding-8b",
  "input": ["text1", "text2"]
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, ...], "index": 0},
    {"object": "embedding", "embedding": [0.3, 0.4, ...], "index": 1}
  ],
  "model": "text-embedding-qwen3-embedding-8b",
  "usage": {"prompt_tokens": 10, "total_tokens": 10}
}
```
