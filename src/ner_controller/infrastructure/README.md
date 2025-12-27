# Infrastructure Layer

External integrations such as GLiNER-backed NER extractors.

## GLiNER Configuration

The service uses `urchade/gliner_multi-v2.1` model for multilingual NER.

### Model Loading Behavior

- **First run**: Model is downloaded automatically via `download_model.py` script
  called from `setup_supervisor.sh`
- **Subsequent runs**: Model is loaded from `~/.cache/huggingface` cache with
  `local_files_only=True`
- **Offline mode**: `TRANSFORMERS_OFFLINE=1` prevents unintended downloads during
  normal operation

### Configuration

See `ner/configs/gliner_entity_extractor_config.py` for all settings:
- `model_name`: HuggingFace model identifier
- `device`: CPU or GPU device
- `cache_dir`: Custom cache directory (None = default HF cache)
- `local_files_only`: Restrict to cached models only
