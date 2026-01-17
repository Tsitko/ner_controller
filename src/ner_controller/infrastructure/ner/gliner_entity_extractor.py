"""GLiNER-based entity extraction implementation."""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from gliner import GLiNER
from huggingface_hub import snapshot_download

from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.levenshtein_utils import deduplicate_entities
from ner_controller.infrastructure.ner.configs.gliner_entity_extractor_config import (
    GlinerEntityExtractorConfig,
)

logger = logging.getLogger(__name__)


class GlinerEntityExtractor(EntityExtractorInterface):
    """Extracts entities using the GLiNER model."""

    def __init__(self, config: GlinerEntityExtractorConfig) -> None:
        """Initialize the extractor with model configuration."""
        self._config = config
        self._model: Optional[GLiNER] = None
        self._validate_config()
        if self._config.offline_mode:
            self._apply_offline_environment()

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[str]:
        """Extract entities from text using GLiNER."""
        if not text or not entity_types:
            return []

        model = self._load_model()
        predictions = model.predict_entities(text, list(entity_types))

        # Extract only entity text from predictions
        entities = [prediction.get("text", "") for prediction in predictions]

        # Apply deduplication with Levenshtein distance (threshold <= 2)
        return deduplicate_entities(entities, threshold=2)

    def _load_model(self) -> GLiNER:
        """Load the GLiNER model if it hasn't been loaded yet."""
        if self._model is None:
            try:
                model_id = self._config.model_name
                if self._config.offline_mode:
                    model_id = str(self._prepare_local_model_dir())
                self._model = GLiNER.from_pretrained(
                    model_id,
                    cache_dir=self._config.cache_dir,
                    local_files_only=self._config.local_files_only,
                )
            except Exception as exc:
                logger.error("Failed to load GLiNER model: %s", exc)
                if self._config.offline_mode:
                    raise ValueError(
                        "GLiNER model is not available locally while offline mode is enabled."
                    ) from exc
                raise
            if hasattr(self._model, "to"):
                self._model.to(self._config.device)
        return self._model

    def _apply_offline_environment(self) -> None:
        """Apply offline environment variables to prevent network access."""
        for key, value in self._config.offline_env_vars.items():
            os.environ[key] = value

    def _prepare_local_model_dir(self) -> Path:
        """Ensure local GLiNER model directory contains tokenizer files."""
        model_dir = self._snapshot_path(self._config.model_name)
        gliner_config_path = model_dir / "gliner_config.json"
        if not gliner_config_path.exists():
            raise FileNotFoundError(f"Missing GLiNER config at {gliner_config_path}")

        base_model_name = self._read_base_model_name(gliner_config_path)
        if base_model_name:
            base_model_dir = self._snapshot_path(base_model_name)
            self._ensure_tokenizer_files(model_dir, base_model_dir)
        return model_dir

    def _snapshot_path(self, model_name: str) -> Path:
        """Resolve a cached model snapshot directory without network access."""
        snapshot_path = snapshot_download(
            repo_id=model_name,
            cache_dir=self._config.cache_dir,
            local_files_only=True,
        )
        return Path(snapshot_path)

    def _read_base_model_name(self, config_path: Path) -> str:
        """Read base model name from GLiNER config."""
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        return str(config.get("model_name", "")).strip()

    def _ensure_tokenizer_files(self, model_dir: Path, base_model_dir: Path) -> None:
        """Copy tokenizer files from base model cache into GLiNER model directory."""
        tokenizer_files = (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "spm.model",
            "added_tokens.json",
            "chat_template.jinja",
        )
        for filename in tokenizer_files:
            target_path = model_dir / filename
            source_path = base_model_dir / filename
            if target_path.exists() or not source_path.exists():
                continue
            target_path.write_bytes(source_path.read_bytes())

    def _validate_config(self) -> None:
        """Validate extractor configuration for offline-only behavior."""
        if self._config.offline_mode and not self._config.local_files_only:
            raise ValueError("offline_mode requires local_files_only=True.")
        if self._config.offline_mode and not self._config.offline_env_vars:
            raise ValueError("offline_mode requires offline_env_vars to be set.")
