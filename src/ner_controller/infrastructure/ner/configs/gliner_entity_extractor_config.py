"""Configuration for the GLiNER entity extractor."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GlinerEntityExtractorConfig:
    """Configuration values for GLiNER model loading and inference."""

    model_name: str = "urchade/gliner_multi-v2.1"
    base_model_name: str = "microsoft/mdeberta-v3-base"
    device: str = "cpu"
    batch_size: int = 8
    cache_dir: Optional[str] = None
    local_files_only: bool = True
    offline_mode: bool = True
    max_segment_chars: int = 1200
    min_segment_chars: int = 200
    prediction_threshold: float = 0.2
    offline_env_vars: dict[str, str] = field(
        default_factory=lambda: {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
