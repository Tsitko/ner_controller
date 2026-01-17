"""Configuration for the GLiNER entity extractor."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GlinerEntityExtractorConfig:
    """Configuration values for GLiNER model loading and inference."""

    model_name: str = "urchade/gliner_multi-v2.1"
    device: str = "cpu"
    batch_size: int = 8
    cache_dir: Optional[str] = None
    local_files_only: bool = True
    offline_mode: bool = True
    offline_env_vars: dict[str, str] = field(
        default_factory=lambda: {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
