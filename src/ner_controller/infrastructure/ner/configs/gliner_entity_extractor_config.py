"""Configuration for the GLiNER entity extractor."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GlinerEntityExtractorConfig:
    """Configuration values for GLiNER model loading and inference."""

    model_name: str = "urchade/gliner_small-v2.1"
    device: str = "cpu"
    batch_size: int = 8
    cache_dir: Optional[str] = None
    local_files_only: bool = True
