"""Configuration for Natasha entity extractor."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NatashaEntityExtractorConfig:
    """Configuration values for Natasha NER model loading and inference."""

    cache_dir: str = "~/.cache/natasha"
    navec_model_filename: str = "navec_news_v1_1B_250K_300d_100q.tar"
    ner_model_filename: str = "slovnet_ner_news_v1.tar"

    def navec_model_path(self) -> Path:
        """Return full path to local Navec model file."""
        cache_path = Path(self.cache_dir).expanduser() / self.navec_model_filename
        if cache_path.exists():
            return cache_path
        return self._package_navec_model_path()

    def ner_model_path(self) -> Path:
        """Return full path to local Slovnet NER model file."""
        cache_path = Path(self.cache_dir).expanduser() / self.ner_model_filename
        if cache_path.exists():
            return cache_path
        return self._package_ner_model_path()

    def _package_navec_model_path(self) -> Path:
        """Return Navec model path bundled with natasha package."""
        import natasha

        return Path(natasha.__file__).resolve().parent / "data" / "emb" / self.navec_model_filename

    def _package_ner_model_path(self) -> Path:
        """Return Slovnet NER model path bundled with natasha package."""
        import natasha

        return Path(natasha.__file__).resolve().parent / "data" / "model" / self.ner_model_filename
