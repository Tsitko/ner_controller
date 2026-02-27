#!/usr/bin/env python3
"""Download all required NLP models to local cache for offline use."""

import sys
from pathlib import Path
from urllib.request import urlretrieve

from gliner import GLiNER
from huggingface_hub import snapshot_download

MODEL_NAME = "urchade/gliner_multi-v2.1"
BASE_MODEL_NAME = "microsoft/mdeberta-v3-base"
NATASHA_CACHE_DIR = Path("~/.cache/natasha").expanduser()
NATASHA_NAVEC_URL = (
    "https://storage.yandexcloud.net/natasha-slovnet/packs/"
    "navec_news_v1_1B_250K_300d_100q.tar"
)
NATASHA_NER_URL = (
    "https://storage.yandexcloud.net/natasha-slovnet/packs/"
    "slovnet_ner_news_v1.tar"
)


def _download_file(url: str, destination: Path) -> None:
    """Download a file from URL unless it is already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already cached: {destination}")
        return
    print(f"Downloading {url} -> {destination}")
    urlretrieve(url, destination)
    print(f"Saved: {destination}")


def download_model() -> None:
    """Download GLiNER, base transformer, and Natasha models."""
    print("Downloading all required models...")
    print("This may take a while on first run. Next runs are fully local.")

    try:
        GLiNER.from_pretrained(MODEL_NAME, local_files_only=False)
        snapshot_download(repo_id=BASE_MODEL_NAME, local_files_only=False)
        _download_file(NATASHA_NAVEC_URL, NATASHA_CACHE_DIR / "navec_news_v1_1B_250K_300d_100q.tar")
        _download_file(NATASHA_NER_URL, NATASHA_CACHE_DIR / "slovnet_ner_news_v1.tar")
        print(f"Successfully downloaded {MODEL_NAME}")
        print(f"Successfully downloaded {BASE_MODEL_NAME}")
        print("Successfully downloaded Natasha models")
        print(f"Cached at: ~/.cache/huggingface/hub/models--{MODEL_NAME.replace('/', '--')}")
        print(f"Cached at: ~/.cache/huggingface/hub/models--{BASE_MODEL_NAME.replace('/', '--')}")
        print(f"Cached at: {NATASHA_CACHE_DIR}")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_model()
