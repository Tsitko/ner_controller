#!/usr/bin/env python3
"""Download GLiNER model to HuggingFace cache."""

import sys
from pathlib import Path

from gliner import GLiNER

MODEL_NAME = "urchade/gliner_multi-v2.1"


def download_model() -> None:
    """Download model to HuggingFace cache directory."""
    print(f"Downloading {MODEL_NAME}...")
    print("This may take a while on first run...")

    try:
        model = GLiNER.from_pretrained(MODEL_NAME, local_files_only=False)
        print(f"Successfully downloaded {MODEL_NAME}")
        print(f"Model cached at: ~/.cache/huggingface/hub/models--{MODEL_NAME.replace('/', '--')}")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_model()
