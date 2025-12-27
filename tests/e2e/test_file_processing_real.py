"""E2E test for file processing with real HTTP call and embedding model."""

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests

# Configuration
SERVICE_URL = "http://localhost:1304"
ENDPOINT = "/file/process"
TIMEOUT = 120  # seconds
TEST_FILE_PATH = Path(__file__).parent.parent.parent / "test.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def read_test_file() -> str:
    """Read test.txt file content."""
    if not TEST_FILE_PATH.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE_PATH}")

    with open(TEST_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def encode_to_base64(text: str) -> str:
    """Encode text to base64."""
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def send_request(file_content_b64: str, file_name: str) -> dict:
    """Send POST request to file processing endpoint."""
    url = f"{SERVICE_URL}{ENDPOINT}"
    file_id = str(uuid.uuid4())

    payload = {
        "file": file_content_b64,
        "file_name": file_name,
        "file_id": file_id,
        "file_path": str(TEST_FILE_PATH),
        "chunk_size": 3000,
        "chunk_overlap": 300,
        # Use default entity types (None)
        "entity_types": None,
    }

    print(f"Sending request to {url}")
    print(f"File: {file_name} ({len(file_content_b64)} chars base64)")
    print(f"Timeout: {TIMEOUT}s")

    start_time = time.time()

    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )

    elapsed = time.time() - start_time

    print(f"Response status: {response.status_code}")
    print(f"Response time: {elapsed:.2f}s")

    if response.status_code != 200:
        print(f"Response body: {response.text}")
        raise Exception(f"Request failed with status {response.status_code}")

    return response.json()


def save_response(response_data: dict, file_name: str) -> Path:
    """Save response JSON to output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / file_name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)

    print(f"Response saved to: {output_file}")
    return output_file


def print_summary(response_data: dict, output_path: Path):
    """Print summary of the response."""
    print("\n" + "=" * 60)
    print("RESPONSE SUMMARY")
    print("=" * 60)

    print(f"File ID: {response_data.get('file_id')}")
    print(f"Total entities found: {len(response_data.get('entities', []))}")
    print(f"Total chunks: {len(response_data.get('chanks', []))}")

    entities = response_data.get('entities', [])
    if entities:
        print("\nUnique entities:")
        # entities are now just strings, not objects with labels
        print(f"  Total unique entities: {len(entities)}")
        print(f"  Examples: {', '.join(entities[:10])}")
        if len(entities) > 10:
            print(f"  ... and {len(entities) - 10} more")

    chunks = response_data.get('chanks', [])
    if chunks:
        print(f"\nChunks info:")
        chunks_with_embedding = sum(1 for c in chunks if c.get('embedding') is not None)
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Chunks with embeddings: {chunks_with_embedding}")
        print(f"  Chunks without embeddings: {len(chunks) - chunks_with_embedding}")

        if chunks:
            first_chunk = chunks[0]
            print(f"\nFirst chunk:")
            print(f"  ID: {first_chunk.get('id')}")
            print(f"  Text length: {len(first_chunk.get('text', ''))}")
            print(f"  Entities: {len(first_chunk.get('entities', []))}")
            if first_chunk.get('embedding'):
                emb_dim = len(first_chunk['embedding'])
                print(f"  Embedding dimension: {emb_dim}")

    print(f"\nFull response saved to: {output_path}")
    print("=" * 60)


def main():
    """Main test execution."""
    print("=" * 60)
    print("E2E TEST: File Processing with Real Embeddings")
    print("=" * 60)

    try:
        # 1. Read test file
        print(f"\n1. Reading test file: {TEST_FILE_PATH}")
        text_content = read_test_file()
        print(f"   File size: {len(text_content)} characters")

        # 2. Encode to base64
        print("\n2. Encoding to base64...")
        file_b64 = encode_to_base64(text_content)
        print(f"   Base64 size: {len(file_b64)} characters")

        # 3. Send request
        print("\n3. Sending HTTP request...")
        response_data = send_request(file_b64, "test.txt")

        # 4. Save response
        print("\n4. Saving response...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = save_response(response_data, f"file_processing_result_{timestamp}.json")

        # 5. Print summary
        print_summary(response_data, output_file)

        print("\n✅ TEST PASSED")
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
