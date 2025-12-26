# NER Controller

Service to detect potential LLM hallucinations by comparing entities extracted from the input prompt
and the model response using GLiNER.

## Features
- FastAPI service exposing `/hallucination/check`.
- Entity extraction via GLiNER with configurable entity types.
- Reports potential hallucinations and missing entities.

## Requirements
- Python 3.12
- venv in project root (`venv/`)

## Setup
```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Run
```bash
venv/bin/uvicorn ner_controller.main:app --host 0.0.0.0 --port 1304
```

OpenAPI docs are available at `http://localhost:1304/docs`.

Note: The service expects GLiNER and its dependencies to be cached locally.
If you want to forbid downloads, set `TRANSFORMERS_OFFLINE=1` and point to the cache with
`HF_HOME=/home/denis/.cache/huggingface`.

## Autostart (WSL)
1) Enable systemd for WSL by adding this to `/etc/wsl.conf`:
```ini
[boot]
systemd=true
```
2) Restart WSL: `wsl.exe --shutdown` from Windows, then open WSL again.
3) Run the setup script:
```bash
./setup_supervisor.sh
```

## Test
```bash
venv/bin/python -m unittest discover -s tests
```
