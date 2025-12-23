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

## Test
```bash
venv/bin/python -m unittest discover -s tests
```
