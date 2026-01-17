# LLM Readme

## Purpose
- NER integrations and extractors.

## Key Files
- See files in this folder for implementation details.

## Usage Notes
- Follow project naming and one-class-per-file rules.
- GLiNER loads in offline-only mode; tokenizer files are copied from the cached
  base model (`microsoft/mdeberta-v3-base`) into the GLiNER snapshot directory
  to avoid network access.

## Interfaces
- This folder contains internal modules used by higher-level layers.
