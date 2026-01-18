# ner_controller Package

Core package providing API, domain logic, and infrastructure for the hallucination checker.

Notes:
- Request schemas accept legacy `entities_types` in addition to `entity_types`.
- Entity deduplication uses normalized Levenshtein similarity (relative threshold).
