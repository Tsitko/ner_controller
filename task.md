# Task

Разработать сервис для проверки галюцинаций LLM.

На вход передаем полный запрос к LLM со всем контекстом и ответ LLM. В формате

```json
{
  "request": "...",
  "response": "...",
  "entities_types": ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "ARTIFACT", "OTHER", "UI_ELEMENT", "API_ENDPOINT", "...etc"]
}
```

На выходе получаем список потенциальных галюцинаций и потенциално упущенных сущностей.

```json
{
  "potential_hallucinations": [
    "..."
  ],
  "missing_entities": [
    "..."
  ]
}
```

## Requirements

- Сервис должен быть написан на Python с использованием библиотеки gliner для NER.
- Сервис должен быть реализован как сервис на FastAPI.
- Сервис должен быть развернут на порту 1304
- Сервис должен содержать документацию в формате OpenAPI на localhost:1304/docs

## Принцип работы

1. Сервис использует библиотеку gliner для NER получая сущностий типов переданных в entities_types из request и response.
2. Сервис сравнивает полученные сущности с переданными в entities_types и определяет потенциальные галюцинации и упущенные сущности. Галлюцинации - которые не найдены в request, но найдены в response. Упущенные сущности - которые найдены в request, но не найдены в response.
3. Сервис возвращает результат в формате JSON.

### Architecture Design

#### Created Structure

```
src/ner_controller/
  __init__.py - package marker
  main.py - ApplicationFactory (FastAPI wiring)
  configs/
    __init__.py
    app_config.py - AppConfig
  api/
    __init__.py
    configs/
      __init__.py
      hallucination_router_config.py - HallucinationRouterConfig
    routers/
      __init__.py
      hallucination_router.py - HallucinationRouter
    schemas/
      __init__.py
      hallucination_check_request.py - HallucinationCheckRequest
      hallucination_check_response.py - HallucinationCheckResponse
  domain/
    __init__.py
    entities/
      __init__.py
      entity.py - Entity
      entity_diff_result.py - EntityDiffResult
      hallucination_detection_result.py - HallucinationDetectionResult
    interfaces/
      __init__.py
      entity_extractor_interface.py - EntityExtractorInterface
    services/
      __init__.py
      entity_diff_calculator.py - EntityDiffCalculator
      hallucination_detection_service.py - HallucinationDetectionService
  infrastructure/
    __init__.py
    ner/
      __init__.py
      configs/
        __init__.py
        gliner_entity_extractor_config.py - GlinerEntityExtractorConfig
      gliner_entity_extractor.py - GlinerEntityExtractor
requirements.txt
```

#### Components Overview

- `ApplicationFactory` (main.py): Builds FastAPI app and wires dependencies.
- `AppConfig`: App host/port/title/docs configuration.
- `HallucinationRouter` + `HallucinationRouterConfig`: HTTP routing and endpoint registration.
- `HallucinationCheckRequest/Response`: Pydantic API schemas.
- `Entity`: Extracted entity representation.
- `EntityDiffResult`: Differences between request/response entities.
- `HallucinationDetectionResult`: Domain output for use case.
- `EntityExtractorInterface`: Abstraction for NER engines.
- `GlinerEntityExtractor`: GLiNER-backed extractor implementation.
- `EntityDiffCalculator`: Compares entities across request/response.
- `HallucinationDetectionService`: Orchestrates extraction and comparison.

#### Implementation Recommendations

1. **Entity extraction**: Load GLiNER once per process, reuse for all requests. Use the
   provided `entities_types` as labels for extraction.
2. **Normalization**: Compare entities by normalized text + label (trim, casefold),
   optionally deduplicate by (text, label) to avoid repeated matches.
3. **Diff logic**: `potential_hallucinations` = response-only entities; `missing_entities`
   = request-only entities. Keep output stable and deterministic (sorted).
4. **Error handling**: Validate non-empty inputs; raise clear HTTP 422/400 on invalid
   entity types or empty payloads. Log model inference errors with context.
5. **Performance**: Batch inference if GLiNER supports it; apply minimal preprocessing.
6. **Configuration**: Keep GLiNER model name, device, batch size in
   `GlinerEntityExtractorConfig`. Keep API prefix and tags in `HallucinationRouterConfig`.
7. **Dependency wiring**: `main.py` should construct `GlinerEntityExtractor` and
   `HallucinationDetectionService`, then register routes.

#### Considerations

- **Assumptions**: GLiNER model name `gliner-community/gliner_small` is suitable for
  initial implementation. Templates for README/llm_readme are not present in repo.
- **Edge cases**: Repeated entities, overlapping spans, entities with same text and
  different labels, empty `entities_types`, very long inputs.
- **Performance notes**: Model loading should be singleton; avoid per-request reload.
- **Security notes**: Input validation and size limits to prevent abuse.
- **Testing strategy**: Unit tests per class; integration tests for router/service;
  E2E for the single use case; mock GLiNER in unit tests.

#### Suggested Implementation Order

1. Infrastructure: `GlinerEntityExtractor` + config.
2. Domain: `EntityDiffCalculator` and `HallucinationDetectionService`.
3. API schemas and router.
4. App wiring in `ApplicationFactory`.
5. Tests (unit, integration, e2e) per rules.
