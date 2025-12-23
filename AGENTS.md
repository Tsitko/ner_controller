# AGENTS.md

## Project Overview

Develop a Python FastAPI service that detects potential LLM hallucinations using NER via the `gliner` library. The service accepts a full LLM prompt with context and response plus a list of entity types. It returns:
- `potential_hallucinations`: entities found in response but not in request.
- `missing_entities`: entities found in request but not in response.

Target:
- FastAPI service
- Port: 1304
- OpenAPI docs at `http://localhost:1304/docs`

## Development Order (must follow)

1. **Architecture & design** based on use cases and project rules. Define folder structure, class/method signatures, imports, dependencies, and update `requirements.txt`. Create and use a `venv` in project root; run code only inside the venv. Update documentation per rules.
2. **Tests first**: Write unit and integration tests for each use case, following testing rules. Update documentation per rules.
3. **Implementation**: Implement code per designed architecture and style rules. Run tests in venv, fix failures, improve code. Update documentation per rules.

## Architecture Rules

- Dependencies flow strictly bottom-up.
- Configurations live at the class level in `configs/` next to importing class.
- One file = one class.
- Inheritance only through interfaces.

## Testing Rules

- Unit and integration tests at each project level in `tests/`.
- Every class must have unit tests.
- If a class imports another class, add integration tests at the importing class level.
- At least one integration test for each top-level use case.
- For every use case, include at least one full E2E test.

## Documentation Rules

- After each code change, update README.md files in corresponding folders.
- Root `README.md` must be human-readable and include project info, modules, dependencies, run/test instructions.
- Each folder has an LLM README based on `templates/readme_template.md`.
- Root has `llm_readme.md` based on `templates/llm_readme_template.md`.

## Naming Rules

- Classes: nouns, PascalCase, match file name.
- Functions/methods: verbs, snake_case (Python).
- Variables: nouns, snake_case.
- Constants: UPPER_SNAKE_CASE.
- Folders: lowercase with underscores or hyphens.

## Code Style Rules

- All files UTF-8.
- Docstrings required in all files.
- Consistent formatting (use linters/formatters).
- Line length 100-120 chars.
- Functions/methods max 50 lines.
- Classes max 300 lines.
- Avoid deep nesting (max 3-4 levels).
- Use explicit names, no magic numbers/strings.
- Remove unused code/comments.

## Error Handling

- Handle all errors explicitly; no silent failures.
- Use specific exception types.
- Log errors with enough context.
- User-facing errors must be clear.
- Critical errors must stop execution; non-critical errors need fallback logic.

## Dependency Management

- Pin all external dependencies in `requirements.txt` with versions.
- Keep dependencies minimal and from trusted sources.
- Keep dependencies updated for security.
- Document module dependencies explicitly.

## Security Rules

- No secrets in code or VCS.
- Use environment variables or secret managers.
- Validate and sanitize all user input.
- Use parameterized queries for DB.
- Regularly check dependencies for known vulnerabilities.

## Performance Rules

- Avoid premature optimization; consider perf in design.
- Cache frequently used data.
- Optimize DB queries (indexes, avoid N+1).
- Use async for I/O where applicable.
- Profile before optimizing.

## Git Rules

- Atomic, logically complete commits.
- Clear commit messages (what/why).
- Use branches for features/fixes.
- Run tests and linter before commit.
- Do not commit temp/build/log/IDE files.

## Refactoring Rules

- Separate refactors from new features.
- Ensure tests pass before and after refactor.
- Improve readability/maintainability without changing behavior.
- Document significant architecture changes.

## Code Review Rules

- All changes must be reviewed before merge.
- Review checks: project rules, code quality, test coverage.
- Feedback must be constructive and specific.
- Author must address all comments.
- At least one approval required to merge.

## Deployment Rules

- Changes must pass through test environment before production.
- Prefer CI/CD automation.
- Use SemVer for releases.
- Maintain changelog for significant changes.
- Have rollback plan for each deployment.

## Service-Specific Behavior

- Input JSON:
  - `request`: full prompt + context
  - `response`: LLM response
  - `entities_types`: list of entity type strings
- NER with `gliner` on both `request` and `response` using the provided entity types.
- `potential_hallucinations`: entities in response but not in request.
- `missing_entities`: entities in request but not in response.
- FastAPI app runs on port `1304` with OpenAPI docs enabled.

## Subagents (if supported)

- **architect**: Use for architecture design only (no implementation).
- **tdd-test-architect**: Use to create tests before implementation.
- **tdd-implementation-coder**: Use to implement based on task and pre-written tests.
