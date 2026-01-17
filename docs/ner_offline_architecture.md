## Offline GLiNER Loading

### Goal

Prevent any network access during GLiNER model loading and inference. The service must
use only locally cached model artifacts and fail fast if they are missing.

### Design

- `GlinerEntityExtractorConfig` adds `offline_mode` and `offline_env_vars`.
- `GlinerEntityExtractor` applies offline environment variables during initialization
  and enforces `local_files_only=True` when `offline_mode` is enabled.
- If the local model is missing, model loading raises an explicit error instead of
  attempting a network download.

### Dependencies

- No new external dependencies.
