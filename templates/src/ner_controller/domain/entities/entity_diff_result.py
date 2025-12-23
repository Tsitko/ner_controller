"""Result of comparing entities between request and response."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityDiffResult:
    """Computed differences between request and response entities."""

    potential_hallucinations: list[str]
    missing_entities: list[str]
