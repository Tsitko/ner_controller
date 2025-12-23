"""Entity extracted from text."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    """Represents a named entity extracted by the NER engine."""

    text: str
    label: str
    start: int
    end: int
