"""Interface for entity extraction engines."""

from abc import ABC, abstractmethod
from typing import Sequence

from ner_controller.domain.entities.entity import Entity


class EntityExtractorInterface(ABC):
    """Contract for extracting entities from text."""

    @abstractmethod
    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[Entity]:
        """Extract entities of the given types from the text."""
        raise NotImplementedError
