"""Service for computing differences between entity sets."""

from typing import Sequence

from ner_controller.domain.entities.entity import Entity
from ner_controller.domain.entities.entity_diff_result import EntityDiffResult


class EntityDiffCalculator:
    """Calculates hallucinations and missing entities from extracted data."""

    def calculate(
        self,
        request_entities: Sequence[Entity],
        response_entities: Sequence[Entity],
    ) -> EntityDiffResult:
        """Compare request/response entities and return a diff result."""
        request_map, request_order = self._deduplicate(request_entities)
        response_map, response_order = self._deduplicate(response_entities)

        hallucinations = [
            self._normalized_text(response_map[key].text)
            for key in response_order
            if key not in request_map
        ]
        missing = [
            self._normalized_text(request_map[key].text)
            for key in request_order
            if key not in response_map
        ]

        return EntityDiffResult(
            potential_hallucinations=hallucinations,
            missing_entities=missing,
        )

    def _deduplicate(
        self, entities: Sequence[Entity]
    ) -> tuple[dict[tuple[str, str], Entity], list[tuple[str, str]]]:
        """Return a map and order of unique entities by label and normalized text."""
        ordered_entities = sorted(entities, key=lambda entity: entity.start)
        result: dict[tuple[str, str], Entity] = {}
        order: list[tuple[str, str]] = []

        for entity in ordered_entities:
            key = self._entity_key(entity)
            if key in result:
                continue
            result[key] = entity
            order.append(key)

        return result, order

    def _entity_key(self, entity: Entity) -> tuple[str, str]:
        """Build a comparison key from entity label and normalized text."""
        return entity.label.casefold(), self._normalized_text(entity.text).casefold()

    def _normalized_text(self, text: str) -> str:
        """Normalize text for comparison and output."""
        return text.strip()
