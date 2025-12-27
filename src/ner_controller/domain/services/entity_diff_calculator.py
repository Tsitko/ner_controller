"""Service for computing differences between entity sets."""

from typing import Sequence

from ner_controller.domain.entities.entity_diff_result import EntityDiffResult
from ner_controller.domain.services.levenshtein_utils import deduplicate_entities


class EntityDiffCalculator:
    """Calculates hallucinations and missing entities from extracted data."""

    def calculate(
        self,
        request_entities: Sequence[str],
        response_entities: Sequence[str],
    ) -> EntityDiffResult:
        """Compare request/response entities and return a diff result."""
        # Deduplicate both lists using Levenshtein distance
        unique_request = self._deduplicate(request_entities)
        unique_response = self._deduplicate(response_entities)

        # Find hallucinations (entities in response but not in request)
        hallucinations = [
            entity for entity in unique_response
            if not self._is_similar_to_any(entity, unique_request)
        ]

        # Find missing entities (entities in request but not in response)
        missing = [
            entity for entity in unique_request
            if not self._is_similar_to_any(entity, unique_response)
        ]

        return EntityDiffResult(
            potential_hallucinations=hallucinations,
            missing_entities=missing,
        )

    def _deduplicate(self, entities: Sequence[str]) -> list[str]:
        """Deduplicate entities using Levenshtein distance with threshold <= 2."""
        return deduplicate_entities(entities, threshold=2)

    def _is_similar_to_any(self, entity: str, entity_list: Sequence[str]) -> bool:
        """
        Check if entity is similar to any entity in the list.

        Uses case-insensitive comparison and Levenshtein distance.

        Args:
            entity: Entity to check.
            entity_list: List of entities to compare against.

        Returns:
            True if entity is similar (Levenshtein distance <= 2) to any entity in the list.
        """
        from ner_controller.domain.services.levenshtein_utils import levenshtein_distance

        entity_normalized = entity.strip().casefold()
        for other in entity_list:
            other_normalized = other.strip().casefold()
            # First try exact match (case-insensitive)
            if entity_normalized == other_normalized:
                return True
            # Then try Levenshtein distance
            if levenshtein_distance(entity_normalized, other_normalized) <= 2:
                return True
        return False

