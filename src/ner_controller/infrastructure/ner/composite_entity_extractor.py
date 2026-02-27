"""Composite entity extractor implementation."""

import logging
from typing import Sequence

from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface
from ner_controller.domain.services.levenshtein_utils import deduplicate_entities

logger = logging.getLogger(__name__)


class CompositeEntityExtractor(EntityExtractorInterface):
    """Combines multiple entity extractors and deduplicates results."""

    def __init__(self, extractors: Sequence[EntityExtractorInterface]) -> None:
        """
        Initialize the composite extractor.

        Args:
            extractors: Sequence of entity extractors to run in order.
        """
        if not extractors:
            raise ValueError("At least one extractor must be provided")

        self._extractors = extractors

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[str]:
        """
        Extract entities by running all extractors and combining results.

        Extractors are run in order, and all results are combined and deduplicated.

        Args:
            text: Text to extract entities from.
            entity_types: Entity types to extract.

        Returns:
            Deduplicated list of extracted entity strings.
        """
        if not text or not entity_types:
            return []

        all_entities = []

        # Run each extractor and collect results
        for extractor in self._extractors:
            try:
                entities = extractor.extract(text, entity_types)
            except Exception as exc:
                logger.error(
                    "Extractor %s failed: %s",
                    extractor.__class__.__name__,
                    exc,
                )
                continue
            all_entities.extend(entities)

        # Deduplicate using normalized Levenshtein distance
        return deduplicate_entities(all_entities)
