"""Regex-based API endpoint extractor implementation."""

import re
from typing import Sequence

from ner_controller.domain.interfaces.entity_extractor_interface import EntityExtractorInterface


class RegexApiEndpointExtractor(EntityExtractorInterface):
    """Extracts API endpoints from text using regex patterns."""

    API_ENDPOINT_PATTERN = re.compile(
        r'\b(?:POST|GET|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+[a-zA-Z0-9/_-]+',
        re.IGNORECASE
    )

    def extract(self, text: str, entity_types: Sequence[str]) -> Sequence[str]:
        """
        Extract API endpoints from text.

        Args:
            text: Text to extract endpoints from.
            entity_types: Entity types to extract (not used in regex extraction,
                         but kept for interface compatibility).

        Returns:
            List of extracted API endpoint strings.
        """
        if not text:
            return []

        matches = self.API_ENDPOINT_PATTERN.findall(text)

        # Normalize endpoints (collapse multiple spaces to single space, strip)
        normalized_matches = []
        for match in matches:
            # Collapse multiple whitespace characters to single space
            normalized = re.sub(r'\s+', ' ', match.strip())
            if normalized:
                normalized_matches.append(normalized)

        return normalized_matches
