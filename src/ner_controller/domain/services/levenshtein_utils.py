"""Levenshtein distance utilities for entity deduplication."""

from typing import Sequence


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The minimum number of single-character edits (insertions, deletions, or substitutions)
        required to change s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def deduplicate_entities(entities: Sequence[str], threshold: int = 2) -> list[str]:
    """
    Deduplicate entities using Levenshtein distance and case-insensitive comparison.

    Entities are considered duplicates if their Levenshtein distance is <= threshold.
    Preserves the order of first occurrences.

    Args:
        entities: Sequence of entity strings to deduplicate.
        threshold: Maximum Levenshtein distance to consider entities as duplicates.
                  Default is 2.

    Returns:
        List of deduplicated entities in original order of first occurrences.
    """
    if not entities:
        return []

    unique_entities: list[str] = []

    for entity in entities:
        # Check if this entity is similar to any already seen entity
        is_duplicate = False
        entity_normalized = entity.strip().casefold()

        for seen_entity in unique_entities:
            seen_normalized = seen_entity.strip().casefold()
            # First try exact match (case-insensitive)
            if entity_normalized == seen_normalized:
                is_duplicate = True
                break
            # Then try Levenshtein distance
            if levenshtein_distance(entity_normalized, seen_normalized) <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_entities.append(entity.strip())

    return unique_entities
