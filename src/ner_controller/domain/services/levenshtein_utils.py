"""Levenshtein distance utilities for entity deduplication."""

from typing import Sequence


def normalized_similarity(s1: str, s2: str, threshold_percent: float = 0.2) -> bool:
    """
    Check if two strings are similar using normalized Levenshtein distance.

    Similarity is determined by:
    1. Exact case-insensitive match
    2. Levenshtein distance <= max(3, len * threshold_percent)

    This approach handles both short strings (min absolute distance of 3)
    and long strings (relative threshold).

    Args:
        s1: First string.
        s2: Second string.
        threshold_percent: Maximum relative distance as percentage of string length.
                           Default is 0.2 (20%).

    Returns:
        True if strings are considered similar, False otherwise.

    Examples:
        >>> normalized_similarity("Apple", "Apples")  # distance=1, max(3, 5*0.2)=3 -> True
        True
        >>> normalized_similarity("сисма мониторинга", "системы мониторинга")  # distance~3, max(3, 16*0.2)=4 -> True
        True
    """
    s1_normalized = s1.strip().casefold()
    s2_normalized = s2.strip().casefold()

    # Exact match
    if s1_normalized == s2_normalized:
        return True

    # Calculate Levenshtein distance
    distance = levenshtein_distance(s1_normalized, s2_normalized)

    # Use relative threshold: max(3, length * 20%)
    max_length = max(len(s1_normalized), len(s2_normalized))
    threshold = max(3, int(max_length * threshold_percent))

    return distance <= threshold


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


def deduplicate_entities(entities: Sequence[str], threshold: int | None = None) -> list[str]:
    """
    Deduplicate entities using normalized Levenshtein distance.

    Uses relative similarity threshold (20% of string length, min 3) by default.
    For backward compatibility, accepts absolute threshold parameter.

    Entities are considered duplicates if they are similar according to
    normalized_similarity function.

    Preserves the order of first occurrences.

    Args:
        entities: Sequence of entity strings to deduplicate.
        threshold: Deprecated. Use None for default relative threshold.
                  If provided, uses absolute threshold for backward compatibility.

    Returns:
        List of deduplicated entities in original order of first occurrences.
    """
    if not entities:
        return []

    unique_entities: list[str] = []

    for entity in entities:
        # Check if this entity is similar to any already seen entity
        is_duplicate = False

        for seen_entity in unique_entities:
            if threshold is not None:
                # Legacy behavior: absolute threshold
                entity_normalized = entity.strip().casefold()
                seen_normalized = seen_entity.strip().casefold()
                if entity_normalized == seen_normalized:
                    is_duplicate = True
                    break
                if levenshtein_distance(entity_normalized, seen_normalized) <= threshold:
                    is_duplicate = True
                    break
            else:
                # New behavior: relative threshold
                if normalized_similarity(entity, seen_entity):
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_entities.append(entity.strip())

    return unique_entities
