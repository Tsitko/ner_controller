"""Domain result for hallucination detection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HallucinationDetectionResult:
    """Domain output for hallucination detection use case."""

    potential_hallucinations: list[str]
    missing_entities: list[str]
