"""Configuration for the hallucination router."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HallucinationRouterConfig:
    """Configuration values for the hallucination endpoints."""

    prefix: str = "/hallucination"
    tags: tuple[str, ...] = ("hallucination",)
