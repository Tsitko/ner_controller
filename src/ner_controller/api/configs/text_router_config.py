"""Configuration for the text processing router."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextRouterConfig:
    """Configuration values for the text processing endpoints."""

    prefix: str = "/text"
    tags: tuple[str, ...] = ("text-processing",)
