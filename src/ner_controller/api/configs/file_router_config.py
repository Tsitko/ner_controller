"""Configuration for the file processing router."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FileRouterConfig:
    """Configuration values for the file processing endpoints."""

    prefix: str = "/file"
    tags: tuple[str, ...] = ("file-processing",)
