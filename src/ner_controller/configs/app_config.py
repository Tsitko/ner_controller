"""Application-level configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """Configuration values for the FastAPI application."""

    host: str = "0.0.0.0"
    port: int = 1304
    title: str = "LLM Hallucination Checker"
    docs_url: str = "/docs"
