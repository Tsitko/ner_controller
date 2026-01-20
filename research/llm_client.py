"""
LLM клиент для работы с LM Studio через OpenAI-совместимый API.
"""

import httpx
import requests
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Результат запроса к LLM."""
    content: str
    model: str
    processing_time: float  # в секундах
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """Клиент для LM Studio API."""

    def __init__(self, base_url: str = "http://192.168.1.16:1234"):
        self.base_url = base_url
        self.timeout = 300.0  # 5 минут для длинных запросов

    def get_models(self) -> list[dict]:
        """Получить список доступных моделей."""
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json().get("data", [])

    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Отправить запрос на chat completion."""
        start_time = time.time()

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        processing_time = time.time() - start_time

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", model),
            processing_time=processing_time,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0)
        )


class HallucinationChecker:
    """Клиент для NER Controller API (проверка галлюцинаций)."""

    def __init__(self, base_url: str = "http://localhost:1304"):
        self.base_url = base_url
        self.timeout = 120.0

    def check_hallucination(
        self,
        request_text: str,
        response_text: str,
        entity_types: Optional[list[str]] = None
    ) -> dict:
        """Проверить ответ LLM на галлюцинации."""
        if entity_types is None:
            entity_types = [
                "Person", "Location", "Organization", "Event",
                "Date", "Time", "Quantity"
            ]

        payload = {
            "request": request_text,
            "response": response_text,
            "entity_types": entity_types
        }

        response = requests.post(
            f"{self.base_url}/hallucination/check",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    # Тест получения моделей
    client = LLMClient()
    try:
        models = client.get_models()
        print("Доступные модели:")
        for m in models:
            print(f"  - {m.get('id', 'unknown')}")
    except Exception as e:
        print(f"Ошибка: {e}")
