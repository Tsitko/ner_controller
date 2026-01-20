"""
Скрипт для исследования уменьшения галлюцинаций LLM с использованием NER-модели.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from llm_client import LLMClient, HallucinationChecker, LLMResponse


# Маппинг моделей из задачи на реальные ID в LM Studio
MODEL_MAPPING = {
    "qwen2.5-7b": "qwen2.5-coder-7b-instruct-mlx",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-next": "qwen/qwen3-next-80b",
    "llama4-70b": "nousresearch/hermes-4-70b",
}

# Типы сущностей для проверки галлюцинаций
ENTITY_TYPES = [
    "Person", "Location", "Organization", "Event",
    "Date", "Time", "Quantity", "Concept"
]

INITIAL_PROMPT = """Analyze the Declaration of Independence and provide a brief summary of each main section.

Structure your response as follows:
1. Preamble - brief description
2. Philosophical foundations - brief description
3. List of grievances against King George III - brief description of main categories
4. Conclusion and declaration of independence - brief description

Declaration:
{declaration}
"""

CORRECTION_PROMPT = """You previously provided the following analysis of the Declaration of Independence:

{first_response}

The hallucination control system detected the following issues:
- Potential hallucinations (entities in your response not present in the original): {hallucinations}
- Missing entities (important entities from the original not mentioned in your response): {missing}

Original text of the Declaration:
{declaration}

Please correct your response:
1. Remove or correct information about entities marked as hallucinations
2. Add mentions of missing important entities where appropriate
3. Ensure all facts correspond to the original text

Provide corrected analysis:
"""


def load_declaration() -> str:
    """Загрузить текст декларации."""
    declaration_path = Path(__file__).parent / "declaration.txt"
    return declaration_path.read_text(encoding="utf-8")


def run_experiment(
    llm_client: LLMClient,
    hallucination_checker: HallucinationChecker,
    model_name: str,
    model_id: str,
    declaration: str
) -> dict:
    """Провести эксперимент для одной модели."""
    print(f"\n{'='*60}")
    print(f"Модель: {model_name} ({model_id})")
    print(f"{'='*60}")

    result = {
        "model_name": model_name,
        "model_id": model_id,
        "timestamp": datetime.now().isoformat(),
    }

    # Шаг 1: Первоначальный запрос к LLM
    print("\n[1/4] Отправка первоначального запроса к LLM...")
    initial_messages = [
        {"role": "user", "content": INITIAL_PROMPT.format(declaration=declaration)}
    ]

    try:
        first_response: LLMResponse = llm_client.chat_completion(
            messages=initial_messages,
            model=model_id,
            temperature=0.7
        )
        result["first_response"] = {
            "content": first_response.content,
            "processing_time": first_response.processing_time,
            "tokens": first_response.total_tokens
        }
        print(f"    Время: {first_response.processing_time:.2f}с")
        print(f"    Токены: {first_response.total_tokens}")
    except Exception as e:
        print(f"    ОШИБКА: {e}")
        result["first_response"] = {"error": str(e)}
        return result

    # Шаг 2: Проверка на галлюцинации
    print("\n[2/4] Проверка на галлюцинации через NER Controller...")
    try:
        hallucination_result = hallucination_checker.check_hallucination(
            request_text=declaration,
            response_text=first_response.content,
            entity_types=ENTITY_TYPES
        )
        result["hallucination_check"] = hallucination_result
        print(f"    Потенциальные галлюцинации: {len(hallucination_result.get('potential_hallucinations', []))}")
        print(f"    Пропущенные сущности: {len(hallucination_result.get('missing_entities', []))}")
    except Exception as e:
        print(f"    ОШИБКА: {e}")
        result["hallucination_check"] = {"error": str(e)}
        return result

    # Шаг 3: Повторный запрос с исправлениями
    print("\n[3/4] Отправка запроса на исправление галлюцинаций...")
    hallucinations = hallucination_result.get("potential_hallucinations", [])
    missing = hallucination_result.get("missing_entities", [])

    correction_messages = [
        {
            "role": "user",
            "content": CORRECTION_PROMPT.format(
                first_response=first_response.content,
                hallucinations=", ".join(hallucinations) if hallucinations else "none detected",
                missing=", ".join(missing) if missing else "none detected",
                declaration=declaration
            )
        }
    ]

    try:
        corrected_response: LLMResponse = llm_client.chat_completion(
            messages=correction_messages,
            model=model_id,
            temperature=0.5  # Снижаем температуру для более точного ответа
        )
        result["corrected_response"] = {
            "content": corrected_response.content,
            "processing_time": corrected_response.processing_time,
            "tokens": corrected_response.total_tokens
        }
        print(f"    Время: {corrected_response.processing_time:.2f}с")
        print(f"    Токены: {corrected_response.total_tokens}")
    except Exception as e:
        print(f"    ОШИБКА: {e}")
        result["corrected_response"] = {"error": str(e)}
        return result

    # Шаг 4: Повторная проверка на галлюцинации
    print("\n[4/4] Повторная проверка на галлюцинации...")
    try:
        final_hallucination_check = hallucination_checker.check_hallucination(
            request_text=declaration,
            response_text=corrected_response.content,
            entity_types=ENTITY_TYPES
        )
        result["final_hallucination_check"] = final_hallucination_check
        print(f"    Потенциальные галлюцинации: {len(final_hallucination_check.get('potential_hallucinations', []))}")
        print(f"    Пропущенные сущности: {len(final_hallucination_check.get('missing_entities', []))}")
    except Exception as e:
        print(f"    ОШИБКА: {e}")
        result["final_hallucination_check"] = {"error": str(e)}

    # Расчет суммарного времени
    first_time = result.get("first_response", {}).get("processing_time", 0)
    corrected_time = result.get("corrected_response", {}).get("processing_time", 0)
    result["total_time"] = first_time + corrected_time

    print(f"\n    Суммарное время: {result['total_time']:.2f}с")

    return result


def main():
    """Основная функция исследования."""
    print("=" * 60)
    print("ИССЛЕДОВАНИЕ: Уменьшение галлюцинаций LLM с NER")
    print("=" * 60)

    # Инициализация клиентов
    llm_client = LLMClient(base_url="http://192.168.1.16:1234")
    hallucination_checker = HallucinationChecker(base_url="http://localhost:1304")

    # Загрузка декларации
    declaration = load_declaration()
    print(f"\nДекларация загружена: {len(declaration)} символов")

    # Проверка доступности сервисов
    print("\nПроверка доступности сервисов...")
    try:
        models = llm_client.get_models()
        available_models = [m.get("id", "") for m in models]
        print(f"  LM Studio: OK ({len(models)} моделей)")
    except Exception as e:
        print(f"  LM Studio: ОШИБКА - {e}")
        return

    # Результаты
    results = []

    # Запуск экспериментов
    for model_name, model_id in MODEL_MAPPING.items():
        if model_id not in available_models:
            print(f"\n⚠ Модель {model_name} ({model_id}) недоступна, пропускаем")
            results.append({
                "model_name": model_name,
                "model_id": model_id,
                "error": "Модель недоступна"
            })
            continue

        try:
            result = run_experiment(
                llm_client=llm_client,
                hallucination_checker=hallucination_checker,
                model_name=model_name,
                model_id=model_id,
                declaration=declaration
            )
            results.append(result)
        except Exception as e:
            print(f"\nОШИБКА при обработке {model_name}: {e}")
            results.append({
                "model_name": model_name,
                "model_id": model_id,
                "error": str(e)
            })

    # Сохранение результатов
    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Результаты сохранены в: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
