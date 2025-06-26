import json
import os
import random
import uuid

import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()

DIFFICULTY_LEVELS = [
    "A. Простой (факт из 1 чанка)",
    "B. С явным использованием дат",
    "C. Требующий рассуждений/выводов",
]


def format_chunks_for_prompt(selected_chunks):
    """Форматирует выбранные чанки для подачи в промпт GPT."""
    prompt_chunks_str = "Контекстные чанки:\n"
    for i, chunk_data in enumerate(selected_chunks):
        prompt_chunks_str += f"--- Чанк {i+1} (ID: {chunk_data['chunk_id']}) ---\n"
        prompt_chunks_str += f"Дата публикации: {chunk_data['date']}\n"
        prompt_chunks_str += f"Текст: {chunk_data['text']}\n\n"
    return prompt_chunks_str


def generate_test_case_with_gpt4(selected_chunks):
    """
    Генерирует один тестовый пример с помощью GPT-4 на основе предоставленных чанков.
    """
    if not selected_chunks:
        return None

    chunk_context_str = format_chunks_for_prompt(selected_chunks)

    use_date_explicitly = len(selected_chunks) > 1 or random.choice([True, True, False])

    system_prompt = f"""
Ты — ассистент по созданию тестовых данных для RAG-системы СПбГУ.
Твоя задача — на основе предоставленных чанков текста и их метаданных (особенно даты публикации) сгенерировать тестовый пример.
Тестовый пример должен включать:
1.  `question_text`: Вопрос, ответ на который можно найти ТОЛЬКО в предоставленных чанках. Чанк ДОЛЖЕН полностью отвечать на вопрос. Вопрос должен быть сформулирован естественно, не должен быть избыточно специфичным, но и не слишком общим.
    Если `use_date_explicitly` будет True, постарайся сформулировать вопрос так, чтобы дата публикации была важна для ответа (например, "Что произошло после [дата]?", "Какая информация была актуальна на [дата]?"). При этом можешь использовать не только даты публикаций чанков, но и временные промежутки, включающие эти даты ("Что происходило в период с [дата1] по [дата2]?", "Что произошло за [месяц (и год)]"), но НЕ ИСПОЛЬЗУЙ промежутки больше двух месяцев.
2.  `expected_answer_ideal`: Идеальный, полный и фактически верный ответ, основанный ИСКЛЮЧИТЕЛЬНО на предоставленных чанках.
3.  `metadata_constraints`: JSON-объект с ограничениями по метаданным, которые RAG должен был бы учесть. Например, {{ "date_after": "YYYY-MM-DD", "date_exact": "YYYY-MM-DD" }}.
    Если чанк один, можно использовать `date_exact`. Если чанков несколько, можно использовать диапазон `date_after`, `date_before` или оставить пустым, если дата не ключевая.
4.  `difficulty_level`: Выбери один из предложенных уровней: {', '.join(DIFFICULTY_LEVELS)}.

Предоставленные чанки и их даты — единственный источник правды. Не придумывай информацию.
Если контекста недостаточно для хорошего вопроса или ответа, укажи это.
Верни результат в формате JSON.
"""

    user_prompt = f"""
{chunk_context_str}

Инструкция по датам: `use_date_explicitly` = {use_date_explicitly}

Сгенерируй тестовый пример в JSON-формате со следующими ключами:
"question_text", "expected_answer_ideal", "difficulty_level".
.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        # response_content = completion.choices[0].message.content
        # print("RAW GPT Response:", response_content) # Для отладки
        # generated_data = json.loads(response_content)

        generated_data = json.loads(completion.choices[0].message.content)

        test_case = {
            "question_id": str(uuid.uuid4()),
            "question_text": generated_data.get("question_text"),
            "expected_answer_ideal": generated_data.get("expected_answer_ideal"),
            "expected_relevant_chunk_ids": [
                chunk["chunk_id"] for chunk in selected_chunks
            ],
            "expected_relevant_chunk_texts": [
                chunk["text"] for chunk in selected_chunks
            ],
            "difficulty_level": generated_data.get("difficulty_level"),
        }

        if not all(
            [
                test_case["question_text"],
                test_case["expected_answer_ideal"],
                test_case["difficulty_level"],
            ]
        ):
            print(f"Warning: GPT-4 generated incomplete data: {generated_data}")
            return None

        return test_case

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from GPT: {e}")
        print(
            f"Raw response: {completion.choices[0].message.content if completion else 'No completion object'}"
        )
        return None
    except Exception as e:
        print(f"An error occurred while calling OpenAI API: {e}")
        return None


def select_chunks_for_generation(chunk_db, num_chunks_to_select=1):
    """Выбирает случайные чанки из базы. Можно усложнить логику."""
    return random.sample(chunk_db, num_chunks_to_select)


def generate_test_set(chunk_database, num_test_cases=10):
    test_set = []

    chunk_counts_distribution = [1] * 8 + [2] * 2

    for i in range(num_test_cases):
        print(f"Generating test case {i+1}/{num_test_cases}...")

        num_chunks_for_this_question = random.choice(chunk_counts_distribution)
        if len(chunk_database) < num_chunks_for_this_question:
            num_chunks_for_this_question = len(chunk_database)
            if num_chunks_for_this_question == 0:
                print("Chunk database is empty. Cannot generate test cases.")
                break

        selected_chunks = select_chunks_for_generation(
            chunk_database, num_chunks_to_select=num_chunks_for_this_question
        )

        if not selected_chunks:
            print("No chunks selected, skipping test case generation.")
            continue

        max_retries = 2
        for attempt in range(max_retries):
            print(
                f"  Attempt {attempt + 1} for test case {i+1} with {len(selected_chunks)} chunk(s)..."
            )
            test_case = generate_test_case_with_gpt4(selected_chunks)
            if test_case:
                test_set.append(test_case)
                print(f"  Successfully generated test case {i+1}.")
                break
            else:
                print(f"  Failed to generate test case {i+1} on attempt {attempt + 1}.")

        if not test_case:
            print(f"  Could not generate test case {i+1} after {max_retries} attempts.")

    return test_set


if __name__ == "__main__":
    import json

    chunks = json.load(open("data/chunks.json", "r", encoding="utf-8"))

    generated_dataset = generate_test_set(chunks, num_test_cases=200)

    if generated_dataset:

        with open("data/generated_test_set.json", "w", encoding="utf-8") as f:
            json.dump(generated_dataset, f, indent=2, ensure_ascii=False)
        print("\nTest set saved to generated_test_set.json")
    else:
        print("No test cases were generated.")
