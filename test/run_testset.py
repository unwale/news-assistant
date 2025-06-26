import json
import time

import requests
from tqdm import tqdm

TEST_SET_FILE = "data/test_set.json"
RESULTS_FILE = "data/rag_evaluation_results.json"
K_RELEVANCE = 5

chunks = json.load(open("data/chunks.json", "r", encoding="utf-8"))
id2chunk = {chunk["chunk_id"]: chunk for chunk in chunks}


def query_my_rag_system(question: str, metadata_filters: dict = None) -> dict:
    response = requests.post(
        "http://localhost:8002/api/generate_with_context",
        json={
            "text": question,
        },
    ).json()

    generated_answer = response.get("message", "")
    context_chunks = response.get("context", [])
    retrieved_chunk_ids = [chunk["uuid"] for chunk in context_chunks]
    retrieved_chunk_texts = [chunk["content"] for chunk in context_chunks]
    retrieved_chunk_scores = [chunk["score"] for chunk in context_chunks]
    return {
        "generated_answer": generated_answer,
        "retrieved_chunk_ids": retrieved_chunk_ids[:K_RELEVANCE],
        "retrieved_chunk_texts": retrieved_chunk_texts[:K_RELEVANCE],
        "retrieved_chunk_scores": retrieved_chunk_scores[:K_RELEVANCE],
    }


def run_rag_on_test_set(test_set_path: str, rag_function: callable):
    try:
        with open(test_set_path, "r", encoding="utf-8") as f:
            test_items = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Тестовый файл {test_set_path} не найден.")
        return []

    evaluation_results = []
    print(f"Запуск RAG на {len(test_items)} тестовых примерах...")

    for item in tqdm(test_items, desc="Обработка тестовых примеров"):
        question = item["question_text"]

        try:
            rag_output = rag_function(question, metadata_constraints)

            result_entry = {
                "question_text": question,
                "difficulty_level": item["difficulty_level"],
                "expected_answer_ideal": item["expected_answer_ideal"],
                "expected_relevant_chunk_ids": [item["chunk_id"]],
                "expected_relevant_chunk_texts": [id2chunk[item["chunk_id"]]["text"]],
                "rag_generated_answer": rag_output.get("generated_answer", ""),
                "rag_retrieved_chunk_ids": rag_output.get("retrieved_chunk_ids", []),
                "rag_retrieved_chunk_texts": rag_output.get(
                    "retrieved_chunk_texts", []
                ),
                "rag_retrieved_chunk_scores": rag_output.get(
                    "retrieved_chunk_scores", []
                ),
            }
            evaluation_results.append(result_entry)
        except Exception as e:
            evaluation_results.append(
                {
                    "question_text": question,
                    "error": str(e),
                }
            )

    return evaluation_results


if __name__ == "__main__":
    all_rag_outputs = run_rag_on_test_set(TEST_SET_FILE, query_my_rag_system)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_rag_outputs, f, indent=2, ensure_ascii=False)

    print(f"\nСырые результаты RAG сохранены в {RESULTS_FILE}")
    print(f"Всего обработано и сохранено: {len(all_rag_outputs)} результатов.")
