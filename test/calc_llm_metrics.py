import json
import os

from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import answer_correctness  # Сравнивает с expected_answer_ideal
from ragas.metrics import answer_similarity  # Сравнивает с expected_answer_ideal
from ragas.metrics import (
    context_precision,  # Оценивает, все ли в извлеченном контексте релевантно вопросу
)
from ragas.metrics import answer_relevancy, context_recall, faithfulness

load_dotenv()

try:
    with open(
        "data/rag_evaluation_results_with_retrieval_metrics.json", "r", encoding="utf-8"
    ) as f:
        results_for_ragas_input = json.load(f)
except FileNotFoundError:
    print("Файл с результатами для RAGAS не найден. Запустите предыдущие шаги.")
    exit()

dataset_dict = {
    "question": [],
    "answer": [],  # Фактический ответ от RAG
    "contexts": [],  # Фактически извлеченные чанки (тексты)
    "ground_truth": [],  # expected_answer_ideal (для answer_correctness, answer_similarity)
}
original_data_map = []

for i, item in enumerate(results_for_ragas_input):
    if "error" in item or not item.get("rag_generated_answer"):
        continue

    dataset_dict["question"].append(item["question_text"])
    dataset_dict["answer"].append(item["rag_generated_answer"])
    dataset_dict["contexts"].append(item["rag_retrieved_chunk_texts"])
    dataset_dict["ground_truth"].append(item.get("expected_answer_ideal", ""))

    results_for_ragas_input[i]["question_id"] = item.get("question_id", f"q{i+1}")

    original_data_map.append(
        {
            "question_id": item["question_id"],
        }
    )

if not dataset_dict["question"]:
    print(
        "Нет данных для оценки RAGAS (возможно, все RAG ответы были пустыми или с ошибками)."
    )
else:
    ragas_dataset = Dataset.from_dict(dataset_dict)

    print(f"\nЗапуск оценки RAGAS на {len(ragas_dataset)} примерах...")

    ragas_result = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,  # Насколько ответ основан на контексте
            answer_relevancy,  # Насколько ответ релевантен вопросу
            context_precision,  # Насколько извлеченный контекст точен/релевантен вопросу
            context_recall,  # Насколько извлеченный контекст покрывает идеальный ответ
            answer_similarity,  # Сходство ответа с эталонным (семантическое)
            answer_correctness,  # Фактическая корректность ответа по отношению к эталонному
        ],
    )

    print("\n--- Результаты RAGAS (средние значения) ---")
    print(ragas_result)

    merged_results_final = []
    ragas_df = ragas_result.to_pandas()
    for i, original_item_info in enumerate(original_data_map):
        full_item_data = next(
            item
            for item in results_for_ragas_input
            if item["question_id"] == original_item_info["question_id"]
        )
        if i < len(ragas_df):
            ragas_scores_for_item = ragas_df.iloc[i].to_dict()
            full_item_data["ragas_scores"] = ragas_scores_for_item
        merged_results_final.append(full_item_data)

    with open("data/rag_evaluation_results_FINAL.json", "w", encoding="utf-8") as f:
        json.dump(merged_results_final, f, indent=2, ensure_ascii=False)
    print(
        "\nФинальные результаты с RAGAS метриками сохранены в rag_evaluation_results_FINAL.json"
    )
