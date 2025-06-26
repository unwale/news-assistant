def calculate_retrieval_metrics_for_item(expected_ids, actual_ids, k):
    if not expected_ids:
        return {
            "precision_at_k": 0,
            "recall_at_k": 0,
            "f1_at_k": 0,
            "hit_rate_at_k": 0,
            "mrr_at_k": 0,
        }

    top_k_actual_ids = actual_ids[:k]

    relevant_and_retrieved_count = len(set(expected_ids) & set(top_k_actual_ids))

    precision = (
        relevant_and_retrieved_count / len(top_k_actual_ids) if top_k_actual_ids else 0
    )

    recall = relevant_and_retrieved_count / len(expected_ids) if expected_ids else 0

    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    hit_rate = 1 if relevant_and_retrieved_count > 0 else 0

    mrr = 0.0
    for i, chunk_id in enumerate(top_k_actual_ids):
        if chunk_id in expected_ids:
            mrr = 1.0 / (i + 1)
            break

    return {
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        f"f1_at_{k}": f1,
        f"hit_rate_at_{k}": hit_rate,
        f"mrr_at_{k}": mrr,
    }


import json

with open("data/rag_evaluation_results.json", "r", encoding="utf-8") as f:
    all_rag_outputs = json.load(f)

processed_results_with_metrics = []
processed_results_with_metrics_no_censorship = []
for res_item in all_rag_outputs:
    if "error" in res_item:
        processed_results_with_metrics.append(res_item)
        continue

    retrieval_metrics = calculate_retrieval_metrics_for_item(
        res_item["expected_relevant_chunk_ids"],
        res_item["rag_retrieved_chunk_ids"],
        k=5,
    )
    res_item["retrieval_metrics"] = retrieval_metrics
    processed_results_with_metrics.append(res_item)
    # check if GigaChat or собственным мнением is in the answer
    if (
        "GigaChat" not in res_item["rag_generated_answer"]
        and "собственным мнением" not in res_item["rag_generated_answer"]
        and "ограничены" not in res_item["rag_generated_answer"]
    ):
        processed_results_with_metrics_no_censorship.append(res_item)
        print(len(processed_results_with_metrics_no_censorship))

with open(
    "data/rag_evaluation_results_with_retrieval_metrics.json", "w", encoding="utf-8"
) as f:
    json.dump(processed_results_with_metrics, f, indent=2, ensure_ascii=False)

print(
    f"\nРезультаты с метриками ретривера сохранены в rag_evaluation_results_with_retrieval_metrics.json"
)

avg_retrieval_metrics = {}
avg_retrieval_metrics_no_censorship = {}
counts = {}
counts_no_censorship = {}
for item in processed_results_with_metrics:
    if "retrieval_metrics" in item:
        for metric_name, value in item["retrieval_metrics"].items():
            avg_retrieval_metrics[metric_name] = (
                avg_retrieval_metrics.get(metric_name, 0) + value
            )
            counts[metric_name] = counts.get(metric_name, 0) + 1
for item in processed_results_with_metrics_no_censorship:
    if "retrieval_metrics" in item:
        for metric_name, value in item["retrieval_metrics"].items():
            avg_retrieval_metrics_no_censorship[metric_name] = (
                avg_retrieval_metrics_no_censorship.get(metric_name, 0) + value
            )
            counts_no_censorship[metric_name] = (
                counts_no_censorship.get(metric_name, 0) + 1
            )


for metric_name, total_value in avg_retrieval_metrics.items():
    avg_retrieval_metrics[metric_name] = (
        total_value / counts[metric_name] if counts[metric_name] > 0 else 0
    )

for metric_name, total_value in avg_retrieval_metrics_no_censorship.items():
    avg_retrieval_metrics_no_censorship[metric_name] = (
        total_value / counts_no_censorship[metric_name]
        if counts_no_censorship[metric_name] > 0
        else 0
    )

print("\n--- Средние метрики ретривера ---")
for metric, value in avg_retrieval_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n--- Средние метрики ретривера (без цензуры) ---")
for metric, value in avg_retrieval_metrics_no_censorship.items():
    print(f"{metric}: {value:.4f}")
