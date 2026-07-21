from collections import defaultdict


def compute_metrics(per_sample_results):
    total = len(per_sample_results)
    correct = sum(r["correct"] for r in per_sample_results)
    unparseable = sum(1 for r in per_sample_results if r["predicted_index"] is None)
    errored = sum(1 for r in per_sample_results if r.get("error"))

    by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_relation = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in per_sample_results:
        t = r["question_type"] or "unknown"
        rel = r["question_relation"] or "unknown"
        by_type[t]["total"] += 1
        by_type[t]["correct"] += int(r["correct"])
        by_relation[rel]["total"] += 1
        by_relation[rel]["correct"] += int(r["correct"])

    def with_accuracy(d):
        return {
            k: {**v, "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0}
            for k, v in d.items()
        }

    return {
        "overall_accuracy": round(correct / total, 4) if total else 0.0,
        "total_samples": total,
        "unparseable_responses": unparseable,
        "inference_errors": errored,
        "accuracy_by_question_type": with_accuracy(by_type),
        "accuracy_by_question_relation": with_accuracy(by_relation),
    }
