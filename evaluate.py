"""
Run a model over the dataset and save per-sample + aggregate results.

Usage:
    python evaluate.py \
        --json_path /path/to/dataset.json \
        --base_dir  /path/to/dataset_root \
        --model qwen2.5-omni \
        --out_dir results

    # quick pipeline sanity check with no GPU/model needed:
    python evaluate.py --json_path data.json --base_dir . --model random --limit 20
"""

import argparse
import json
import os
from datetime import datetime

from dataset import AVQADataset
from metrics import compute_metrics


def get_model(name: str, **kwargs):
    """
    Model registry / factory. Imports are done lazily inside each branch
    so that missing dependencies for one model don't block others.
    Add new models here as you extend the benchmark.
    """
    name = name.lower()
    if name in ("qwen2.5-omni", "qwen_omni", "qwen2.5omni"):
        from models.qwen_omni_model import QwenOmniModel
        return QwenOmniModel(**kwargs)
    elif name == "random":
        from models.random_model import RandomBaselineModel
        return RandomBaselineModel(**kwargs)
    else:
        raise ValueError(
            f"Unknown model '{name}'. Add a branch for it in get_model() "
            f"in evaluate.py, backed by a models/<name>.py subclassing BenchmarkModel."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to the dataset JSON")
    parser.add_argument("--base_dir", default=None, required=True, help="Root dir the *_path fields are relative to")
    parser.add_argument("--model", required=True, help="Model key, e.g. qwen2.5-omni, random")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N samples (debug)")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    dataset = AVQADataset(args.json_path, args.base_dir)

    missing = dataset.missing_files()
    if missing:
        print(f"WARNING: {len(missing)} referenced media files were not found on disk.")
        print("First few:", missing[:5])

    samples = dataset.samples[: args.limit] if args.limit else dataset.samples
    model = get_model(args.model)

    os.makedirs(args.out_dir, exist_ok=True)
    per_sample_results = []

    for i, sample in enumerate(samples):
        result = model.predict(sample)
        correct = result.predicted_index == sample.answer
        per_sample_results.append(
            {
                "id": sample.id,
                "question_text": sample.question_text,
                "question_type": sample.question_type,
                "question_relation": sample.question_relation,
                "choices": sample.choices,
                "gt_answer": sample.answer,
                "predicted_index": result.predicted_index,
                "raw_response": result.raw_response,
                "correct": bool(correct),
                "error": result.error,
            }
        )
        status = "OK" if result.error is None else f"ERROR: {result.error}"
        print(f"[{i + 1}/{len(samples)}] id={sample.id} pred={result.predicted_index} "
              f"gt={sample.answer} correct={correct} ({status})")

    metrics = compute_metrics(per_sample_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model.name.replace("/", "_")
    out_path = os.path.join(args.out_dir, f"{safe_name}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": model.name,
                "num_samples": len(per_sample_results),
                "metrics": metrics,
                "per_sample_results": per_sample_results,
            },
            f,
            indent=2,
        )

    print("\n=== Summary ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
