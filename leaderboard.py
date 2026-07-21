"""
Once you've run evaluate.py for several models, aggregate all the
results/*.json files into one leaderboard table for the paper.

Usage:
    python leaderboard.py --results_dir results --out_csv leaderboard.csv
"""

import argparse
import glob
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--out_csv", default="leaderboard.csv")
    args = parser.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.results_dir, "*.json")):
        with open(path) as f:
            data = json.load(f)
        m = data["metrics"]
        row = {
            "model": data["model"],
            "num_samples": data["num_samples"],
            "overall_accuracy": m["overall_accuracy"],
            "unparseable_responses": m["unparseable_responses"],
            "result_file": os.path.basename(path),
        }
        for rel, stats in m.get("accuracy_by_question_relation", {}).items():
            row[f"acc_relation_{rel}"] = stats["accuracy"]
        for qt, stats in m.get("accuracy_by_question_type", {}).items():
            row[f"acc_type_{qt}"] = stats["accuracy"]
        rows.append(row)

    if not rows:
        print(f"No result files found in {args.results_dir}. Run evaluate.py first.")
        return

    df = pd.DataFrame(rows).sort_values("overall_accuracy", ascending=False)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved leaderboard to {args.out_csv}")


if __name__ == "__main__":
    main()
