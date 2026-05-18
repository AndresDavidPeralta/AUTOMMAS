"""
compute_metrics.py
──────────────────
Runs AFTER the QA experts complete their ratings.

Expected inputs:
  results/manual_eval/blinded_pool.csv       — RATED by QA experts 
  results/manual_eval/blinding_map.json      — produced during the pipeline
  results/phase2/multiagent/phase2_log.jsonl — produced during the pipeline

Expected ratings format (one row per evaluator per TC):
    eval_id, evaluator, atomicity, clarity, traceability

Output:
  results/metrics/phase2_metrics.json — full breakdown per system
"""

import json
import os
import pandas as pd

from constants import DIRS
from evaluation.metrics import aggregate_phase2


def main(ratings_csv: str = None):
    if ratings_csv is None:
        ratings_csv = os.path.join(DIRS["manual_eval"], "ratings_long.csv")

    blinding_map_path = os.path.join(DIRS["manual_eval"], "blinding_map.json")
    phase2_log = os.path.join(DIRS["phase2_multiagent"], "phase2_log.jsonl")

    with open(blinding_map_path) as f:
        blinding = json.load(f)["map"]

    ratings = pd.read_csv(ratings_csv)
    results = aggregate_phase2(ratings, phase2_log, blinding)

    out_path = os.path.join(DIRS["metrics"], "phase2_metrics.json")
    os.makedirs(DIRS["metrics"], exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Metrics] Saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
