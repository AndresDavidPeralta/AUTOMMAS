"""
evaluation/sampling.py
──────────────────────
Samples 30 Use Cases from the deduplicated evaluation set and
produces the blinded CSV that the QA experts will use to rate every
TC against the qualitative criteria (Atomicity, Clarity, Traceability).

The blinded CSV randomises the order of system outputs and removes any
identifier of the producing system. Each row contains:
    eval_id, uc_id (visible for context), name, tc_text,
    atomicity, clarity, traceability
The three rating columns are left empty for the evaluator to fill.
"""

import os
import json
import random
import pandas as pd

from constants import EVAL_SAMPLE_SIZE, RANDOM_SEED


# ──────────────────────────────────────────────────────────────────────
def sample_use_cases(deduped_csv: str,
                  output_csv: str,
                  n: int = EVAL_SAMPLE_SIZE,
                  seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Draws a simple random sample of N Use Cases from the
    deduplicated evaluation-set CSV.
    """
    df = pd.read_csv(deduped_csv)
    if len(df) < n:
        raise ValueError(f"Only {len(df)} Use Cases available; need {n}.")

    sample = df.sample(n=n, random_state=seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sample.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Sampling] {n} Use Cases saved to {output_csv}")
    return sample


# ──────────────────────────────────────────────────────────────────────
def build_blinded_pool(systems: dict,
                       output_csv: str,
                       blinding_map_path: str,
                       seed: int = RANDOM_SEED) -> None:
    """
    Combines TC outputs from every system into a single blinded pool
    and writes (i) the CSV the QA experts will rate and (ii) the
    de-blinding key as JSON.

    Args:
        systems : dict mapping system_name → DataFrame with columns
                  [uc_id, name, tc_text]. Example:
                  {
                    "multiagent":     df_mas,
                    "gemini":         df_pro,
                    "gpt":            df_4o,
                    …
                  }
        output_csv        : path for the blinded CSV.
        blinding_map_path : path for the JSON de-blinding key.
    """
    rng = random.Random(seed)
    pool = []

    for system_name, df in systems.items():
        for _, row in df.iterrows():
            pool.append({
                "system":   system_name,
                "uc_id": row["uc_id"],
                "name":     row["name"],
                "tc_text":  row["tc_text"],
            })

    rng.shuffle(pool)

    blinded_rows  = []
    blinding_map  = {}
    for idx, item in enumerate(pool, start=1):
        eval_id = f"EVAL-{idx:04d}"
        blinding_map[eval_id] = {
            "system":   item["system"],
            "uc_id": item["uc_id"],
        }
        blinded_rows.append({
            "eval_id":      eval_id,
            "uc_id":     item["uc_id"],
            "name":         item["name"],
            "tc_text":      item["tc_text"],
            "atomicity":    "",     
            "clarity":      "",     
            "traceability": "",     
        })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(blinded_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    with open(blinding_map_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "n_items": len(pool), "map": blinding_map},
                  f, indent=2)

    print(f"[Blinding] {len(pool)} TCs pooled and blinded.")
    print(f"[Blinding] Blinded CSV   : {output_csv}")
    print(f"[Blinding] De-blinding   : {blinding_map_path}")
