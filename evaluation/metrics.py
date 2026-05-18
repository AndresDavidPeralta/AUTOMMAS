"""
evaluation/metrics.py
─────────────────────
Computes the quantitative and qualitative metrics defined in Section 5.

  Phase 1
    - Extraction Validity Rate (EVR)

  Phase 2
    - Test Case Accuracy (TCA)        
    - Convergence Rate                
    - Average Iterations               
    - Atomicity / Clarity / Traceability 
    - Fleiss' Kappa                    
"""

import json
from typing import List, Dict
from collections import Counter
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# PHASE 1
# ──────────────────────────────────────────────────────────────────────

def extraction_validity_rate(ratings_df: pd.DataFrame,
                             rating_column: str = "valid") -> dict:
    """
    EVR = (Use Cases judged valid) / (Total Use Cases) × 100
    Expects a long-format DataFrame with one row per (uc_id, evaluator).
    Final validity is decided by majority vote across evaluators.
    """
    pivot = ratings_df.pivot_table(
        index="uc_id", columns="evaluator", values=rating_column, aggfunc="first"
    )

    def majority(row):
        votes = [v for v in row if pd.notna(v)]
        if not votes:
            return None
        c = Counter(votes)
        return c.most_common(1)[0][0]

    pivot["majority"] = pivot.apply(majority, axis=1)
    valid_count = (pivot["majority"] == "valid").sum()
    total = pivot["majority"].notna().sum()
    return {
        "valid": int(valid_count),
        "total": int(total),
        "evr":   round(100.0 * valid_count / total, 2) if total else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# PHASE 2 — automated metrics from orchestrator
# ──────────────────────────────────────────────────────────────────────

def phase2_automated_metrics(phase2_log_jsonl: str) -> dict:
    """
    Parses the Phase 2 orchestrator JSONL log and computes:
      - TCA              : % of TCs that converged (accepted by validator)
      - Convergence Rate : same as TCA (within 20-iteration budget)
      - Avg Iterations   : mean iterations used (across all TCs)
      - Avg Iterations (converged only)
    """
    converged_flags = []
    iterations      = []

    with open(phase2_log_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            converged_flags.append(bool(rec["converged"]))
            iterations.append(int(rec["iterations_used"]))

    n_total     = len(converged_flags)
    n_converged = sum(converged_flags)
    iter_conv   = [it for it, ok in zip(iterations, converged_flags) if ok]

    return {
        "total_tcs":          n_total,
        "converged":          n_converged,
        "tca":                round(100.0 * n_converged / n_total, 2) if n_total else 0.0,
        "convergence_rate":   round(100.0 * n_converged / n_total, 2) if n_total else 0.0,
        "avg_iterations_all": round(float(np.mean(iterations)), 2) if iterations else 0.0,
        "avg_iterations_conv":round(float(np.mean(iter_conv)),  2) if iter_conv else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# PHASE 2 — qualitative metrics from QA expert ratings
# ──────────────────────────────────────────────────────────────────────

def qualitative_majority(ratings_df: pd.DataFrame,
                         criterion: str) -> pd.DataFrame:
    """
    Returns a per-TC majority verdict for the given qualitative criterion.

    Input DataFrame is in form with columns:
        eval_id, evaluator, atomicity, clarity, traceability
    """
    pivot = ratings_df.pivot_table(
        index="eval_id", columns="evaluator", values=criterion, aggfunc="first"
    )

    def majority(row):
        votes = [v for v in row if pd.notna(v) and str(v).strip() != ""]
        if not votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    pivot["majority"] = pivot.apply(majority, axis=1)
    return pivot["majority"].reset_index()


def qualitative_pass_rate(ratings_df: pd.DataFrame,
                          criterion: str) -> dict:
    """
    Computes the proportion of TCs marked 'valid' by majority vote
    for the given criterion (atomicity, clarity, or traceability).
    """
    maj = qualitative_majority(ratings_df, criterion)
    valid = (maj["majority"] == "valid").sum()
    total = maj["majority"].notna().sum()
    return {
        "criterion": criterion,
        "valid": int(valid),
        "total": int(total),
        "pass_rate": round(100.0 * valid / total, 2) if total else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# FLEISS' KAPPA  (inter-rater agreement for ≥3 evaluators)
# ──────────────────────────────────────────────────────────────────────

def fleiss_kappa(ratings_df: pd.DataFrame,
                 criterion: str,
                 categories: List[str] = ("valid", "invalid")) -> float:
    """
    Computes Fleiss' Kappa for a single criterion across all evaluators.

    Expects long-format DataFrame columns:
        eval_id, evaluator, <criterion>
    """
    pivot = ratings_df.pivot_table(
        index="eval_id", columns="evaluator", values=criterion, aggfunc="first"
    )
    pivot = pivot.dropna(how="all")

    N, n = pivot.shape
    if N == 0 or n < 2:
        return float("nan")

    k = len(categories)
    M = np.zeros((N, k), dtype=float)
    for i, (_, row) in enumerate(pivot.iterrows()):
        votes = [v for v in row if pd.notna(v)]
        for j, cat in enumerate(categories):
            M[i, j] = votes.count(cat)

    n_eff = M.sum(axis=1)
    P_i = (np.sum(M ** 2, axis=1) - n_eff) / (n_eff * (n_eff - 1) + 1e-12)
    P_bar = P_i.mean()
    p_j = M.sum(axis=0) / M.sum()
    P_e = np.sum(p_j ** 2)

    if abs(1 - P_e) < 1e-12:
        return float("nan")
    return round(float((P_bar - P_e) / (1 - P_e)), 4)


# ──────────────────────────────────────────────────────────────────────
def aggregate_phase2(ratings_df: pd.DataFrame,
                     phase2_log_jsonl: str,
                     blinding_map: dict) -> dict:
    """
    End-to-end aggregator: combines QA ratings, the
    orchestrator log, and the blinding key to produce per-system
    qualitative pass rates and inter-rater agreement.
    """
    sys_map = {eid: meta["system"] for eid, meta in blinding_map.items()}
    df = ratings_df.copy()
    df["system"] = df["eval_id"].map(sys_map)

    results = {"systems": {}, "fleiss_kappa": {}}

    for criterion in ["atomicity", "clarity", "traceability"]:
        kappa = fleiss_kappa(df, criterion)
        results["fleiss_kappa"][criterion] = kappa

    for system, sub in df.groupby("system"):
        per_system = {}
        for criterion in ["atomicity", "clarity", "traceability"]:
            per_system[criterion] = qualitative_pass_rate(sub, criterion)
        results["systems"][system] = per_system


    if phase2_log_jsonl:
        results["multiagent_automated"] = phase2_automated_metrics(phase2_log_jsonl)

    return results
