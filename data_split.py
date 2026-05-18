"""
data_split.py
─────────────
Stratified random split of PDF page-chunks into 70% development
and 30% evaluation sets. Reproducible via RANDOM_SEED.
"""

import os
import json
import random
from typing import List, Tuple

from constants import DEV_SPLIT_RATIO, RANDOM_SEED


def split_chunks(chunk_paths: List[str],
                 dev_ratio: float = DEV_SPLIT_RATIO,
                 seed: int = RANDOM_SEED,
                 save_to: str = None) -> Tuple[List[str], List[str]]:
    """
    Splits the chunk list into (dev_set, eval_set).

    Args:
        chunk_paths : Full ordered list of chunk PDF paths.
        dev_ratio   : Fraction allocated to development set (default 0.70).
        seed        : Random seed for reproducibility.
        save_to     : If provided, dumps the split as JSON for auditing.

    Returns:
        (dev_chunks, eval_chunks) — two lists of file paths.
    """
    random.seed(seed)
    shuffled = list(chunk_paths)
    random.shuffle(shuffled)

    cut = int(len(shuffled) * dev_ratio)
    dev_set  = sorted(shuffled[:cut])
    eval_set = sorted(shuffled[cut:])

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "w") as f:
            json.dump({
                "seed": seed,
                "dev_ratio": dev_ratio,
                "n_total": len(chunk_paths),
                "n_dev":   len(dev_set),
                "n_eval":  len(eval_set),
                "dev":  [os.path.basename(p) for p in dev_set],
                "eval": [os.path.basename(p) for p in eval_set],
            }, f, indent=2)

    return dev_set, eval_set
