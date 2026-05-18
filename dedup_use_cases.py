"""
dedup_use_cases.py
---------------
Post-processing deduplication for the use-case extraction pipeline.

Strategy
────────
1. Case-normalize names           merges capitalisation variants
2. Apply semantic merge map       merges confirmed near-duplicates
3. Select BEST entry per group    highest step count, longest description
4. Reassign sequential uc_ids     sUC-0001 … UC-NNNN
5. Export clean CSV

Usage (standalone):
    python dedup_use_cases.py \
        --input  results/use_cases_all.csv \
        --output results/use_cases_deduped.csv
"""

import argparse
import re
import pandas as pd


SEMANTIC_MERGE_MAP = {
    "change the device's ringtone": ["change the ringtone"],
    "change the screen lock settings": ["change the screen lock"],
    "set up a new contact": ["add a contact"],
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    return str(name).lower().strip()


def count_steps(steps_str) -> int:
    if not isinstance(steps_str, str) or not steps_str.strip():
        return 0
    return len([s for s in steps_str.split("\n") if s.strip()])


def fix_tech_terms(text: str) -> str:
    """Enforce correct capitalisation for technical terms."""
    text = re.sub(r'\bsim\b',   'SIM',   text, flags=re.IGNORECASE)
    text = re.sub(r'\bwi-fi\b', 'Wi-Fi', text, flags=re.IGNORECASE)
    text = re.sub(r'\besim\b',  'eSIM',  text, flags=re.IGNORECASE)
    return text


def select_best(group: pd.DataFrame) -> pd.Series:
    """
    Returns the row with the most steps (tiebreak: longest description).
    """
    g = group.copy()
    g["_n"]  = g["steps"].apply(count_steps)
    g["_dl"] = g["description"].fillna("").apply(len)
    best = g.sort_values(["_n", "_dl"], ascending=False).iloc[0].copy()
    best.drop(labels=["_n", "_dl"], inplace=True)
    return best


def apply_semantic_merge(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces alias normalised names with their canonical counterpart."""
    reverse = {alias: canon
               for canon, aliases in SEMANTIC_MERGE_MAP.items()
               for alias in aliases}
    df = df.copy()
    df["name_norm"] = df["name_norm"].replace(reverse)
    return df


# ─── Main deduplication function ──────────────────────────────────────────────

def deduplicate(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    original_count = len(df)

    # Step 1 – case normalization
    df["name_norm"] = df["name"].apply(normalize_name)

    # Step 2 – semantic merge
    df = apply_semantic_merge(df)

    # Step 3 – best-row selection per group
    best_rows = (
        df.groupby("name_norm", sort=False)
          .apply(select_best, include_groups=False)
          .reset_index()
    )

    # Step 4 – canonical name: use SEMANTIC_MERGE_MAP key or original name
    def get_display_name(row) -> str:
        norm = row["name_norm"]
        base = (SEMANTIC_MERGE_MAP.get(norm)  # should not trigger (keys = canonicals)
                or norm) if norm in SEMANTIC_MERGE_MAP else row["name"]
        base = base[0].upper() + base[1:] if base else base
        return fix_tech_terms(base)

    best_rows["name"] = best_rows.apply(get_display_name, axis=1)
    best_rows.drop(columns=["name_norm"], inplace=True)

    # Step 5 – sort by first appearance and reassign IDs
    best_rows = best_rows.sort_values("source_page_abs").reset_index(drop=True)
    best_rows["uc_id"] = [f"UC-{i + 1:04d}" for i in range(len(best_rows))]

    cols = ["uc_id", "name", "description", "steps",
            "source_page", "source_page_abs", "chunk_id"]
    best_rows = best_rows[[c for c in cols if c in best_rows.columns]]

    best_rows.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print("  DEDUPLICATION SUMMARY")
    print("=" * 55)
    print(f"  Input rows           : {original_count:>6}")
    print(f"  After case norm      : {df['name_norm'].nunique():>6}")
    print(f"  After semantic merge : {len(best_rows):>6}")
    print(f"  Removed              : {original_count - len(best_rows):>6}")
    print("=" * 55)
    for _, row in best_rows.iterrows():
        n = count_steps(row["steps"])
        print(f"  {row['uc_id']}  {row['name']:<45} ({n} steps)")
    print(f"\n  ✓ Saved to: {output_path}")
    return best_rows


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="results/use_cases_all.csv")
    parser.add_argument("--output", default="results/use_cases_deduped.csv")
    args = parser.parse_args()
    deduplicate(args.input, args.output)
