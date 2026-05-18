"""
run_pipeline.py
────────────────
Top-level orchestrator: runs the complete experimental pipeline in
the order required by the paper.

  Phase 0   — Split PDF, 70/30 train/test
  Phase 1a  — Multimodal extraction (Qwen2.5-VL) on the EVAL set
  Phase 1b  — OCR + LLM baseline                 on the EVAL set
  Phase 1c  — Deduplication of the EVAL Use Cases
  Phase 1d  — Random sample of 30 Use Cases (for blinded human evaluation)
  Phase 2a  — Multi-agent TC generation on ALL deduplicated Use Cases
  Phase 2b  — Four frontier zero-shot baselines  (on the 30-sample)
  Phase 3   — Build the blinded pool for QA expert evaluation

Usage:
    python run_pipeline.py
"""

import os
import json
import pandas as pd

from constants import (
    INPUT_PDF, CHUNKS_DIR, DIRS, BASELINES, MODEL_EXTRACTION
)
from pdf_chunked import split_pdf_into_chunks
from data_split  import split_chunks
from extract_use_cases import process_pdf
from dedup_use_cases   import deduplicate
from model          import Model          

from agents.generator    import TCGenerator
from agents.validator    import TCValidator
from agents.orchestrator import Phase2Orchestrator

from baselines.frontier  import GeminiBaseline, OpenAIBaseline, run_baseline
from baselines.ocr_llm   import OCRLLMExtractor

from evaluation.sampling import sample_use_cases, build_blinded_pool


# ──────────────────────────────────────────────────────────────────────
def ensure_dirs():
    for path in DIRS.values():
        os.makedirs(path, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
def phase0_split():
    print("\n=== PHASE 0 — PDF chunking + 70/30 split ===")
    chunks = split_pdf_into_chunks(INPUT_PDF, out_dir=CHUNKS_DIR, chunk_size=1)
    dev, evalset = split_chunks(
        chunks,
        save_to=os.path.join(DIRS["phase1_multimodal"], "split.json"),
    )
    print(f"  Total chunks: {len(chunks)} | dev: {len(dev)} | eval: {len(evalset)}")
    return dev, evalset


# ──────────────────────────────────────────────────────────────────────
def phase1a_multimodal(eval_chunks):
    print("\n=== PHASE 1a — Multimodal extraction (Qwen2.5-VL) ===")
    model = Model(model_path=MODEL_EXTRACTION)
    raw = []
    for i, ch in enumerate(eval_chunks, 1):
        print(f"  ({i}/{len(eval_chunks)}) {os.path.basename(ch)}")
        raw.extend(process_pdf(ch, model,
                               errors_dir=os.path.join(DIRS["phase1_multimodal"], "errors")))
    raw_csv = os.path.join(DIRS["phase1_multimodal"], "use_cases_raw.csv")
    pd.DataFrame(raw).to_csv(raw_csv, index=False, encoding="utf-8-sig")

    deduped_csv = os.path.join(DIRS["phase1_multimodal"], "use_cases_deduped.csv")
    df_final = deduplicate(input_path=raw_csv, output_path=deduped_csv)

    # Release VRAM before next phases
    del model
    return df_final, deduped_csv


# ──────────────────────────────────────────────────────────────────────
def phase1b_ocr(eval_chunks):
    print("\n=== PHASE 1b — OCR + LLM baseline ===")
    ocr_baseline = OCRLLMExtractor()
    all_use_cases = []
    for i, ch in enumerate(eval_chunks, 1):
        print(f"  ({i}/{len(eval_chunks)}) {os.path.basename(ch)}")
        log = os.path.join(DIRS["phase1_ocr"], f"{os.path.basename(ch)}.jsonl")
        all_use_cases.extend(ocr_baseline.process_pdf(ch, log_path=log))
    out = os.path.join(DIRS["phase1_ocr"], "use_cases_raw.csv")
    pd.DataFrame(all_use_cases).to_csv(out, index=False, encoding="utf-8-sig")
    del ocr_baseline
    return out


# ──────────────────────────────────────────────────────────────────────
def phase1d_sample(deduped_csv):
    """
    Random sample of 30 Use Cases.

    These 30 Use Cases define the BLINDED EVALUATION POOL — they are the
    ones rated by QA experts and the ones for which the frontier baselines
    generate Test Cases. They are a SUBSET of the full 227 deduplicated 
    Use Cases that Phase 2a will process.
    """
    print("\n=== PHASE 1d — Random sample of 30 Use Cases ===")
    sample_csv = os.path.join(DIRS["manual_eval"], "sampled_use_cases.csv")
    return sample_use_cases(deduped_csv, output_csv=sample_csv)


# ──────────────────────────────────────────────────────────────────────
def phase2a_multiagent(use_cases_df):
    """
    Multi-agent TC generation for EVERY Use Case in the input DataFrame.
    """
    print("\n=== PHASE 2a — Multi-agent TC generation ===")
    print(f"  → Will process {len(use_cases_df)} Use Cases "
          f"(~{len(use_cases_df) * 10 / 60:.1f} min estimated)")

    gen = TCGenerator()
    val = TCValidator()
    orch = Phase2Orchestrator(gen, val)

    use_cases = use_cases_df.to_dict(orient="records")
    outcomes = orch.process_batch(
        use_cases,
        output_dir=DIRS["phase2_multiagent"],
        log_filename="phase2_log.jsonl",
    )

    # Full results CSV including audit columns
    rows = [{
        "uc_id":           o.uc_id,
        "name":            o.uc_name,
        "tc_text":         o.final_tc,
        "converged":       o.converged,
        "stop_reason":     o.stop_reason,
        "iterations":      o.iterations_used,
        "best_iteration":  o.best_iteration,
        "elapsed_seconds": o.total_elapsed_seconds,
    } for o in outcomes]
    csv_path = os.path.join(DIRS["phase2_multiagent"], "tcs.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    del gen, val, orch
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
def phase2b_baselines(sample_df):
    """Run the four frontier zero-shot baselines on the 30-Use-Case sample."""
    print("\n=== PHASE 2b — Frontier zero-shot baselines ===")
    use_cases = sample_df.to_dict(orient="records")
    outputs = {}

    for key, model_id in BASELINES.items():
        target_dir = DIRS[f"phase2_{key}"]
        baseline = (
            GeminiBaseline(model_id) if key.startswith("gemini")
            else OpenAIBaseline(model_id)
        )
        results = run_baseline(baseline, use_cases, target_dir,
                               log_filename=f"{key}_log.jsonl")
        rows = [{"uc_id": r["uc_id"], "name": r["name"], "tc_text": r["tc_text"]}
                for r in results]
        csv_path = os.path.join(target_dir, "tcs.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        outputs[key] = pd.DataFrame(rows)

    return outputs


# ──────────────────────────────────────────────────────────────────────
def phase3_blinding(mas_df_30, baseline_dfs):
    """
    Build the blinded evaluation pool.

    Input: 30-row DataFrames per system (multiagent + 4 baselines).
    Output: shuffled, anonymised CSV with 150 rows for QA experts.
    """
    print("\n=== PHASE 3 — Build blinded pool for QA experts ===")
    systems = {"multiagent": mas_df_30, **baseline_dfs}
    build_blinded_pool(
        systems=systems,
        output_csv       =os.path.join(DIRS["manual_eval"], "blinded_pool.csv"),
        blinding_map_path=os.path.join(DIRS["manual_eval"], "blinding_map.json"),
    )


# ──────────────────────────────────────────────────────────────────────
def _load_external_baselines(sample_df):
    """
    Loads frontier baseline TC outputs produced externally by a collaborator.

    Expected layout (one CSV per baseline, same schema as Phase 2a output):
        results/phase2/gemini_2_5_pro/tcs.csv
        results/phase2/gpt_4o/tcs.csv

    If any of those CSVs is missing, that baseline is skipped (a warning
    is printed) so the pipeline can still complete with the baselines
    that ARE available.

    NOTE — DIRS keys use shortened names (phase2_gemini_pro), while
    BASELINES keys are the long names (gemini_2_5_pro). This function
    maps between them.
    """
    print("\n=== PHASE 2b — Loading external frontier baseline outputs ===")
    key_to_dir = {
        "gemini_2_5_pro":   "phase2_gemini_pro",
        "gemini_2_5_flash": "phase2_gemini_flash",
        "gpt_5":            "phase2_gpt5",
        "gpt_4o":           "phase2_gpt4o",
    }
    outputs = {}
    for key in BASELINES.keys():
        dir_key = key_to_dir.get(key)
        if dir_key is None or dir_key not in DIRS:
            print(f"  ⚠ {key} — no matching DIRS entry, skipping")
            continue
        csv_path = os.path.join(DIRS[dir_key], "tcs.csv")
        if os.path.exists(csv_path):
            outputs[key] = pd.read_csv(csv_path)
            print(f"  ✓ Loaded {key} ({len(outputs[key])} TCs)")
        else:
            print(f"  ⚠ {key} not found at {csv_path} — skipping")
    return outputs


# ──────────────────────────────────────────────────────────────────────
def main():
    ensure_dirs()

    dev_chunks, eval_chunks = phase0_split()

    # Phase 1a — multimodal extraction over the EVAL set, then dedup
    df_full, deduped_csv = phase1a_multimodal(eval_chunks)

    # Phase 1b — OCR + LLM baseline over the same EVAL set
    phase1b_ocr(eval_chunks)

    # Phase 1d — sample of 30 Use Cases for the blinded human evaluation
    sample_df = phase1d_sample(deduped_csv)

    # Phase 2a — generate TCs for ALL deduplicated Use Cases (publication
    # artefact). The 30 sampled ones are a subset of these results.
    mas_df_full = phase2a_multiagent(df_full)

    print("\n" + "=" * 60)
    print("  PIPELINE PHASE 1 + 2a COMPLETE")
    print("  Outputs:")
    print(f"    • {len(mas_df_full)} multi-agent Test Cases (full set)")
    print(f"    • {len(sample_df)} Use Cases sampled for blinded evaluation")
    print(f"    • results/phase1/multimodal/use_cases_deduped.csv")
    print(f"    • results/phase1/ocr_baseline/use_cases_raw.csv")
    print(f"    • results/manual_eval/sampled_use_cases.csv")
    print(f"    • results/phase2/multiagent/tcs.csv")
    print(f"    • results/phase2/multiagent/phase2_log.jsonl")
    print()
    print("  Awaiting frontier baseline CSVs from collaborator before")
    print("  building the blinded pool and computing metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()