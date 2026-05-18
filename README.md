# Pipelice:
End-to-end pipeline for *AUTOMMAS: Automating Test Case Generation from User Manuals via Multi-Agent Systems* (SBES 2026).

## Project Structure

```
pipeline_v2/
├── README.md                       # This file
├── constants.py                    # Central configuration
├── pdf_chunked.py                  # PDF -> 1-page chunks
├── data_split.py                   # 70/30 stratified split
├── model.py                        # Qwen2.5-VL wrapper (Phase 1)
├── extract_use_cases.py            # Use Case extraction (was extract_macros.py)
├── dedup_use_cases.py              # Use Case deduplication (was dedup_macros.py)
├── run_pipeline.py                 # Top-level orchestrator (baselines disabled)
├── test_smoke.py                   # NEW — smoke test on 3 known pages
├── compute_metrics.py              # Post-evaluation metric computation
├── requirements.txt
│
├── prompts/
│   └── tc_prompts.py               
│
├── agents/
│   ├── generator.py                # Llama-3.1-70B TC Generator
│   ├── validator.py                # Llama-3.3-70B TC Validator
│   └── orchestrator.py             # Iterative loop 
│
├── baselines/
│   ├── ocr_llm.py                  # Phase 1 baseline: PaddleOCR + Llama
│   └── frontier.py                 # Phase 2 baselines: Gemini + OpenAI
│                                   # (not executed by run_pipeline.py)
│
└── evaluation/
    ├── sampling.py                 # Random sample + blinded pool
    └── metrics.py                  # EVR, TCA, Fleiss' Kappa
```

---

## Pre-flight

### 1. Install dependencies

```bash
# CRITICAL: llama-cpp-python must be compiled with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python

pip install -r requirements.txt
```

### 2. Verify model paths in `constants.py`

```python
MODEL_EXTRACTION     = "<path to Qwen2.5-VL-7B-Instruct>"
MODEL_GENERATOR      = "<path to Llama-3.1-70B-Instruct-IQ2_M.gguf>"
MODEL_OCR_LLM        = "<path to Llama-3.2-11B-Instruct>"
MODEL_VALIDATOR_GGUF = "<path to Llama-3.3-70B-Instruct-Q3_K_L.gguf>"
INPUT_PDF            = "<path to user manual PDF>"
```

> **NOTE:** Verify `MODEL_GENERATOR` points to the 70B GGUF.

### 3. (Optional) Set API keys if you want to run baselines locally

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
```

Otherwise, the frontier baseline outputs must be produced externally and dropped into:
```
results/phase2/gemini_2_5_pro/tcs.csv
results/phase2/gpt_4o/tcs.csv
```
(Same schema as the multi-agent output: `uc_id, name, tc_text`.)

---

## Execution

### Step  1- Run Pipeline
```bash
python run_pipeline.py
```

Phases executed (note: Phase 2b baselines are now **loaded externally**, not generated locally):

| Phase | Action |
|---|---|
| 0 | PDF split + 70/30 partition |
| 1a | Multimodal extraction (Qwen) on the ~111 evaluation-set pages |
| 1b | OCR + LLM baseline on the same pages |
| 1c | Deduplication |
| 1d | Random sample of 30 Use Cases |
| 2a | Multi-agent generation (30 × up to 20 iter) |
| 2b | **Loads pre-existing baseline CSVs from disk** |
| 3 | Builds the blinded pool for QA experts |


### Step 2 — QA experts rate the blinded pool

Output: `results/manual_eval/blinded_pool.csv` (150 rows, 3 empty rating columns).

### Step 3 — Compute final metrics

Once ratings are collected (long-form CSV: `eval_id,evaluator,atomicity,clarity,traceability`):

```bash
python compute_metrics.py
```

Output: `results/metrics/phase2_metrics.json` — per-system pass rates and Fleiss' Kappa.

---
