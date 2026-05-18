"""
constants.py
─────────────
Central configuration for the complete pipeline.
Edit paths, model versions, and API model identifiers before running.
"""

import os

# ─── Hardware ─────────────────────────────────────────────────────────────────
GPU_DEVICE = 0   

# ─── Input data ───────────────────────────────────────────────────────────────
INPUT_PDF = "Directory path" 

# ─── Output directory layout ──────────────────────────────────────────────────
RESULTS_ROOT = "results"
LOGS_ROOT    = "logs"
CHUNKS_DIR   = "chunks"
ERRORS_DIR   = "errors"

DIRS = {
    "phase1_multimodal":  f"{RESULTS_ROOT}/phase1/multimodal",
    "phase1_ocr":         f"{RESULTS_ROOT}/phase1/ocr_baseline",
    "phase2_multiagent":  f"{RESULTS_ROOT}/phase2/multiagent",
    "phase2_gemini_pro":  f"{RESULTS_ROOT}/phase2/gemini_xxxx",
    "phase2_gpt4o":       f"{RESULTS_ROOT}/phase2/gpt_xxxx",
    "manual_eval":        f"{RESULTS_ROOT}/manual_eval",
    "metrics":            f"{RESULTS_ROOT}/metrics",
}

# ─── Phase 1: Multimodal extraction model ──────
MODEL_EXTRACTION = "Directory path" 

# ─── Phase 1 baseline: OCR + LLM (Llama-3.1-70B via llama-cpp-python) ────────
MODEL_OCR_LLM_GGUF = "Directory path" \
                     "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
OCR_BACKEND        = "paddleocr"   

# ─── Phase 2: TC Generator (Llama-3.1-70B via llama-cpp-python) ──────────────
MODEL_GENERATOR_GGUF = "Directory path" \
                       "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"
GENERATOR_N_GPU_LAYERS = -1    
GENERATOR_N_CTX        = 8192

# ─── Phase 2: TC Validator (Llama-3.3-70B via llama-cpp-python) ──────────────
MODEL_VALIDATOR_GGUF = "Directory path" \
                       "Llama-3.3-70B-Instruct-Q3_K_L.gguf"
VALIDATOR_N_GPU_LAYERS = -1   
VALIDATOR_N_CTX        = 8192

# ─── Generation hyperparameters ──────────────────────────────────────────────
GEN_PARAMS = {
    "extraction":  {"temperature": 0.0, "max_new_tokens": 1024},
    "generator":   {"temperature": 0.0, "max_tokens":     1024},
    "validator":   {"temperature": 0.0, "max_tokens":      512},
    "ocr_llm":     {"temperature": 0.0, "max_tokens":     1024},
}

# ─── Iterative loop parameters ────────────────────────────────────────────────
MAX_ITERATIONS = 20     # Phase 2 generator-validator loop limit

# ─── Train/test split ─────────────────────────────────────────────────────────
DEV_SPLIT_RATIO  = 0.70
EVAL_SAMPLE_SIZE = 30   # Use Cases sampled for manual evaluation
RANDOM_SEED      = 42

# ─── Frontier baselines ───────────────────────────────────────────────────────
BASELINES = {
    "gemini_2_5_pro":   "gemini-2.5-pro",
    "gpt_4o":           "gpt-4o",
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ─── PDF processing ───────────────────────────────────────────────────────────
MAX_IMAGES_PER_PAGE = 4
MIN_TEXT_LENGTH     = 150