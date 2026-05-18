"""
baselines/ocr_llm.py
────────────────────
Phase 1 baseline — text-only extraction pipeline.

  Step 1 — EasyOCR processes each PDF page into recognised text
  Step 2 — Llama-3.1-70B-Instruct (Q4_K_M, llama-cpp-python) extracts
            Use Cases from that text

Uses the same system + user prompts as the multimodal Extraction Agent
but receives no image input. This isolates the contribution of
multimodality.

Note: shares the same GGUF model file as the Phase 2 Generator
(Llama-3.1-70B-Instruct-Q4_K_M.gguf) — but is loaded in a separate
process to keep Phase 1 and Phase 2 independent.
"""

import fitz                 
from PIL import Image
import numpy as np
import io
import json
import os
import time

from llama_cpp import Llama

from constants import (
    MODEL_OCR_LLM_GGUF,
    GENERATOR_N_GPU_LAYERS,
    GENERATOR_N_CTX,
    GEN_PARAMS,
)
from extract_use_cases import UC_SYSTEM_PROMPT, UC_USER_PROMPT, clean_json_response


# ──────────────────────────────────────────────────────────────────────
def _init_easyocr():
    """Lazy-init EasyOCR to avoid import cost when not needed."""
    import easyocr
    # gpu=True #uses PyTorch CUDA backend.
    return easyocr.Reader(['en'], gpu=True, verbose=False)


def ocr_page(ocr_engine, page) -> str:
    """
    Renders a PDF page to an image and runs EasyOCR on it.
    Returns the concatenated recognised text in reading order.
    """
    pix = page.get_pixmap(dpi=200)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img_array = np.array(img)

    results = ocr_engine.readtext(img_array, paragraph=False)

    def _y_then_x(item):
        bbox = item[0]                         
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        return (sum(ys) / 4.0, sum(xs) / 4.0)   
    results.sort(key=_y_then_x)

    lines = []
    for item in results:
        if len(item) >= 2:
            txt = item[1].strip()
            if txt:
                lines.append(txt)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
class OCRLLMExtractor:
    """OCR + LLM Use Case extractor (Phase 1 baseline) using llama-cpp-python."""

    def __init__(self, model_path: str = MODEL_OCR_LLM_GGUF):
        print(f"[OCR Baseline] Loading EasyOCR …")
        self.ocr = _init_easyocr()

        print(f"[OCR Baseline] Loading {model_path}…")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=GENERATOR_N_GPU_LAYERS,   
            n_ctx=GENERATOR_N_CTX,
            verbose=False,
        )
        self.gen_params = GEN_PARAMS["ocr_llm"]
        print("[OCR Baseline] Ready.")

    # ────────────────────────────────────────────────────────────────
    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.gen_params["temperature"],
            max_tokens=self.gen_params["max_tokens"],
        )
        return out["choices"][0]["message"]["content"].strip()

    # ────────────────────────────────────────────────────────────────
    def process_pdf(self, pdf_path: str, log_path: str = None) -> list:
        """
        Runs OCR + LLM on every page of pdf_path and returns the
        concatenated list of extracted Use Cases.
        """
        all_use_cases = []
        doc = fitz.open(pdf_path)
        log_f = open(log_path, "w", encoding="utf-8") if log_path else None

        for page_num, page in enumerate(doc, start=1):
            t0 = time.time()
            text = ocr_page(self.ocr, page)
            if len(text.strip()) < 100:
                if log_f:
                    log_f.write(json.dumps({
                        "page": page_num, "skipped": True,
                        "reason": "OCR returned <100 chars"
                    }) + "\n")
                continue

            full_user_prompt = (
                f"{UC_USER_PROMPT}\n\n[PAGE TEXT — extracted via OCR]\n{text}"
            )
            response = self._chat(UC_SYSTEM_PROMPT, full_user_prompt)

            try:
                use_cases = json.loads(clean_json_response(response))
                if not isinstance(use_cases, list):
                    use_cases = []
            except json.JSONDecodeError:
                use_cases = []

            for u in use_cases:
                u["source_page"] = page_num
                u["source_page_abs"] = page_num
                u["chunk_id"] = os.path.splitext(os.path.basename(pdf_path))[0]
            all_use_cases.extend(use_cases)

            if log_f:
                log_f.write(json.dumps({
                    "page": page_num,
                    "elapsed_seconds": round(time.time() - t0, 2),
                    "use_cases_found": len(use_cases),
                    "raw_response": response[:500],
                }) + "\n")

        if log_f:
            log_f.close()
        doc.close()
        return all_use_cases