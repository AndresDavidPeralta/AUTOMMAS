"""
baselines/frontier.py
─────────────────────
Zero-shot TC generation through frontier LLM APIs:
  - Gemini(Google Generative AI SDK)
  - GPT-5 (OpenAI Python SDK)
"""

import time
import json
import os
from typing import Optional

from constants import GEMINI_API_KEY, OPENAI_API_KEY
from prompts.tc_prompts import (
    GENERATOR_SYSTEM_PROMPT,
    build_generator_prompt,
)


# ──────────────────────────────────────────────────────────────────────
class GeminiBaseline:
    """Zero-shot baseline backed by Google's Gemini family."""

    def __init__(self, model_name: str):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=GENERATOR_SYSTEM_PROMPT,
            generation_config={"temperature": 0.0},
        )
        self.model_name = model_name
        print(f"[Baseline] Gemini ready: {model_name}")

    def generate(self, use_case: dict) -> str:
        prompt = build_generator_prompt(use_case)
        resp = self.model.generate_content(prompt)
        return resp.text.strip() if resp.text else ""


# ──────────────────────────────────────────────────────────────────────
class OpenAIBaseline:
    """Zero-shot baseline backed by OpenAI's GPT-5 / GPT-4o."""

    def __init__(self, model_name: str):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        print(f"[Baseline] OpenAI ready: {model_name}")

    def generate(self, use_case: dict) -> str:
        prompt = build_generator_prompt(use_case)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────────────
def run_baseline(baseline,
                 use_cases: list,
                 output_dir: str,
                 log_filename: str = "baseline_log.jsonl") -> list:
    """
    Runs a baseline sequentially over the Use Case list and
    persists every (input, output, elapsed) tuple to a JSONL log.

    Args:
        baseline     : Instance of GeminiBaseline or OpenAIBaseline.
        use_cases    : List of Use Case dicts.
        output_dir   : Directory where the JSONL log is written.
        log_filename : Log file name.

    Returns:
        List of result dicts with {uc_id, name, tc_text, elapsed_seconds}.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    results = []
    with open(log_path, "w", encoding="utf-8") as log_f:
        for i, use_case in enumerate(use_cases, 1):
            print(f"  [{baseline.model_name}] ({i}/{len(use_cases)}) {use_case['name']}")
            t0 = time.time()
            try:
                tc_text = baseline.generate(use_case)
                error   = None
            except Exception as e:
                tc_text = ""
                error   = str(e)
            elapsed = round(time.time() - t0, 2)

            record = {
                "uc_id":         use_case["uc_id"],
                "name":             use_case["name"],
                "model":            baseline.model_name,
                "tc_text":          tc_text,
                "elapsed_seconds":  elapsed,
                "error":            error,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            results.append(record)

    return results
