"""
agents/validator.py
───────────────────
TC Validation Agent — Llama-3.3-70B-Instruct served via
llama-cpp-python with all layers offloaded to the H200 NVL.

Acts as an independent LLM-as-a-judge that evaluates each generated
Test Case against four criteria: completeness, atomicity, clarity,
and traceability.
"""

import re
from dataclasses import dataclass

from llama_cpp import Llama

from constants import (
    MODEL_VALIDATOR_GGUF,
    VALIDATOR_N_GPU_LAYERS,
    VALIDATOR_N_CTX,
    GEN_PARAMS,
)
from prompts.tc_prompts import (
    VALIDATOR_SYSTEM_PROMPT,
    build_validator_prompt,
)


# ──────────────────────────────────────────────────────────────────────
@dataclass
class ValidationResult:
    """Structured outcome of a single validation call."""
    completeness: bool
    atomicity:    bool
    clarity:      bool
    traceability: bool
    overall:      bool
    feedback:     str
    raw:          str          

    @property
    def passed(self) -> bool:
        return self.overall

    def to_dict(self) -> dict:
        return {
            "completeness": self.completeness,
            "atomicity":    self.atomicity,
            "clarity":      self.clarity,
            "traceability": self.traceability,
            "overall":      self.overall,
            "feedback":     self.feedback,
        }


# ──────────────────────────────────────────────────────────────────────
class TCValidator:
    """Wrapper around Llama-3.3-70B Validator via llama-cpp-python."""

    def __init__(self, model_path: str = MODEL_VALIDATOR_GGUF):
        print(f"[Validator] Loading {model_path}…")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=VALIDATOR_N_GPU_LAYERS,   
            n_ctx=VALIDATOR_N_CTX,
            verbose=False,
        )
        self.gen_params = GEN_PARAMS["validator"]
        print("[Validator] Ready.")

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
    @staticmethod
    def _parse(raw: str) -> ValidationResult:
        """Parses the validator's structured output into a ValidationResult."""

        def pick(label: str) -> bool:
            m = re.search(rf'{label}\s*:\s*(PASS|FAIL)', raw, re.IGNORECASE)
            return bool(m and m.group(1).upper() == "PASS")

        feedback_match = re.search(r'Feedback\s*:\s*(.+)', raw,
                                   flags=re.IGNORECASE | re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else raw.strip()

        return ValidationResult(
            completeness=pick("Completeness"),
            atomicity   =pick("Atomicity"),
            clarity     =pick("Clarity"),
            traceability=pick("Traceability"),
            overall     =pick("Overall"),
            feedback    =feedback,
            raw         =raw,
        )

    # ────────────────────────────────────────────────────────────────
    def validate(self, use_case: dict, tc_text: str) -> ValidationResult:
        """Validates a single TC against its originating Use Case."""
        raw = self._chat(
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            user_prompt=build_validator_prompt(use_case, tc_text),
        )
        return self._parse(raw)
