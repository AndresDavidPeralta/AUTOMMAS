"""
agents/generator.py
───────────────────
TC Generation Agent — Llama-3.1-70B-Instruct
served via llama-cpp-python with all layers offloaded to the H200 NVL.

Produces a complete TC from a single Use Case, returning plain-text
output that follows the Hori (2010) structural rules.
"""

from llama_cpp import Llama

from constants import (
    MODEL_GENERATOR_GGUF,
    GENERATOR_N_GPU_LAYERS,
    GENERATOR_N_CTX,
    GEN_PARAMS,
)
from prompts.tc_prompts import (
    GENERATOR_SYSTEM_PROMPT,
    build_generator_prompt,
    build_revision_prompt,
)


class TCGenerator:
    """Wrapper around the Llama-3.1-70B Generation Agent (llama-cpp backend)."""

    def __init__(self, model_path: str = MODEL_GENERATOR_GGUF):
        print(f"[Generator] Loading {model_path}…")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=GENERATOR_N_GPU_LAYERS,   # -1 = all on GPU
            n_ctx=GENERATOR_N_CTX,
            verbose=False,
        )
        self.gen_params = GEN_PARAMS["generator"]
        print("[Generator] Ready.")

    # ────────────────────────────────────────────────────────────────
    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        """Single deterministic chat completion."""
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
    def generate(self, use_case: dict) -> str:
        """Produces the first TC version for the given Use Case."""
        return self._chat(
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=build_generator_prompt(use_case),
        )

    # ────────────────────────────────────────────────────────────────
    def revise(self, previous_tc: str, feedback: str) -> str:
        """Produces a revised TC given previous version and validator feedback."""
        return self._chat(
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=build_revision_prompt(previous_tc, feedback),
        )
