"""
agents/orchestrator.py
──────────────────────
Iterative Generator-Validator loop for Phase 2.

For each Use Case, this module produces one final TC by alternating
between the Generation Agent and the Validation Agent until the TC
passes all four criteria, OR the loop is terminated by one of the
early-stopping rules:

- Convergence — TC passes all four criteria.
- Stagnation  — Three consecutive iterations produce the same set of
                  failing criteria, meaning the loop is oscillating.
- Degradation — TC length grows excessively (e.g., 10× the size of
                  the source Use Case), a sign the Generator is
                  fabricating content to placate the Validator.
- Budget      — MAX_ITERATIONS exhausted.

"""

import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from constants import MAX_ITERATIONS
from agents.generator import TCGenerator
from agents.validator import TCValidator, ValidationResult


# ──────────────────────────────────────────────────────────────────────
# Early-stopping configuration
# ──────────────────────────────────────────────────────────────────────
STAGNATION_WINDOW       = 3      
DEGRADATION_RATIO       = 10     
DEGRADATION_MIN_CHARS   = 4000   


# ──────────────────────────────────────────────────────────────────────
@dataclass
class IterationLog:
    """Per-iteration audit record."""
    iteration: int
    tc_text: str
    validation: dict
    elapsed_seconds: float


@dataclass
class GenerationOutcome:
    """Final result of one Use Case being processed by the loop."""
    uc_id: str
    uc_name: str
    final_tc: str
    converged: bool                   
    stop_reason: str                 
    iterations_used: int
    best_iteration: int               
    total_elapsed_seconds: float
    iterations: List[IterationLog] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ──────────────────────────────────────────────────────────────────────
def _failing_criteria_signature(verdict_dict: dict) -> tuple:
    """Returns a canonical tuple of the sub-criteria that FAILED.
    Used to detect oscillation across iterations."""
    keys = ("completeness", "atomicity", "clarity", "traceability")
    return tuple(k for k in keys if verdict_dict.get(k) is False)


def _passing_score(verdict_dict: dict) -> int:
    """Number of sub-criteria the Validator marked as PASS (0-4)."""
    keys = ("completeness", "atomicity", "clarity", "traceability")
    return sum(1 for k in keys if verdict_dict.get(k) is True)


def _use_case_chars(use_case: dict) -> int:
    """Heuristic measure of the Use Case 'size' in characters."""
    steps = use_case.get("steps", "")
    if isinstance(steps, list):
        steps_str = " ".join(str(s) for s in steps)
    else:
        steps_str = str(steps)
    return len(use_case.get("name", "")) \
         + len(use_case.get("description", "")) \
         + len(steps_str)


# ──────────────────────────────────────────────────────────────────────
class Phase2Orchestrator:
    """Runs the iterative Generator-Validator loop for every Use Case."""

    def __init__(self,
                 generator: TCGenerator,
                 validator: TCValidator,
                 max_iterations: int = MAX_ITERATIONS):
        self.gen = generator
        self.val = validator
        self.max_iter = max_iterations

    # ────────────────────────────────────────────────────────────────
    def process_one(self, use_case: dict) -> GenerationOutcome:
        """
        Generates and iteratively refines a TC for a single Use Case.

        Args:
            use_case : Dict with keys uc_id, name, description, steps.

        Returns:
            GenerationOutcome with the BEST TC seen and the full audit trail.
        """
        outcome = GenerationOutcome(
            uc_id   = use_case["uc_id"],
            uc_name = use_case["name"],
            final_tc   = "",
            converged  = False,
            stop_reason = "",
            iterations_used = 0,
            best_iteration  = 0,
            total_elapsed_seconds = 0.0,
        )

        t_total_start = time.time()

        # Initial generation
        tc_text = self.gen.generate(use_case)
        uc_size       = _use_case_chars(use_case)
        max_tc_chars  = max(DEGRADATION_MIN_CHARS, uc_size * DEGRADATION_RATIO)

        # Tracking for early stopping
        recent_signatures: List[tuple] = []
        best_score   = -1
        best_idx     = 0
        best_tc      = tc_text
        stop_reason  = "budget"    

        for i in range(1, self.max_iter + 1):
            t_iter_start = time.time()

            verdict: ValidationResult = self.val.validate(use_case, tc_text)
            elapsed = time.time() - t_iter_start

            v_dict = verdict.to_dict()
            outcome.iterations.append(IterationLog(
                iteration=i,
                tc_text=tc_text,
                validation=v_dict,
                elapsed_seconds=round(elapsed, 2),
            ))

            # Track best-seen TC by passing-criteria score
            score = _passing_score(v_dict)
            if score > best_score:
                best_score = score
                best_idx   = i
                best_tc    = tc_text

            # 1. Convergence
            if verdict.passed:
                stop_reason = "converged"
                best_idx    = i
                best_tc     = tc_text
                break

            # 2. Degradation — Generator inflating beyond the Use Case
            if len(tc_text) > max_tc_chars:
                stop_reason = "degradation"
                break

            # 3. Stagnation — Validator oscillating on the same failures
            signature = _failing_criteria_signature(v_dict)
            recent_signatures.append(signature)
            if len(recent_signatures) > STAGNATION_WINDOW:
                recent_signatures.pop(0)
            if (len(recent_signatures) == STAGNATION_WINDOW
                    and len(set(recent_signatures)) == 1
                    and signature != ()):
                stop_reason = "stagnation"
                break

            tc_text = self.gen.revise(previous_tc=tc_text,
                                      feedback=verdict.feedback)

        outcome.converged             = (stop_reason == "converged")
        outcome.stop_reason           = stop_reason
        outcome.iterations_used       = len(outcome.iterations)
        outcome.best_iteration        = best_idx
        outcome.final_tc              = best_tc
        outcome.total_elapsed_seconds = round(time.time() - t_total_start, 2)
        return outcome

    # ────────────────────────────────────────────────────────────────
    def process_batch(self,
                      use_cases: list,
                      output_dir: str,
                      log_filename: str = "phase2_log.jsonl") -> List[GenerationOutcome]:
        """
        Processes every Use Case sequentially and logs each outcome
        as a JSON line for auditability.
        """
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_filename)

        results: List[GenerationOutcome] = []
        with open(log_path, "w", encoding="utf-8") as log_f:
            for i, use_case in enumerate(use_cases, 1):
                print(f"\n[Phase 2] ({i}/{len(use_cases)}) "
                      f"{use_case['uc_id']} — {use_case['name']}")
                outcome = self.process_one(use_case)
                log_f.write(json.dumps(outcome.to_dict()) + "\n")
                log_f.flush()
                results.append(outcome)
                print(f"    → stop={outcome.stop_reason} "
                      f"(best_iter={outcome.best_iteration}), "
                      f"{outcome.iterations_used} iter(s), "
                      f"{outcome.total_elapsed_seconds:.1f}s")

        return results