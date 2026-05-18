"""
extract_use_cases.py
-----------------
Page-by-page use-case extraction from a PDF user manual.
Uses the Qwen2.5-VL model (via model.py) for multimodal inference.

Input strategy (hybrid):
  - The full page is rendered as a high-resolution image (200 dpi)
    so Qwen2.5-VL can perceive UI screenshots, icons, and layout.
  - The page's extracted text is also appended to the user prompt
    to guarantee that every word reaches the model, even icons or
    characters that the vision encoder might miss.

Both inputs together give the most robust Use Case extraction.
"""

import fitz                 
from PIL import Image
import json
import re
import os
import torch

from constants import MIN_TEXT_LENGTH

# ─── Prompts ──────────────────────────────────────────────────────────────────

UC_SYSTEM_PROMPT = (
    "You are an expert QA Automation Engineer specialized in reading technical "
    "documentation and user manuals. Your primary task is to identify and extract "
    "use cases: complete, multi-step user procedures that interact with the system "
    "to accomplish a single functional goal (as defined by Jacobson, 1992)."
)

UC_USER_PROMPT = """Analyze the provided page image AND text from a user manual for an Android device. Identify ALL 'use cases' on this page, where a 'use case' is a complete, multi-step procedure a user follows to achieve a specific goal (e.g., 'Set up voicemail', 'Connect to a Wi-Fi network', 'Take a screenshot').

IMPORTANT: A single page may contain MORE THAN ONE use case. Look carefully for every distinct procedure described, even if they share the same section heading. Each top-level instructional block (typically marked by a bold heading or a numbered step list) represents a separate use case.

For each use case you find on this page, extract the following information:
1. "name": A short, descriptive name for the use case.
2. "description": A one-sentence summary of its purpose.
3. "steps": A precise, ordered list of the user steps as described in the text.

If you do not find any complete use cases on the page, return an empty list [].

**FORMAT**: Format your entire output as a single, valid JSON list of objects. Do not add any text, markdown, or explanations before or after the JSON list.

Example format:
[
  {
    "name": "Set up voicemail",
    "description": "Procedure to configure the voicemail greeting and options.",
    "steps": [
      "1. Touch the Phone app.",
      "2. Touch & hold 1 to dial into your mailbox.",
      "3. Follow your carrier's system prompts."
    ]
  }
]
"""


# ─── JSON cleaning ────────────────────────────────────────────────────────────

def clean_json_response(text: str) -> str:
    """
    Extracts and sanitises the JSON payload from the model's raw response.
    Handles both ```json ... ``` fenced blocks and bare JSON output.
    Works with Llama, Qwen, and other instruction-tuned LLMs.
    """
    # 1. Strip markdown code fences (```json ... ``` or ``` ... ```)
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        json_str = match.group(1).strip()
    else:
        # 2. Bare JSON — find the outermost [ ... ]
        start = text.find('[')
        end   = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
        else:
            # 3. Fallback: strip role tokens some models emit
            json_str = re.split(r'\bassistant\b', text, flags=re.IGNORECASE)[-1].strip()

    json_str = re.sub(r'[\x00-\x1f\x7f]', ' ', json_str)
    return json_str.strip()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _render_page_image(page, dpi: int = 200) -> Image.Image:
    """Renders a PDF page as a single PIL image at the given DPI."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _build_user_prompt(page_text: str) -> str:
    """Concatenates the static instruction with the extracted page text."""
    snippet = page_text.strip()
    if not snippet:
        return UC_USER_PROMPT
    return (
        UC_USER_PROMPT
        + "\n\n[PAGE TEXT — extracted from PDF, in reading order]\n"
        + snippet
    )


# ─── Core processing ──────────────────────────────────────────────────────────

def process_pdf(pdf_path: str, model, errors_dir: str = "errors") -> list:
    """
    Processes a PDF file page by page to extract Use Cases.

    For every page:
      1. The page is rendered as a high-resolution image (200 dpi).
      2. The page's text is extracted from the PDF and appended to the prompt.
      3. Both go into a single multimodal call to Qwen2.5-VL.

    Args:
        pdf_path  : Path to the PDF file (single page recommended).
        model     : Initialised Model instance (Qwen2.5-VL).
        errors_dir: Directory where invalid JSON responses are saved.

    Returns:
        List of use-case dicts with fields {name, description, steps, source_page}.
    """
    all_use_cases = []
    doc = fitz.open(pdf_path)
    os.makedirs(errors_dir, exist_ok=True)

    config = {"max_new_tokens": 1024}

    for page_num, page in enumerate(doc):
        print(f"  · Page {page_num + 1}/{len(doc)}", end="  ")

        text = page.get_text("text")

 
        if len(text.strip()) < MIN_TEXT_LENGTH:
            print("skipped (insufficient content)")
            continue

        # ── Build inputs (hybrid: rendered page + text in prompt) ─────────
        try:
            page_image = _render_page_image(page, dpi=200)
            page_images = [page_image]
        except Exception as e:
            print(f"\n    ⚠ Page render failed: {e}")
            page_images = []

        user_prompt_full = _build_user_prompt(text)

        # ── Inference ─────────────────────────────────────────────────────
        try:
            response_text = model.forward(
                images=page_images,
                system_prompt=UC_SYSTEM_PROMPT,
                user_prompt=user_prompt_full,
                configuration=config,
            )
        except torch.cuda.OutOfMemoryError:
            print("OOM → retrying text-only with 512 tokens", end="  ")
            torch.cuda.empty_cache()
            try:
                response_text = model.forward(
                    images=[],
                    system_prompt=UC_SYSTEM_PROMPT,
                    user_prompt=user_prompt_full,
                    configuration={"max_new_tokens": 512},
                )
            except torch.cuda.OutOfMemoryError:
                print("OOM again → skipping page")
                continue

        # ── Parse JSON ────────────────────────────────────────────────────
        clean = clean_json_response(response_text)
        try:
            use_cases = json.loads(clean)
            if use_cases and isinstance(use_cases, list):
                for use_case in use_cases:
                    use_case["source_page"]     = page_num + 1
                    use_case["source_page_abs"] = page_num + 1
                    use_case["chunk_id"]        = os.path.splitext(
                                                    os.path.basename(pdf_path)
                                                  )[0]
                all_use_cases.extend(use_cases)
                print(f"found {len(use_cases)} use case(s)")
            else:
                print("no use cases found")
        except json.JSONDecodeError:
            err_file = os.path.join(errors_dir, f"page_{page_num + 1}.txt")
            with open(err_file, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"JSON error → saved to {err_file}")

    doc.close()
    return all_use_cases