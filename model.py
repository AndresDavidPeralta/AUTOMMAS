"""
model.py
--------
Qwen2.5-VL vision-language model wrapper.

Exposes the same interface as the original Llama model class so that
extract_use_cases.py and main.py require zero modifications:

    from model import Model
    m = Model(model_path=MODEL)
    text = m.forward(images=[...], system_prompt="...",
                     user_prompt="...", configuration={})
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from constants import GPU_DEVICE

try:
    from qwen_vl_utils import process_vision_info
    _QWEN_UTILS = True
except ImportError:
    _QWEN_UTILS = False
    print("[Model] Warning: qwen-vl-utils not found. "
          "Run: pip install qwen-vl-utils\n"
          "Falling back to manual image extraction (slightly lower quality).")


class Model:
    """
    Wraps Qwen2.5-VL-7B-Instruct for multimodal (text + image) inference.
    Drop-in replacement for the original Llama-based Model class.
    """

    def __init__(self, model_path: str):
        self.target_device = f"cuda:{GPU_DEVICE}"
        print(f"[Model] Loading Qwen2.5-VL from: {model_path}")
        print(f"[Model] Target GPU : {self.target_device}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": GPU_DEVICE},   # pin all layers to H200
        )
        self.model.eval()

        # min/max_pixels: recommended range for Qwen2-VL document processing
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        print("[Model] Ready.\n")

    # ──────────────────────────────────────────────────────────────────────
    def _build_messages(
        self,
        images: list,
        system_prompt: str,
        user_prompt: str,
    ) -> list:
        """Builds the OpenAI-style message list for Qwen's chat template."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]
        user_content = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        images: list,
        system_prompt: str,
        user_prompt: str,
        configuration: dict = None,
    ) -> str:
        """
        Runs one multimodal inference call and returns the response string.

        Args:
            images        : List of PIL Images from the PDF page ([] for text-only).
            system_prompt : System-role instruction.
            user_prompt   : User-role instruction.
            configuration : Overrides for generation kwargs
        Returns:
            Raw text response from the model.
        """
        gen_kwargs = {"max_new_tokens": 1024, "do_sample": False}
        if configuration:
            gen_kwargs.update(configuration)

        messages = self._build_messages(images, system_prompt, user_prompt)

        # ── Tokenise ──────────────────────────────────────────────────────
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ── Vision inputs ─────────────────────────────────────────────────
        if _QWEN_UTILS:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [
                item["image"]
                for msg in messages
                for item in (msg["content"] if isinstance(msg["content"], list) else [])
                if isinstance(item, dict) and item.get("type") == "image"
            ] or None
            video_inputs = None

        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.target_device)

        # ── Generate ──────────────────────────────────────────────────────
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # ── Decode (trim prompt tokens) ───────────────────────────────────
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
