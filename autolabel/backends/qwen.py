"""Local Qwen3-VL backend: zero-shot, or fine-tuned with the Autolabel LoRA adapter + decision head.

    get_backend("qwen")                                   # zero-shot base model
    get_backend("qwen", adapter_path="runs/qwen2b/best")  # LoRA + decision head (autolabel.training)

With an adapter the concise prompt (FINAL_COC only) used in training is sent,
and the decision head's prediction is stored in label.meta["head_decisions"]
so the quality filter can cross-check it against the sentence (rule H01).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..prompt import build_prompt
from ..schemas import Window
from .base import LabelBackend


class QwenLoRABackend(LabelBackend):
    name = "qwen"
    needs_frames = True

    def __init__(
        self,
        base_model: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        min_pixels: int = 64 * 28 * 28,
        max_pixels: int = 256 * 28 * 28,
        image_max_side: int = 448,
        concise: Optional[bool] = None,
        tiny: bool = False,
    ):
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install torch transformers peft to use the qwen backend") from exc
        from ..training.collate import Collator
        from ..training.model import AutolabelModel
        from ..training.train import pick_device

        self.torch = torch
        self.adapter_path = adapter_path or os.environ.get("AUTOLABEL_QWEN_ADAPTER")
        self.base_model = base_model or os.environ.get("AUTOLABEL_QWEN_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
        self.device = pick_device(device)
        dtype = torch.float32 if self.device.type == "cpu" else (torch.bfloat16 if self.device.type == "cuda" else torch.float16)
        tiny_vlm = None
        if tiny:                                    # smoke tests: random weights, same architecture
            from ..training.model import make_tiny_vlm

            tiny_vlm = make_tiny_vlm(self.base_model)
            dtype = torch.float32
        if self.adapter_path:
            self.model = AutolabelModel.load(self.adapter_path, base_model=self.base_model, dtype=dtype, vlm=tiny_vlm).to(self.device)
            self.base_model = self.model.vlm.peft_config["default"].base_model_name_or_path or self.base_model
            self.name = "qwen+lora"
            self.concise = True if concise is None else concise
        else:
            from transformers import AutoModelForImageTextToText

            vlm = (tiny_vlm or AutoModelForImageTextToText.from_pretrained(self.base_model, dtype=dtype, low_cpu_mem_usage=True)).to(self.device)
            self.model = None
            self.vlm = vlm
            self.name = "qwen"
            self.concise = False if concise is None else concise
        self.processor = AutoProcessor.from_pretrained(self.base_model, min_pixels=min_pixels, max_pixels=max_pixels)
        self.collate = Collator(self.processor, image_max_side=image_max_side, with_target=False)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.image_max_side = image_max_side
        (self.model.vlm if self.model else self.vlm).eval()

    def generate(self, window: Window, frames: Optional[list[np.ndarray]] = None) -> str:
        from PIL import Image

        from ..training.collate import build_messages, to_device

        prompt = build_prompt(window, n_frames=len(frames or []), history_frames=window.history_frames, concise=self.concise)
        images = []
        for f in frames or []:
            im = Image.fromarray(f)
            im.thumbnail((self.image_max_side, self.image_max_side))
            images.append(im)
        inputs = self.processor.apply_chat_template(build_messages(prompt, images), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = to_device(dict(inputs), self.device)
        gen = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0}
        if self.temperature > 0:
            gen["temperature"] = self.temperature
        vlm = self.model.vlm if self.model else self.vlm
        with self.torch.no_grad():
            out = vlm.generate(**inputs, **gen)
        text = self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if self.model is not None:
            inputs["prompt_last_idx"] = self.torch.tensor([inputs["input_ids"].shape[1] - 1], device=self.device)
            lon, lat = self.model.predict_head(inputs)
            self._meta = {"head_decisions": {"longitudinal": lon[0], "lateral": lat[0]}, "adapter": str(self.adapter_path)}
        return text
