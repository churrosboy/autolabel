"""OpenAI-compatible chat backend (vLLM / Ollama / LM Studio serving a VLM).

Uses only the standard library so it works without extra packages.  Point it
at a server that hosts e.g. ``Qwen/Qwen3-VL-8B-Instruct`` (with or without a
merged LoRA) and it becomes the local auto-labeler.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Optional

import numpy as np

from ..prompt import build_prompt
from ..schemas import Window
from .base import LabelBackend


class OpenAICompatBackend(LabelBackend):
    name = "openai_compat"
    needs_frames = True

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 400,
        max_side: int = 640,
        timeout_s: float = 180.0,
    ):
        self.base_url = (base_url or os.environ.get("AUTOLABEL_OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("AUTOLABEL_OPENAI_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_side = max_side
        self.timeout_s = timeout_s
        self.name = f"openai:{self.model}"

    def _data_url(self, frame: np.ndarray) -> str:
        from PIL import Image

        img = Image.fromarray(frame)
        img.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(self, window: Window, frames: Optional[list[np.ndarray]] = None) -> str:
        prompt = build_prompt(window, n_frames=len(frames or []), history_frames=window.history_frames)
        content = [{"type": "image_url", "image_url": {"url": self._data_url(f)}} for f in (frames or [])]
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"].strip()
