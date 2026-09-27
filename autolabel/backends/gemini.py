"""Gemini (google-genai) VLM backend."""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from ..prompt import build_prompt
from ..schemas import Window
from .base import LabelBackend


class GeminiBackend(LabelBackend):
    name = "gemini"
    needs_frames = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_side: int = 768,
        retries: int = 3,
        sleep_s: float = 1.0,
    ):
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install google-genai to use the gemini backend") from exc
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("AUTOLABEL_GEMINI_MODEL", "gemini-3.1-pro-preview")
        self.temperature = temperature
        self.max_side = max_side
        self.retries = retries
        self.sleep_s = sleep_s
        self.name = f"gemini:{self.model}"

    def _to_pil(self, frames: list[np.ndarray]):
        from PIL import Image

        out = []
        for f in frames:
            img = Image.fromarray(f)
            img.thumbnail((self.max_side, self.max_side))
            out.append(img)
        return out

    def generate(self, window: Window, frames: Optional[list[np.ndarray]] = None) -> str:
        prompt = build_prompt(window, n_frames=len(frames or []), history_frames=window.history_frames)
        cfg = self._types.GenerateContentConfig(
            temperature=self.temperature,
            safety_settings=[
                self._types.SafetySetting(category=c, threshold="BLOCK_NONE")
                for c in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                )
            ],
        )
        images = self._to_pil(frames or [])
        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=images + [prompt], config=cfg)
                time.sleep(self.sleep_s)
                return (resp.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc).lower()
                if "429" in msg or "quota" in msg or "exhausted" in msg or "503" in msg:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"gemini failed after {self.retries} retries: {last_err}")
