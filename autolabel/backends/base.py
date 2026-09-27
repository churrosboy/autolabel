"""Backend interface: turn a Window (+ optional frames) into a CoC label."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..prompt import parse_final_coc
from ..schemas import Label, Window


class LabelBackend(ABC):
    name: str = "base"
    needs_frames: bool = False

    @abstractmethod
    def generate(self, window: Window, frames: Optional[list[np.ndarray]] = None) -> str:
        """Return the *raw* model text (may contain THOUGHT_PROCESS...)."""

    def label(self, window: Window, frames: Optional[list[np.ndarray]] = None) -> Label:
        if self.needs_frames and not frames:
            raise ValueError(f"backend '{self.name}' needs frames but none were given")
        t0 = time.time()
        self._meta = {}
        raw = self.generate(window, frames)
        coc = parse_final_coc(raw) or ""
        return Label(
            clip_id=window.clip_id,
            index=window.index,
            keyframe_type=window.keyframe_type,
            first_frame_timestamp_us=int(window.frame_timestamps_us[0]) if window.frame_timestamps_us else int(window.start_us),
            keyframe_timestamp_us=int(window.keyframe.timestamp_us),
            coc=coc,
            backend=self.name,
            raw=raw,
            latency_s=round(time.time() - t0, 3),
            meta=dict(getattr(self, "_meta", {}) or {}),
        )

    def close(self) -> None:  # pragma: no cover
        pass
