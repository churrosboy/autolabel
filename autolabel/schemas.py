"""Plain dataclasses shared across the pipeline (no torch / pandas dependency)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np


@dataclass
class EgoMotion:
    """Ego-motion samples on a uniform time grid.

    Arrays are (N,) for timestamps/curvatures and (N, 3) for the rest.
    velocities/accelerations are in the ego frame (x = forward, y = left).

    ``source`` says where the numbers came from:
      * "sensor"  : GPS/IMU or dataset egomotion (metric units)
      * "video"   : estimated from optical flow (relative units, see motion.py)
    """

    timestamps_us: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    curvatures: np.ndarray
    source: str = "sensor"

    def __post_init__(self) -> None:
        self.timestamps_us = np.asarray(self.timestamps_us, dtype=np.int64)
        self.positions = np.asarray(self.positions, dtype=np.float64).reshape(-1, 3)
        self.velocities = np.asarray(self.velocities, dtype=np.float64).reshape(-1, 3)
        self.accelerations = np.asarray(self.accelerations, dtype=np.float64).reshape(-1, 3)
        self.curvatures = np.asarray(self.curvatures, dtype=np.float64).reshape(-1)
        n = len(self.timestamps_us)
        for name in ("positions", "velocities", "accelerations", "curvatures"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")

    @property
    def n(self) -> int:
        return int(len(self.timestamps_us))

    @property
    def time_step_s(self) -> float:
        if self.n < 2:
            return 0.1
        return float(np.median(np.diff(self.timestamps_us))) / 1e6

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocities, axis=1)

    @property
    def ax(self) -> np.ndarray:
        return self.accelerations[:, 0]

    @property
    def duration_s(self) -> float:
        if self.n < 2:
            return 0.0
        return float(self.timestamps_us[-1] - self.timestamps_us[0]) / 1e6

    def slice(self, start_us: int, end_us: int) -> "EgoMotion":
        m = (self.timestamps_us >= start_us) & (self.timestamps_us <= end_us)
        return EgoMotion(
            self.timestamps_us[m],
            self.positions[m],
            self.velocities[m],
            self.accelerations[m],
            self.curvatures[m],
            source=self.source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "timestamps_us": self.timestamps_us.tolist(),
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "accelerations": self.accelerations.tolist(),
            "curvatures": self.curvatures.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EgoMotion":
        return cls(
            d["timestamps_us"],
            d["positions"],
            d["velocities"],
            d["accelerations"],
            d["curvatures"],
            source=d.get("source", "sensor"),
        )


@dataclass
class Clip:
    """A registered input clip (usually one user-recorded video file)."""

    clip_id: str
    path: Optional[str] = None          # video file path (None for sensor-only datasets)
    duration_s: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Clip":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Keyframe:
    timestamp_us: int
    type: str            # e.g. gentle_decel, steer_r, stop, go_straight
    magnitude: float
    category: str        # longitudinal | lateral

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Keyframe":
        return cls(int(d["timestamp_us"]), d["type"], float(d["magnitude"]), d["category"])


@dataclass
class Window:
    """An 8-second decision window around a keyframe."""

    clip_id: str
    index: int
    keyframe: Keyframe
    start_us: int
    end_us: int
    ego: EgoMotion
    frame_timestamps_us: list[int] = field(default_factory=list)   # 16 sample points @2Hz
    history_frames: int = 4                                           # first N frames are history

    @property
    def keyframe_type(self) -> str:
        return self.keyframe.type

    def to_dict(self, include_ego: bool = False) -> dict[str, Any]:
        d = {
            "clip_id": self.clip_id,
            "index": self.index,
            "keyframe": self.keyframe.to_dict(),
            "start_us": int(self.start_us),
            "end_us": int(self.end_us),
            "frame_timestamps_us": [int(t) for t in self.frame_timestamps_us],
            "history_frames": self.history_frames,
        }
        if include_ego:
            d["ego"] = self.ego.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Window":
        return cls(
            clip_id=d["clip_id"],
            index=int(d["index"]),
            keyframe=Keyframe.from_dict(d["keyframe"]),
            start_us=int(d["start_us"]),
            end_us=int(d["end_us"]),
            ego=EgoMotion.from_dict(d["ego"]),
            frame_timestamps_us=[int(t) for t in d.get("frame_timestamps_us", [])],
            history_frames=int(d.get("history_frames", 4)),
        )


@dataclass
class Label:
    """One CoC label produced by a backend (plus review state)."""

    clip_id: str
    index: int
    keyframe_type: str
    first_frame_timestamp_us: int
    keyframe_timestamp_us: int
    coc: str
    backend: str
    raw: Optional[str] = None            # full model output (with THOUGHT_PROCESS etc.)
    latency_s: float = 0.0
    filter: dict[str, Any] = field(default_factory=dict)   # FilterReport.to_dict()
    review: str = "unreviewed"           # unreviewed | approved | rejected | edited
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Label":
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        known.setdefault("meta", {})
        known.setdefault("filter", {})
        return cls(**known)


@dataclass
class Issue:
    code: str
    severity: str      # error | warning | info
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FilterReport:
    verdict: str                  # pass | review | reject
    score: float                  # 0..1 (1 = clean)
    issues: list[Issue] = field(default_factory=list)
    decisions: dict[str, list[str]] = field(default_factory=dict)   # {"longitudinal": [...], "lateral": [...]}
    components: list[str] = field(default_factory=list)
    kinematics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": round(self.score, 4),
            "issues": [i.to_dict() for i in self.issues],
            "decisions": self.decisions,
            "components": self.components,
            "kinematics": {k: round(float(v), 4) for k, v in self.kinematics.items()},
        }

    @property
    def codes(self) -> list[str]:
        return [i.code for i in self.issues]
