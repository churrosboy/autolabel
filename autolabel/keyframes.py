"""Keyframe (high-level driving decision) detection from ego-motion.

The detector looks for *state transitions* of the smoothed longitudinal
acceleration and of the smoothed |curvature|, then applies three filters:

1. cooldown        - same action type within ``cooldown_s`` is dropped,
2. per-window cap  - at most one longitudinal + one lateral decision per
                     ``window_s`` window (Alpamayo-R1 data recipe),
3. margin          - keyframes whose [-before, +after] window would fall
                     outside the clip are dropped.

Sensor data uses metric thresholds; video-estimated motion (``source ==
"video"``) uses thresholds in robust-scale units (see motion.py).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .motion import moving_average
from .schemas import EgoMotion, Keyframe
from .vocab import KEYFRAME_CATEGORY


@dataclass
class KeyframeConfig:
    # longitudinal thresholds (m/s^2 for sensor, robust-scale units for video)
    strong_accel: float = 3.0
    gentle_accel: float = 1.5
    gentle_decel: float = -1.0
    strong_decel: float = -7.0
    stop_speed: float = 0.1            # m/s
    stop_min_s: float = 0.5
    # lateral thresholds (1/m for sensor, robust units for video)
    steer: float = 0.02
    sharp_steer: float = 0.15
    # smoothing / timing
    smooth_window: int = 2             # samples
    lead_s: float = 0.5                # place keyframe this long before the transition
    cooldown_s: float = 3.0
    window_s: float = 8.0
    per_window_cap: bool = True
    confirm_with_speed: bool = True    # drop accel/decel events the speed trace does not confirm
    confirm_horizon_s: float = 2.0
    confirm_delta: float = 0.5         # m/s (sensor) or robust units (video)
    before_s: float = 2.0
    after_s: float = 6.0
    start_offset_s: float = 1.0

    @classmethod
    def for_source(cls, source: str) -> "KeyframeConfig":
        if source == "video":
            # signals are normalised by their robust scale (MAD): thresholds in "sigmas"
            return cls(
                strong_accel=2.5, gentle_accel=1.0, gentle_decel=-1.0, strong_decel=-2.5,
                stop_speed=0.15, stop_min_s=0.8,
                steer=1.0, sharp_steer=2.5,
                smooth_window=5, lead_s=0.5, cooldown_s=3.0, confirm_delta=0.3,
            )
        return cls()


def _speed_confirms(speed: np.ndarray, i: int, sign: int, cfg: KeyframeConfig, dt: float) -> bool:
    """True if the speed trace moves in ``sign`` direction within the horizon after index i."""
    if not cfg.confirm_with_speed:
        return True
    h = int(round(cfg.confirm_horizon_s / dt))
    seg = speed[i : i + h + 1]
    if len(seg) < 3:
        return True
    ref = float(np.mean(speed[max(0, i - 2) : i + 1]))
    if sign < 0:
        return float(np.min(seg)) < ref - cfg.confirm_delta
    return float(np.max(seg)) > ref + cfg.confirm_delta


def _detect_longitudinal(ts: np.ndarray, ax: np.ndarray, speed: np.ndarray, cfg: KeyframeConfig, dt: float):
    out = []
    lead = int(round(cfg.lead_s / dt))
    a = moving_average(ax, cfg.smooth_window)
    for i in range(1, len(a)):
        p, c = a[i - 1], a[i]
        j = max(0, i - lead)
        if c > cfg.strong_accel >= p and _speed_confirms(speed, i, +1, cfg, dt):
            out.append((int(ts[j]), "strong_accel", float(c)))
        elif cfg.gentle_accel < c <= cfg.strong_accel and p <= cfg.gentle_accel and _speed_confirms(speed, i, +1, cfg, dt):
            out.append((int(ts[j]), "gentle_accel", float(c)))
        elif cfg.strong_decel <= c < cfg.gentle_decel and p >= cfg.gentle_decel and _speed_confirms(speed, i, -1, cfg, dt):
            out.append((int(ts[j]), "gentle_decel", float(abs(c))))
        elif c < cfg.strong_decel <= p and _speed_confirms(speed, i, -1, cfg, dt):
            out.append((int(ts[j]), "strong_decel", float(abs(c))))
    n_stop = max(1, int(round(cfg.stop_min_s / dt)))
    for i in range(0, len(speed) - n_stop + 1):
        if np.all(speed[i:i + n_stop] < cfg.stop_speed) and (i == 0 or speed[i - 1] >= cfg.stop_speed):
            j = max(0, i - lead)
            out.append((int(ts[j]), "stop", 0.0))
    return out


def _detect_lateral(ts: np.ndarray, curv: np.ndarray, cfg: KeyframeConfig, dt: float):
    out = []
    lead = int(round(cfg.lead_s / dt))
    ca = moving_average(np.abs(curv), cfg.smooth_window)
    for i in range(1, len(ca)):
        p, c = ca[i - 1], ca[i]
        j = max(0, i - lead)
        side = "l" if curv[i] > 0 else "r"
        if c > cfg.sharp_steer >= p:
            out.append((int(ts[j]), f"sharp_steer_{side}", float(c)))
        elif cfg.steer < c <= cfg.sharp_steer and p <= cfg.steer:
            out.append((int(ts[j]), f"steer_{side}", float(c)))
        elif c <= cfg.steer < p:
            out.append((int(ts[j]), "go_straight", 0.0))
    return out


def filter_cooldown(kfs: list[tuple[int, str, float]], cooldown_us: int):
    last: dict[str, int] = {}
    out = []
    for ts, typ, mag in sorted(kfs, key=lambda k: k[0]):
        if typ in last and ts - last[typ] < cooldown_us:
            continue
        out.append((ts, typ, mag))
        last[typ] = ts
    return out


def filter_per_window(kfs: list[tuple[int, str, float]], window_us: int):
    """Keep the strongest longitudinal and lateral event per fixed window."""
    buckets: dict[int, dict[str, list]] = {}
    for k in kfs:
        b = buckets.setdefault(k[0] // window_us, {"longitudinal": [], "lateral": []})
        b[KEYFRAME_CATEGORY.get(k[1], "longitudinal")].append(k)
    out = []
    for b in buckets.values():
        for cat in ("longitudinal", "lateral"):
            if b[cat]:
                out.append(max(b[cat], key=lambda k: k[2]))
    return sorted(out, key=lambda k: k[0])


def detect_keyframes(ego: EgoMotion, cfg: KeyframeConfig | None = None) -> list[Keyframe]:
    cfg = cfg or KeyframeConfig.for_source(ego.source)
    if ego.n < 5:
        return []
    dt = ego.time_step_s
    t0 = int(ego.timestamps_us[0] + cfg.start_offset_s * 1e6)
    m = ego.timestamps_us >= t0
    ts = ego.timestamps_us[m]
    if len(ts) < 5:
        return []
    ax, speed, curv = ego.ax[m], ego.speed[m], ego.curvatures[m]

    kfs = _detect_longitudinal(ts, ax, speed, cfg, dt) + _detect_lateral(ts, curv, cfg, dt)
    kfs = filter_cooldown(kfs, int(cfg.cooldown_s * 1e6))
    if cfg.per_window_cap:
        kfs = filter_per_window(kfs, int(cfg.window_s * 1e6))
    lo = int(ego.timestamps_us[0] + cfg.before_s * 1e6)
    hi = int(ego.timestamps_us[-1] - cfg.after_s * 1e6)
    kfs = [k for k in kfs if lo <= k[0] <= hi]
    return [Keyframe(ts, typ, mag, KEYFRAME_CATEGORY.get(typ, "longitudinal")) for ts, typ, mag in kfs]


def uniform_keyframes(ego: EgoMotion, every_s: float = 8.0, cfg: KeyframeConfig | None = None) -> list[Keyframe]:
    """Fallback for clips with no detectable events: one window every ``every_s``."""
    cfg = cfg or KeyframeConfig.for_source(ego.source)
    lo = int(ego.timestamps_us[0] + cfg.before_s * 1e6)
    hi = int(ego.timestamps_us[-1] - cfg.after_s * 1e6)
    out = []
    t = lo
    while t <= hi:
        out.append(Keyframe(int(t), "go_straight", 0.0, "lateral"))
        t += int(every_s * 1e6)
    return out


def relabel_lateral_side(kf: Keyframe, ego: EgoMotion, cfg: KeyframeConfig) -> Keyframe:
    """Fix the l/r suffix using the sign of curvature after the keyframe (sensor data only)."""
    if not kf.type.startswith(("steer", "sharp_steer")):
        return kf
    seg = ego.slice(kf.timestamp_us, kf.timestamp_us + int(2e6))
    if seg.n == 0:
        return kf
    side = "l" if float(np.mean(seg.curvatures)) > 0 else "r"
    base = "sharp_steer" if kf.type.startswith("sharp") else "steer"
    return replace(kf, type=f"{base}_{side}")
