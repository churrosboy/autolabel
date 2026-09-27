"""Ego-motion sources.

Three ways to get an :class:`EgoMotion` for a clip:

* :func:`estimate_from_video`  - optical-flow based estimate from a plain
  dashcam video (no sensors needed).  Units are *relative*: speed is a
  scene-expansion proxy, curvature is a normalised yaw proxy.  The keyframe
  detector uses adaptive thresholds for ``source == "video"``.
* :func:`load_sensor_csv`      - GPS/IMU log (phone apps, CAN loggers).
* :func:`from_physical_ai_window` - windows exported by the D3D pipeline.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import EgoMotion


# --------------------------------------------------------------------------- #
# Smoothing helpers
# --------------------------------------------------------------------------- #
def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(window) / window
    pad = window // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), (pad, window - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD (falls back to std, then to 1)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    if mad > 1e-9:
        return float(mad)
    s = float(np.std(x))
    return s if s > 1e-9 else 1.0


# --------------------------------------------------------------------------- #
# 1) Video -> ego-motion proxy via dense optical flow
# --------------------------------------------------------------------------- #
@dataclass
class VideoMotionConfig:
    sample_fps: float = 10.0        # analysis rate (Hz)
    width: int = 320                # analysis resolution
    smooth_window: int = 5          # samples (0.5 s at 10 Hz)
    roi_top: float = 0.35           # ignore sky / hood: analyse rows in [roi_top, roi_bottom]
    roi_bottom: float = 0.95
    yaw_band: tuple[float, float] = (0.35, 0.65)   # rows used for yaw estimate (far field)
    max_frames: Optional[int] = None


def _open_video(path: str):
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for video motion estimation") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    return cv2, cap


def probe_video(path: str) -> dict:
    cv2, cap = _open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": float(fps), "frames": n, "width": w, "height": h, "duration_s": (n / fps) if fps else 0.0}


def estimate_from_video(path: str, cfg: VideoMotionConfig | None = None) -> EgoMotion:
    """Estimate forward-speed / yaw proxies from dense optical flow.

    * speed proxy  : mean radial expansion (divergence) of the flow field in the
      lower ROI. Positive when the camera moves forward.
    * yaw proxy    : median horizontal flow in the far-field band.  The scene
      flows left (negative x) when turning right, so the raw sign already
      matches the curvature convention used elsewhere (left turn positive).
    * curvature    : yaw proxy normalised by its robust scale.  The radial
      (speed) estimate is computed after subtracting the global median flow so
      that panning does not leak into the expansion term.
    """
    cfg = cfg or VideoMotionConfig()
    cv2, cap = _open_video(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / cfg.sample_fps)))
    eff_fps = src_fps / step

    prev_gray = None
    times, spd, yaw = [], [], []
    frame_idx = 0
    grid_cache = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            h0, w0 = frame.shape[:2]
            scale = cfg.width / float(w0)
            small = cv2.resize(frame, (cfg.width, int(round(h0 * scale))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                H, W = gray.shape
                if grid_cache is None or grid_cache[0].shape != (H, W):
                    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                    cx, cy = W / 2.0, H * 0.5
                    dx, dy = xx - cx, yy - cy
                    norm = np.sqrt(dx * dx + dy * dy) + 1e-6
                    grid_cache = (dx / norm, dy / norm)
                ux, uy = grid_cache
                b0, b1 = int(H * cfg.yaw_band[0]), int(H * cfg.yaw_band[1])
                yaw_proxy = float(np.median(flow[b0:b1, :, 0]))       # left turn -> positive
                # remove global translation (pan/tilt) before measuring expansion
                fx = flow[..., 0] - float(np.median(flow[..., 0]))
                fy = flow[..., 1] - float(np.median(flow[..., 1]))
                r0, r1 = int(H * cfg.roi_top), int(H * cfg.roi_bottom)
                radial = (fx * ux + fy * uy)[r0:r1]
                speed_proxy = float(np.mean(radial))
                times.append(frame_idx / src_fps)
                spd.append(speed_proxy)
                yaw.append(yaw_proxy)
            prev_gray = gray
        frame_idx += 1
        if cfg.max_frames and frame_idx >= cfg.max_frames:
            break
    cap.release()

    if len(times) < 3:
        raise ValueError(f"video too short for motion estimation: {path}")

    t = np.asarray(times)
    spd = moving_average(np.asarray(spd), cfg.smooth_window)
    yaw = moving_average(np.asarray(yaw), cfg.smooth_window)
    spd = np.clip(spd, 0.0, None)                       # camera does not reverse (assumption)
    dt = 1.0 / eff_fps
    acc = np.gradient(spd, dt)
    acc = moving_average(acc, cfg.smooth_window)

    # normalise to unit robust scale so thresholds can be shared across videos
    s_scale = max(robust_scale(spd), 1e-6)
    spd_n = spd / s_scale
    acc_n = acc / s_scale
    y_scale = max(robust_scale(yaw), 1e-6)
    curv_n = yaw / y_scale

    # dead-reckoned path in relative units (for plots / turn-angle checks)
    heading = np.cumsum(curv_n * dt) * 0.35   # ~rad; scale chosen so |curv_n|=1 for 1s ≈ 20°
    x = np.cumsum(spd_n * np.cos(heading) * dt)
    y = np.cumsum(spd_n * np.sin(heading) * dt)
    positions = np.stack([x, y, np.zeros_like(x)], axis=1)
    velocities = np.stack([spd_n, np.zeros_like(spd_n), np.zeros_like(spd_n)], axis=1)
    accelerations = np.stack([acc_n, np.zeros_like(acc_n), np.zeros_like(acc_n)], axis=1)

    return EgoMotion(
        timestamps_us=(t * 1e6).astype(np.int64),
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        curvatures=curv_n,
        source="video",
    )


# --------------------------------------------------------------------------- #
# 2) Sensor CSV (GPS / IMU)
# --------------------------------------------------------------------------- #
_TIME_KEYS = ("timestamp_us", "time_us", "t_us", "timestamp", "time", "time_s", "t")
_SPEED_KEYS = ("speed", "speed_mps", "v", "velocity")
_AX_KEYS = ("ax", "accel_x", "acceleration_x", "a_x")
_YAW_KEYS = ("yaw_rate", "yawrate", "gyro_z", "gz", "omega")
_CURV_KEYS = ("curvature", "curv", "kappa")
_LAT_KEYS = ("lat", "latitude")
_LON_KEYS = ("lon", "lng", "longitude")


def _pick(row_keys: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    lower = {k.lower().strip(): k for k in row_keys}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def load_sensor_csv(path: str, time_step_s: float = 0.1) -> EgoMotion:
    """Load a GPS/IMU CSV and resample to a uniform grid.

    Recognised columns (case-insensitive): time (s or us), speed (m/s),
    ax (m/s^2), yaw_rate (rad/s) or curvature (1/m), optional lat/lon.
    Missing ax / curvature are derived.
    """
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError(f"empty csv: {path}")
    keys = list(rows[0].keys())
    tk = _pick(keys, _TIME_KEYS)
    if tk is None:
        raise ValueError(f"no time column in {path}; columns={keys}")
    t = np.asarray([float(r[tk]) for r in rows])
    if tk.lower() in ("timestamp_us", "time_us", "t_us") or t.max() > 1e7:
        t_s = t / 1e6
    else:
        t_s = t
    t_s = t_s - t_s[0]

    def col(cands):
        k = _pick(keys, cands)
        return None if k is None else np.asarray([float(r[k] or "nan") for r in rows])

    speed = col(_SPEED_KEYS)
    latc, lonc = col(_LAT_KEYS), col(_LON_KEYS)
    if speed is None and latc is not None and lonc is not None:
        # haversine distance between consecutive fixes
        R = 6371000.0
        phi = np.radians(latc)
        dphi = np.diff(phi)
        dl = np.radians(np.diff(lonc))
        a = np.sin(dphi / 2) ** 2 + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(dl / 2) ** 2
        d = 2 * R * np.arcsin(np.sqrt(a))
        dt = np.diff(t_s)
        speed = np.concatenate([[0.0], d / np.maximum(dt, 1e-3)])
    if speed is None:
        raise ValueError("csv needs a speed column or lat/lon")

    ax = col(_AX_KEYS)
    curv = col(_CURV_KEYS)
    yaw = col(_YAW_KEYS)

    grid = np.arange(0.0, t_s[-1] + 1e-9, time_step_s)
    sp = np.interp(grid, t_s, np.nan_to_num(speed))
    sp = moving_average(sp, 3)
    if ax is None:
        axg = np.gradient(sp, time_step_s)
    else:
        axg = np.interp(grid, t_s, np.nan_to_num(ax))
    if curv is None:
        if yaw is not None:
            yg = np.interp(grid, t_s, np.nan_to_num(yaw))
            curvg = yg / np.maximum(sp, 0.5)
        else:
            curvg = np.zeros_like(sp)
    else:
        curvg = np.interp(grid, t_s, np.nan_to_num(curv))

    heading = np.cumsum(curvg * sp * time_step_s)
    x = np.cumsum(sp * np.cos(heading) * time_step_s)
    y = np.cumsum(sp * np.sin(heading) * time_step_s)
    return EgoMotion(
        timestamps_us=(grid * 1e6).astype(np.int64),
        positions=np.stack([x, y, np.zeros_like(x)], axis=1),
        velocities=np.stack([sp, np.zeros_like(sp), np.zeros_like(sp)], axis=1),
        accelerations=np.stack([axg, np.zeros_like(axg), np.zeros_like(axg)], axis=1),
        curvatures=curvg,
        source="sensor",
    )


# --------------------------------------------------------------------------- #
# 3) D3D / physical_ai_av exported windows
# --------------------------------------------------------------------------- #
def from_physical_ai_window(d: dict) -> EgoMotion:
    return EgoMotion(
        timestamps_us=d["timestamps_us"],
        positions=d["positions"],
        velocities=d["velocities"],
        accelerations=d["accelerations"],
        curvatures=d["curvatures"],
        source="sensor",
    )


# --------------------------------------------------------------------------- #
# Kinematic summaries (shared by rule-based labeler and filter)
# --------------------------------------------------------------------------- #
def heading_change_deg(ego: EgoMotion, start_us: int, end_us: int, min_travel_m: float = 1.0) -> float:
    """Signed heading change (deg, left positive) between two timestamps.

    Headings are taken from the first and last ``min_travel_m`` of travelled
    path (not from single noisy steps), so stop-and-go segments do not produce
    spurious turns.  Falls back to integrated curvature for very short paths.
    """
    seg = ego.slice(start_us, end_us)
    if seg.n < 4:
        return 0.0
    p = seg.positions[:, :2]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    total = float(cum[-1])
    if ego.source == "video":
        min_travel_m = max(0.05 * total, 1e-3)
    if total < 2.0 * min_travel_m:
        ds = seg.speed[:-1] * seg.time_step_s
        return float(np.degrees(np.sum(seg.curvatures[:-1] * ds)))
    i0 = int(np.searchsorted(cum, min_travel_m))
    i1 = int(np.searchsorted(cum, total - min_travel_m))
    i0 = min(max(i0, 1), seg.n - 1)
    i1 = min(max(i1, 0), seg.n - 2)
    v0 = p[i0] - p[0]
    v1 = p[-1] - p[i1]
    if np.linalg.norm(v0) < 1e-6 or np.linalg.norm(v1) < 1e-6:
        return 0.0
    h0 = math.atan2(v0[1], v0[0])
    h1 = math.atan2(v1[1], v1[0])
    dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
    return float(math.degrees(dh))


def lateral_offset_m(ego: EgoMotion, start_us: int, end_us: int, min_travel_m: float = 1.0) -> float:
    """Lateral displacement (m, left positive) of the end point w.r.t. the initial heading."""
    seg = ego.slice(start_us, end_us)
    if seg.n < 4:
        return 0.0
    p = seg.positions[:, :2]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    if cum[-1] < 2.0 * min_travel_m:
        return 0.0
    i0 = min(max(int(np.searchsorted(cum, min_travel_m)), 1), seg.n - 1)
    h = p[i0] - p[0]
    if np.linalg.norm(h) < 1e-6:
        return 0.0
    h = h / np.linalg.norm(h)
    left = np.array([-h[1], h[0]])
    return float(np.dot(p[-1] - p[0], left))


def summarize_window(ego: EgoMotion, kf_us: int, before_s: float = 2.0, after_s: float = 6.0) -> dict[str, float]:
    """Numbers the filter and the rule-based labeler reason about."""
    hist = ego.slice(kf_us - int(before_s * 1e6), kf_us)
    fut = ego.slice(kf_us, kf_us + int(after_s * 1e6))
    if hist.n == 0 or fut.n == 0:
        return {}
    sp_h, sp_f = hist.speed, fut.speed
    return {
        "speed_before": float(np.mean(sp_h[-5:])),
        "speed_after_2s": float(np.mean(sp_f[min(fut.n - 1, 15):min(fut.n, 25)])) if fut.n > 15 else float(sp_f[-1]),
        "speed_end": float(np.mean(sp_f[-5:])),
        "speed_min_future": float(np.min(sp_f)),
        "speed_max_future": float(np.max(sp_f)),
        "ax_mean_future_2s": float(np.mean(fut.ax[:min(fut.n, 20)])),
        "ax_min_future": float(np.min(fut.ax)),
        "ax_max_future": float(np.max(fut.ax)),
        "curv_abs_max_future": float(np.max(np.abs(fut.curvatures))),
        "heading_change_deg": heading_change_deg(ego, kf_us, kf_us + int(after_s * 1e6)),
        "lateral_offset_m": lateral_offset_m(ego, kf_us, kf_us + int(after_s * 1e6)),
        "stopped_frac_future": float(np.mean(sp_f < 0.3)),
        "speed_drop": float(np.mean(sp_h[-5:]) - np.min(sp_f)),
        "dist_future_m": float(np.sum(sp_f[:-1] * fut.time_step_s)) if fut.n > 1 else 0.0,
    }
