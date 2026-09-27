"""Decision windows and frame extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import EgoMotion, Keyframe, Window


def make_windows(
    clip_id: str,
    ego: EgoMotion,
    keyframes: list[Keyframe],
    before_s: float = 2.0,
    after_s: float = 6.0,
    n_frames: int = 16,
    history_frames: int = 4,
) -> list[Window]:
    out = []
    lo, hi = int(ego.timestamps_us[0]), int(ego.timestamps_us[-1])
    for i, kf in enumerate(keyframes):
        start = max(kf.timestamp_us - int(before_s * 1e6), lo)
        end = min(kf.timestamp_us + int(after_s * 1e6), hi)
        if end - start < int(0.5 * (before_s + after_s) * 1e6):
            continue
        frames = np.linspace(start, end, n_frames).astype(np.int64).tolist()
        out.append(
            Window(
                clip_id=clip_id,
                index=i,
                keyframe=kf,
                start_us=int(start),
                end_us=int(end),
                ego=ego.slice(start, end),
                frame_timestamps_us=[int(t) for t in frames],
                history_frames=history_frames,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Frame extraction from video files
# --------------------------------------------------------------------------- #
class FrameReader:
    """Random-access frame reader with a tiny cache (OpenCV)."""

    def __init__(self, video_path: str, max_side: int = 768):
        import cv2

        self.cv2 = cv2
        self.path = str(video_path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.max_side = max_side

    def close(self) -> None:
        self.cap.release()

    def read_at(self, timestamp_us: int) -> np.ndarray:
        idx = int(round(timestamp_us / 1e6 * self.fps))
        idx = min(max(idx, 0), max(self.n - 1, 0))
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {idx} of {self.path}")
        h, w = frame.shape[:2]
        s = self.max_side / float(max(h, w))
        if s < 1.0:
            frame = self.cv2.resize(frame, (int(w * s), int(h * s)), interpolation=self.cv2.INTER_AREA)
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)

    def read_many(self, timestamps_us: list[int]) -> list[np.ndarray]:
        return [self.read_at(t) for t in timestamps_us]


def contact_sheet(frames: list[np.ndarray], cols: int = 4, thumb_w: int = 320, history_frames: int = 4) -> np.ndarray:
    """Tile frames into a grid; history frames get a dark (navy) border, future frames a light grey one."""
    import cv2

    if not frames:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    thumbs = []
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        th = int(round(h * thumb_w / float(w)))
        t = cv2.resize(f, (thumb_w, th), interpolation=cv2.INTER_AREA)
        color = (47, 79, 127) if i < history_frames else (170, 175, 182)
        t = cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=color)
        cv2.putText(t, f"{i + 1}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(t)
    th = max(t.shape[0] for t in thumbs)
    tw = max(t.shape[1] for t in thumbs)
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th : r * th + t.shape[0], c * tw : c * tw + t.shape[1]] = t
    return sheet


def save_image(path: str | Path, rgb: np.ndarray, quality: int = 85) -> None:
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def extract_window_frames(
    video_path: str,
    window: Window,
    out_dir: Optional[str | Path] = None,
    max_side: int = 768,
    reader: Optional[FrameReader] = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Decode the window's 16 frames; optionally save them + a contact sheet."""
    own = reader is None
    reader = reader or FrameReader(video_path, max_side=max_side)
    try:
        frames = reader.read_many(window.frame_timestamps_us)
    finally:
        if own:
            reader.close()
    paths: list[str] = []
    if out_dir is not None:
        d = Path(out_dir) / window.clip_id / f"kf{window.index:03d}"
        for i, f in enumerate(frames):
            p = d / f"frame_{i:02d}.jpg"
            save_image(p, f)
            paths.append(str(p))
        save_image(d / "sheet.jpg", contact_sheet(frames, history_frames=window.history_frames))
    return frames, paths
