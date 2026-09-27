import json
from pathlib import Path

import numpy as np
import pytest

from autolabel.pipeline import Project
from autolabel.sources import load_d3d

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def make_synthetic_video(path: Path, seconds: float = 12.0, fps: int = 20, size=(240, 160)) -> None:
    """Camera driving over a textured ground plane (perspective projection).

    Phases: forward (0-5 s), stopped (5-8 s), forward + turning right (8 s-).
    Forward motion produces radial expansion; turning produces horizontal flow.
    """
    import cv2

    rng = np.random.default_rng(0)
    tile = 256
    tex = rng.integers(0, 255, (tile, tile, 3), dtype=np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.5)
    w, h = size
    f, cam_h = 0.9 * w, 1.4                       # focal length (px), camera height (m)
    cx, cy = w / 2, h * 0.45                      # horizon at 45% height
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ground = v > cy + 2
    Z = np.where(ground, f * cam_h / np.maximum(v - cy, 1e-3), 1e6)   # depth (m)
    X = (u - cx) * Z / f                                              # lateral (m)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = int(seconds * fps)
    dist, heading = 0.0, 0.0
    px_per_m = 40.0
    for i in range(n):
        t = i / fps
        speed = 6.0 if t < 5.0 else (0.0 if t < 8.0 else 6.0)        # m/s
        yaw_rate = 0.0 if t < 8.0 else 0.35                          # rad/s (right turn)
        dist += speed / fps
        heading += yaw_rate / fps
        # world coords of each ground pixel (rotate by heading, translate by dist)
        c, s_ = np.cos(heading), np.sin(heading)
        Xw = c * X - s_ * Z
        Zw = s_ * X + c * Z + dist
        mx = ((Xw * px_per_m) % tile).astype(np.float32)
        my = ((Zw * px_per_m) % tile).astype(np.float32)
        frame = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        # sky: distant texture strip that pans with heading
        sky = np.full((h, w, 3), (200, 170, 120), dtype=np.uint8)
        shift = int(heading * f) % tile
        strip = np.roll(tex[:40], -shift, axis=1)
        strip = cv2.resize(strip, (w, int(cy)))
        sky[: int(cy)] = strip
        frame = np.where(ground[..., None], frame, sky)
        vw.write(np.ascontiguousarray(frame))
    vw.release()


def test_video_project_end_to_end(tmp_path):
    video = tmp_path / "synthetic.mp4"
    make_synthetic_video(video)
    p = Project.create(tmp_path / "proj", fallback_uniform_s=8.0)
    clip = p.add_video(video)
    assert clip.duration_s > 10
    motions = p.run_motion()
    ego = motions[clip.clip_id]
    assert ego.source == "video" and ego.n > 50
    # forward phase should have higher speed proxy than the static phase
    t = ego.timestamps_us / 1e6
    assert ego.speed[(t > 1) & (t < 4.5)].mean() > ego.speed[(t > 5.5) & (t < 7.5)].mean() + 0.1
    wins = p.run_keyframes(extract_frames=True)
    assert wins, "expected at least one window (detected or uniform fallback)"
    sheet = p.frames_dir / wins[0].clip_id / "kf000" / "sheet.jpg"
    assert sheet.exists()
    labels = p.run_label("rule_based")
    assert len(labels) == len(wins)
    assert all(l.filter.get("verdict") for l in labels)
    summary = p.export("ds")
    assert summary["kept"] >= 1
    rows = [json.loads(l) for l in open(Path(summary["export_dir"]) / "samples.jsonl")]
    assert rows[0]["frames"] and len(rows[0]["frame_timestamps_us"]) == 16


@pytest.mark.skipif(not (SAMPLES / "ego_motion_results.jsonl").exists(), reason="sample data missing")
def test_import_d3d_and_evaluate(tmp_path):
    p = Project.create(tmp_path / "d3d")
    wins, gt = load_d3d(SAMPLES / "ego_motion_results.jsonl", SAMPLES / "coc_results_pro.jsonl", limit=5)
    assert wins and gt
    p.add_windows(wins)
    labels = p.run_label("rule_based")
    assert len(labels) == len(wins)
    rep = p.run_filter()
    assert rep["n"] == len(wins)
    # review round-trip
    l0 = labels[0]
    p.set_review(l0.clip_id, l0.index, "edited", "The vehicle performs yield because a pedestrian is crossing.")
    assert any(l.review == "edited" for l in p.labels())
    p.set_review(l0.clip_id, l0.index, "rejected")
    s = p.export("ds", reviewed_only=False)
    assert s["dropped"] >= 1
