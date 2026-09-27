"""Training samples built from an Autolabel project or export (no torch needed)."""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..prompt import build_prompt
from ..schemas import EgoMotion, Keyframe, Window
from ..sources import read_jsonl
from ..vocab import KEYFRAME_CATEGORY, LATERAL, LONGITUDINAL, extract_decisions

LON_CLASSES = LONGITUDINAL + ["none"]
LAT_CLASSES = LATERAL + ["none"]
NONE_LON = len(LON_CLASSES) - 1
NONE_LAT = len(LAT_CLASSES) - 1


@dataclass
class TrainSample:
    clip_id: str
    index: int
    keyframe_type: str
    frames: list[str]                 # image paths, history first
    prompt: str
    target: str                       # assistant text: "FINAL_COC: ..."
    coc: str
    lon_label: int
    lat_label: int
    history_frames: int = 4
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def head_labels(coc: str) -> tuple[int, int]:
    d = extract_decisions(coc)
    lon = LON_CLASSES.index(d["longitudinal"][0]) if d["longitudinal"] else NONE_LON
    lat = LAT_CLASSES.index(d["lateral"][0]) if d["lateral"] else NONE_LAT
    return lon, lat


def target_text(coc: str) -> str:
    return f"FINAL_COC: {coc.strip()}"


def frames_for(project_root: Path, clip_id: str, index: int) -> list[str]:
    d = project_root / "frames" / clip_id / f"kf{index:03d}"
    return sorted(str(p) for p in d.glob("frame_*.jpg")) if d.exists() else []


def make_sample(w: Window, coc: str, frames: list[str], include_trajectory: bool = True) -> TrainSample:
    lon, lat = head_labels(coc)
    prompt = build_prompt(w, include_trajectory=include_trajectory, n_frames=len(frames), history_frames=w.history_frames, concise=True)
    return TrainSample(w.clip_id, w.index, w.keyframe_type, frames, prompt, target_text(coc), coc, lon, lat, w.history_frames)


def load_from_project(project_dir: str | Path, include_verdicts: Iterable[str] = ("pass", "review"), reviewed_only: bool = False,
                      labels_file: str = "labels.jsonl", gt_file: str | None = "gt.json") -> list[TrainSample]:
    """Read windows.jsonl + labels (or gt.json) + frames/ from a project directory.

    If ``gt.json`` exists (imported D3D dataset) its sentences are used as
    targets; otherwise labels.jsonl rows that are not rejected are used.
    """
    root = Path(project_dir)
    wins = {(w["clip_id"], int(w["index"])): Window.from_dict(w) for w in read_jsonl(root / "windows.jsonl")}
    targets: dict[tuple[str, int], str] = {}
    gt_path = root / gt_file if gt_file else None
    if gt_path is not None and gt_path.exists():
        for k, v in json.loads(gt_path.read_text()).items():
            cid, idx = k.rsplit("#", 1)
            targets[(cid, int(idx))] = v
    elif (root / labels_file).exists():
        for l in read_jsonl(root / labels_file):
            if l.get("review") == "rejected":
                continue
            verdict = (l.get("filter") or {}).get("verdict", "pass")
            if l.get("review") not in ("approved", "edited") and (reviewed_only or verdict not in include_verdicts):
                continue
            targets[(l["clip_id"], int(l["index"]))] = l["coc"]
    out = []
    for key, coc in targets.items():
        w = wins.get(key)
        if w is None:
            continue
        frames = frames_for(root, *key)
        if not frames:
            continue
        out.append(make_sample(w, coc, frames))
    return out


def _window_from_sample_row(r: dict) -> Window:
    kf = Keyframe(int(r["keyframe_timestamp_us"]), r["keyframe_type"], 0.0, KEYFRAME_CATEGORY.get(r["keyframe_type"], "longitudinal"))
    ts = [int(t) for t in (r.get("frame_timestamps_us") or [])]
    ego = EgoMotion([], [], [], [], [])
    return Window(r["clip_id"], int(r["index"]), kf, ts[0] if ts else kf.timestamp_us, ts[-1] if ts else kf.timestamp_us, ego, ts, int(r.get("history_frames", 4)))


def load_from_export(export_dir: str | Path) -> list[TrainSample]:
    root = Path(export_dir)
    out = []
    for r in read_jsonl(root / "samples.jsonl"):
        frames = [f if Path(f).is_absolute() else str((root / f).resolve()) for f in r.get("frames", [])]
        if not frames:
            continue
        out.append(make_sample(_window_from_sample_row(r), r["coc"], frames, include_trajectory=False))
    return out


def load_samples(path: str | Path, **kw) -> list[TrainSample]:
    p = Path(path)
    if (p / "windows.jsonl").exists():
        return load_from_project(p, **kw)
    if (p / "samples.jsonl").exists():
        return load_from_export(p)
    raise FileNotFoundError(f"{p} is neither a project (windows.jsonl) nor an export (samples.jsonl)")


def split_by_clip(samples: list[TrainSample], val_frac: float = 0.2, seed: int = 0) -> tuple[list[TrainSample], list[TrainSample]]:
    """Clip-level split so windows of one clip never straddle train/val."""
    clips = sorted({s.clip_id for s in samples})
    rnd = random.Random(seed)
    rnd.shuffle(clips)
    n_val = max(1, int(round(len(clips) * val_frac))) if len(clips) > 1 and val_frac > 0 else 0
    val_clips = set(clips[:n_val])
    return [s for s in samples if s.clip_id not in val_clips], [s for s in samples if s.clip_id in val_clips]


def class_weights(samples: list[TrainSample]) -> tuple[list[float], list[float]]:
    """Inverse-frequency (sqrt-damped) weights for the two head losses."""
    def w(labels: list[int], n: int) -> list[float]:
        counts = [labels.count(i) for i in range(n)]
        tot = max(sum(counts), 1)
        return [math.sqrt(tot / (n * c)) if c > 0 else 1.0 for c in counts]

    return w([s.lon_label for s in samples], len(LON_CLASSES)), w([s.lat_label for s in samples], len(LAT_CLASSES))


def summarize(samples: list[TrainSample]) -> dict:
    return {
        "n": len(samples),
        "clips": len({s.clip_id for s in samples}),
        "keyframe_types": dict(Counter(s.keyframe_type for s in samples)),
        "lon_labels": {LON_CLASSES[i]: c for i, c in sorted(Counter(s.lon_label for s in samples).items())},
        "lat_labels": {LAT_CLASSES[i]: c for i, c in sorted(Counter(s.lat_label for s in samples).items())},
        "frames_per_sample": dict(Counter(len(s.frames) for s in samples)),
    }


def dump_samples(samples: list[TrainSample], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
