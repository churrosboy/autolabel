"""Import existing datasets (D3D / physical_ai_av exports) as windows + GT labels.

Files produced by the D3D ``coc_gemini_modified.py`` pipeline:

* ``ego_motion_results.jsonl``  {clip_id, motions:[{index, ego_motion:{...}}]}
* ``coc_results_pro.jsonl``     {clip_id, coc_results:[{index, type, first_frame_timestamp_us, coc}]}
* ``keyframes_partial.jsonl``   {clip_id, keyframe_timestamps_us, keyframe_types, keyframe_magnitudes}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .motion import from_physical_ai_window
from .schemas import EgoMotion, Keyframe, Window
from .vocab import KEYFRAME_CATEGORY


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Iterate a .jsonl or .jsonl.gz file."""
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_d3d(
    ego_jsonl: str | Path,
    coc_jsonl: Optional[str | Path] = None,
    keyframes_jsonl: Optional[str | Path] = None,
    before_s: float = 2.0,
    n_frames: int = 16,
    limit: Optional[int] = None,
) -> tuple[list[Window], dict[tuple[str, int], str]]:
    """Return (windows, gt) where gt maps (clip_id, index) -> reference CoC."""
    kf_types: dict[tuple[str, int], tuple[str, int]] = {}
    coc_rows: dict[str, list[dict]] = {}
    if coc_jsonl and Path(coc_jsonl).exists():
        for row in read_jsonl(coc_jsonl):
            coc_rows[row["clip_id"]] = row["coc_results"]
            for r in row["coc_results"]:
                kf_types[(row["clip_id"], r["index"])] = (r["type"], int(r["first_frame_timestamp_us"]))
    windows: list[Window] = []
    gt: dict[tuple[str, int], str] = {}
    for n_clip, row in enumerate(read_jsonl(ego_jsonl)):
        if limit and n_clip >= limit:
            break
        clip_id = row["clip_id"]
        for m in row["motions"]:
            idx = int(m["index"])
            ego: EgoMotion = from_physical_ai_window(m["ego_motion"])
            typ, first_ts = kf_types.get((clip_id, idx), ("go_straight", int(ego.timestamps_us[0])))
            kf_us = int(ego.timestamps_us[0] + before_s * 1e6)
            kf = Keyframe(kf_us, typ, 0.0, KEYFRAME_CATEGORY.get(typ, "longitudinal"))
            import numpy as np

            frames = np.linspace(ego.timestamps_us[0], ego.timestamps_us[-1], n_frames).astype(np.int64).tolist()
            windows.append(
                Window(
                    clip_id=clip_id,
                    index=idx,
                    keyframe=kf,
                    start_us=int(ego.timestamps_us[0]),
                    end_us=int(ego.timestamps_us[-1]),
                    ego=ego,
                    frame_timestamps_us=[int(t) for t in frames],
                )
            )
            for r in coc_rows.get(clip_id, []):
                if int(r["index"]) == idx:
                    gt[(clip_id, idx)] = r["coc"]
    return windows, gt


def load_d3d_full_clip_keyframes(keyframes_jsonl: str | Path) -> dict[str, list[Keyframe]]:
    out: dict[str, list[Keyframe]] = {}
    for row in read_jsonl(keyframes_jsonl):
        kfs = []
        for ts, typ, mag in zip(row["keyframe_timestamps_us"], row["keyframe_types"], row["keyframe_magnitudes"]):
            kfs.append(Keyframe(int(ts), typ, float(mag), KEYFRAME_CATEGORY.get(typ, "longitudinal")))
        out[row["clip_id"]] = kfs
    return out
