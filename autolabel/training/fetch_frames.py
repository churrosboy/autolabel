"""Fetch frames for imported D3D windows from the NVIDIA Physical AI AV dataset.

Requires ``pip install physical-ai-av`` and ``HF_TOKEN`` (gated dataset).
For every window in the project it decodes the 16 frame timestamps from the
front-wide camera (``--layout front``) or composes the same 2x2 four-camera
grid the Gemini labels were generated from (``--layout grid``, default) and
writes frames/<clip>/kfNNN/frame_XX.jpg + sheet.jpg.  Resumable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..pipeline import Project
from ..windows import contact_sheet, save_image

log = logging.getLogger("autolabel.fetch")

CAMS = {
    "front_wide": "CAMERA_FRONT_WIDE_120FOV",
    "front_tele": "CAMERA_FRONT_TELE_30FOV",
    "cross_left": "CAMERA_CROSS_LEFT_120FOV",
    "cross_right": "CAMERA_CROSS_RIGHT_120FOV",
}


def _grid(frames_by_cam: dict[str, np.ndarray], i: int, cell=(640, 360)) -> np.ndarray:
    import cv2

    def get(name):
        arr = frames_by_cam.get(name)
        if arr is None:
            return np.zeros((cell[1], cell[0], 3), dtype=np.uint8)
        return cv2.resize(arr[i], cell, interpolation=cv2.INTER_AREA)

    top = np.hstack([get("front_wide"), get("front_tele")])
    bottom = np.hstack([get("cross_left"), get("cross_right")])
    return np.vstack([top, bottom])


def fetch(project_dir: str | Path, layout: str = "grid", limit: Optional[int] = None, force: bool = False, max_side: int = 960) -> dict:
    import physical_ai_av  # type: ignore

    p = Project.open(project_dir)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    cams = list(CAMS) if layout == "grid" else ["front_wide"]
    wins = p.windows()
    by_clip: dict[str, list] = {}
    for w in wins:
        by_clip.setdefault(w.clip_id, []).append(w)
    done = skipped = failed = 0
    for n_clip, (clip_id, cw) in enumerate(by_clip.items()):
        if limit and n_clip >= limit:
            break
        todo = [w for w in cw if force or not (p.frames_dir / clip_id / f"kf{w.index:03d}" / "sheet.jpg").exists()]
        if not todo:
            skipped += len(cw)
            continue
        readers = {}
        try:
            for c in cams:
                readers[c] = avdi.get_clip_feature(clip_id, getattr(avdi.features.CAMERA, CAMS[c]), maybe_stream=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("clip %s: cannot open cameras: %s", clip_id, exc)
            failed += len(todo)
            continue
        for w in todo:
            ts = np.asarray(w.frame_timestamps_us, dtype=np.int64)
            frames_by_cam = {}
            for c, reader in readers.items():
                try:
                    imgs, _ = reader.decode_images_from_timestamps(ts)      # (N, H, W, 3) uint8 RGB
                    frames_by_cam[c] = np.asarray(imgs)
                except Exception as exc:  # noqa: BLE001
                    log.warning("clip %s kf%d cam %s: %s", clip_id, w.index, c, exc)
            if not frames_by_cam:
                failed += 1
                continue
            out_dir = p.frames_dir / clip_id / f"kf{w.index:03d}"
            frames = []
            for i in range(len(ts)):
                img = _grid(frames_by_cam, i) if layout == "grid" else frames_by_cam["front_wide"][i]
                h, wd = img.shape[:2]
                s = max_side / float(max(h, wd))
                if s < 1.0:
                    import cv2

                    img = cv2.resize(img, (int(wd * s), int(h * s)), interpolation=cv2.INTER_AREA)
                save_image(out_dir / f"frame_{i:02d}.jpg", img)
                frames.append(img)
            save_image(out_dir / "sheet.jpg", contact_sheet(frames, history_frames=w.history_frames))
            done += 1
        log.info("clip %s: %d windows done (total done=%d)", clip_id, len(todo), done)
    return {"done": done, "skipped": skipped, "failed": failed, "layout": layout}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="fetch physical_ai_av frames for imported D3D windows")
    ap.add_argument("project")
    ap.add_argument("--layout", choices=["grid", "front"], default="grid")
    ap.add_argument("--limit", type=int, help="first N clips only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-side", type=int, default=960)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(fetch(a.project, a.layout, a.limit, a.force, a.max_side))


if __name__ == "__main__":
    main()
