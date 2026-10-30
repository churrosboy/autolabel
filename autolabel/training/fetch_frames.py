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


def fetch(project_dir: str | Path, layout: str = "grid", limit: Optional[int] = None, force: bool = False, max_side: int = 960,
          clips: Optional[list[str]] = None) -> dict:
    import physical_ai_av  # type: ignore

    p = Project.open(project_dir)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    cams = list(CAMS) if layout == "grid" else ["front_wide"]
    wins = p.windows()
    by_clip: dict[str, list] = {}
    for w in wins:
        if clips and not any(w.clip_id.startswith(c) for c in clips):
            continue
        by_clip.setdefault(w.clip_id, []).append(w)
    done = skipped = failed = 0
    for n_clip, (clip_id, cw) in enumerate(by_clip.items()):
        if limit and n_clip >= limit:
            break
        todo = [w for w in cw if force or not (p.frames_dir / clip_id / f"kf{w.index:03d}" / "frame_00.jpg").exists()]   # sheet.jpg alone may come from D3D snapshots
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


def fetch_video(project_dir: str | Path, clips: list[str], camera: str = "front_wide", h264: bool = True, force: bool = False) -> dict:
    """Pull whole-clip camera videos for the timeline player into <project>/video/<clip_id>.mp4.

    The dataset packs ~97 clips per chunk zip; the clip's mp4 + frame-timestamp members are
    read through HfFileSystem range requests, so only that clip is transferred.  A sidecar
    <clip_id>.video.json records ``t0_us`` (clip time at playback position 0) for the UI.
    The source is HEVC; with ffmpeg available it is transcoded to H.264 for browsers.
    """
    import io
    import json
    import shutil
    import subprocess
    import zipfile

    import huggingface_hub as hf
    import pandas as pd
    import physical_ai_av  # type: ignore

    p = Project.open(project_dir)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    feat = CAMS[camera].lower()
    ids = sorted({w.clip_id for w in p.windows() if any(w.clip_id.startswith(c) for c in clips)})
    fs = getattr(avdi, "fs", None) or hf.HfFileSystem(token=getattr(avdi, "token", None))
    out_dir = p.video_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    done, skipped = [], []
    for clip_id in ids:
        out = out_dir / f"{clip_id}.mp4"
        if out.exists() and not force:
            skipped.append(clip_id)
            continue
        chunk = int(avdi.get_clip_chunk(clip_id))
        path = f"datasets/{avdi.repo_id}/camera/{feat}/{feat}.chunk_{chunk:04d}.zip"
        with fs.open(path, "rb", block_size=8 * 2**20) as f:
            zf = zipfile.ZipFile(f)
            video = zf.read(f"{clip_id}.{feat}.mp4")
            ts = pd.read_parquet(io.BytesIO(zf.read(f"{clip_id}.{feat}.timestamps.parquet")))["timestamp"].to_numpy()
        raw = out_dir / f"{clip_id}.src.mp4"
        raw.write_bytes(video)
        codec = "hevc"
        if h264 and shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
            raw.unlink()
            codec = "h264"
        else:
            raw.rename(out)
        meta = {"clip_id": clip_id, "camera": feat, "t0_us": int(ts[0]), "t1_us": int(ts[-1]), "n_timestamps": int(len(ts)),
                "duration_s": round((int(ts[-1]) - int(ts[0])) / 1e6, 3), "codec": codec,
                "source": f"{avdi.repo_id} {feat} chunk {chunk}"}
        try:
            import cv2

            cap = cv2.VideoCapture(str(out))
            meta.update(fps=round(cap.get(cv2.CAP_PROP_FPS), 3), width=int(cap.get(3)), height=int(cap.get(4)))
            cap.release()
        except Exception:  # noqa: BLE001
            pass
        (out_dir / f"{clip_id}.video.json").write_text(json.dumps(meta, indent=1))
        log.info("clip %s: video %s (%.1f MB, %s)", clip_id, out.name, out.stat().st_size / 1e6, codec)
        done.append(clip_id)
    return {"done": done, "skipped": skipped, "camera": camera}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="fetch physical_ai_av frames for imported D3D windows")
    ap.add_argument("project")
    ap.add_argument("--layout", choices=["grid", "front"], default="grid")
    ap.add_argument("--limit", type=int, help="first N clips only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--clips", nargs="+", help="only these clip ids (prefix match)")
    ap.add_argument("--video", action="store_true", help="also pull the whole-clip camera mp4 for the timeline player (needs --clips)")
    ap.add_argument("--camera", choices=list(CAMS), default="front_wide")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(fetch(a.project, a.layout, a.limit, a.force, a.max_side, a.clips))
    if a.video:
        if not a.clips:
            ap.error("--video needs --clips (one clip's video is ~15 MB; refusing to pull all)")
        print(fetch_video(a.project, a.clips, a.camera, force=a.force))


if __name__ == "__main__":
    main()
