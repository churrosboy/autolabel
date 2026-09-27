"""Project workspace + stage runners.

A *project* is a directory::

    my_project/
      project.json        settings
      clips.jsonl         registered inputs
      motion/<clip>.json  ego-motion per clip
      windows.jsonl       decision windows (with ego slice)
      frames/<clip>/kfNNN/frame_XX.jpg, sheet.jpg
      labels.jsonl        generated labels (+ filter + review state)
      exports/            training-ready datasets

Every stage is resumable: existing rows are kept unless ``--force``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from .backends import LabelBackend, get_backend
from .filters import FilterConfig, QualityFilter, summarize_reports
from .keyframes import KeyframeConfig, detect_keyframes, uniform_keyframes
from .motion import estimate_from_video, load_sensor_csv, probe_video
from .schemas import Clip, EgoMotion, Label, Window
from .sources import read_jsonl, write_jsonl
from .windows import FrameReader, extract_window_frames, make_windows

log = logging.getLogger("autolabel")


@dataclass
class ProjectConfig:
    name: str = "autolabel"
    before_s: float = 2.0
    after_s: float = 6.0
    n_frames: int = 16
    history_frames: int = 4
    frame_max_side: int = 768
    fallback_uniform_s: float = 0.0   # >0: add uniform windows when no keyframe is found
    keyframe: dict = field(default_factory=dict)
    filter: dict = field(default_factory=dict)


class Project:
    def __init__(self, root: str | Path, labels_file: str = "labels.jsonl"):
        self.root = Path(root)
        self.cfg = ProjectConfig()
        self.labels_file = labels_file          # override to keep several label sets side by side

    # ------------------------------------------------------------------ paths
    @property
    def clips_path(self) -> Path:
        return self.root / "clips.jsonl"

    @property
    def windows_path(self) -> Path:
        return self.root / "windows.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.root / self.labels_file

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def motion_dir(self) -> Path:
        return self.root / "motion"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    # ------------------------------------------------------------------ init/open
    @classmethod
    def create(cls, root: str | Path, name: Optional[str] = None, **cfg) -> "Project":
        p = cls(root)
        p.root.mkdir(parents=True, exist_ok=True)
        p.cfg = ProjectConfig(name=name or p.root.name, **cfg)
        p.save_config()
        for d in (p.frames_dir, p.motion_dir, p.exports_dir):
            d.mkdir(exist_ok=True)
        return p

    @classmethod
    def open(cls, root: str | Path, labels_file: str = "labels.jsonl") -> "Project":
        p = cls(root, labels_file)
        cfg_path = p.root / "project.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"not a project (missing project.json): {root}")
        d = json.loads(cfg_path.read_text())
        p.cfg = ProjectConfig(**{k: v for k, v in d.items() if k in ProjectConfig.__dataclass_fields__})
        return p

    def save_config(self) -> None:
        (self.root / "project.json").write_text(json.dumps(asdict(self.cfg), indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------ clips
    def clips(self) -> list[Clip]:
        if not self.clips_path.exists():
            return []
        return [Clip.from_dict(r) for r in read_jsonl(self.clips_path)]

    def _write_clips(self, clips: Iterable[Clip]) -> None:
        write_jsonl(self.clips_path, (c.to_dict() for c in clips))

    @staticmethod
    def clip_id_for(path: Path) -> str:
        h = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
        return f"{path.stem}_{h}"

    def add_video(self, path: str | Path, copy: bool = False, sensor_csv: Optional[str] = None, meta: Optional[dict] = None) -> Clip:
        src = Path(path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        clips = {c.clip_id: c for c in self.clips()}
        cid = self.clip_id_for(src)
        if cid in clips:
            log.info("clip already registered: %s", cid)
            return clips[cid]
        if copy:
            dst = self.root / "videos" / src.name
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)
            src = dst
        info = probe_video(str(src))
        clip = Clip(
            clip_id=cid,
            path=str(src),
            duration_s=info["duration_s"],
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
            meta={**(meta or {}), **({"sensor_csv": str(Path(sensor_csv).resolve())} if sensor_csv else {})},
        )
        clips[cid] = clip
        self._write_clips(clips.values())
        log.info("registered %s (%.1fs, %dx%d @ %.1ffps)", cid, clip.duration_s, clip.width, clip.height, clip.fps)
        return clip

    def add_folder(self, folder: str | Path, exts=(".mp4", ".mov", ".mkv", ".avi", ".m4v"), **kw) -> list[Clip]:
        out = []
        for p in sorted(Path(folder).expanduser().iterdir()):
            if p.suffix.lower() in exts:
                out.append(self.add_video(p, **kw))
        return out

    # ------------------------------------------------------------------ motion
    def motion_path(self, clip_id: str) -> Path:
        return self.motion_dir / f"{clip_id}.json"

    def load_motion(self, clip_id: str) -> Optional[EgoMotion]:
        p = self.motion_path(clip_id)
        if not p.exists():
            return None
        return EgoMotion.from_dict(json.loads(p.read_text()))

    def run_motion(self, force: bool = False, progress: Optional[Callable[[str], None]] = None) -> dict[str, EgoMotion]:
        out = {}
        self.motion_dir.mkdir(exist_ok=True)
        for clip in self.clips():
            p = self.motion_path(clip.clip_id)
            if p.exists() and not force:
                out[clip.clip_id] = EgoMotion.from_dict(json.loads(p.read_text()))
                continue
            t0 = time.time()
            csv_path = clip.meta.get("sensor_csv")
            if csv_path and Path(csv_path).exists():
                ego = load_sensor_csv(csv_path)
            elif clip.path:
                ego = estimate_from_video(clip.path)
            else:
                log.warning("clip %s has neither video nor sensor data", clip.clip_id)
                continue
            p.write_text(json.dumps(ego.to_dict()))
            out[clip.clip_id] = ego
            msg = f"motion {clip.clip_id}: {ego.n} samples ({ego.source}) in {time.time() - t0:.1f}s"
            log.info(msg)
            if progress:
                progress(msg)
        return out

    # ------------------------------------------------------------------ windows
    def windows(self) -> list[Window]:
        if not self.windows_path.exists():
            return []
        return [Window.from_dict(r) for r in read_jsonl(self.windows_path)]

    def run_keyframes(self, force: bool = False, extract_frames: bool = True, progress=None) -> list[Window]:
        existing = {} if force else {(w.clip_id): True for w in self.windows()}
        all_windows = [] if force else self.windows()
        motions = self.run_motion()
        readers: dict[str, FrameReader] = {}
        for clip in self.clips():
            if clip.clip_id in existing:
                continue
            ego = motions.get(clip.clip_id)
            if ego is None:
                continue
            kcfg = KeyframeConfig.for_source(ego.source)
            for k, v in self.cfg.keyframe.items():
                setattr(kcfg, k, v)
            kcfg.before_s, kcfg.after_s = self.cfg.before_s, self.cfg.after_s
            kfs = detect_keyframes(ego, kcfg)
            if not kfs and self.cfg.fallback_uniform_s > 0:
                kfs = uniform_keyframes(ego, self.cfg.fallback_uniform_s, kcfg)
            wins = make_windows(clip.clip_id, ego, kfs, self.cfg.before_s, self.cfg.after_s, self.cfg.n_frames, self.cfg.history_frames)
            if extract_frames and clip.path:
                reader = readers.setdefault(clip.clip_id, FrameReader(clip.path, max_side=self.cfg.frame_max_side))
                for w in wins:
                    extract_window_frames(clip.path, w, self.frames_dir, reader=reader)
            all_windows.extend(wins)
            msg = f"keyframes {clip.clip_id}: {len(kfs)} -> {len(wins)} windows"
            log.info(msg)
            if progress:
                progress(msg)
        for r in readers.values():
            r.close()
        write_jsonl(self.windows_path, (w.to_dict(include_ego=True) for w in all_windows))
        return all_windows

    def add_windows(self, windows: list[Window]) -> None:
        """Register externally built windows (e.g. imported D3D dataset)."""
        cur = {(w.clip_id, w.index): w for w in self.windows()}
        for w in windows:
            cur[(w.clip_id, w.index)] = w
        write_jsonl(self.windows_path, (w.to_dict(include_ego=True) for w in cur.values()))

    def window_frames(self, w: Window) -> list[np.ndarray]:
        d = self.frames_dir / w.clip_id / f"kf{w.index:03d}"
        if not d.exists():
            return []
        import cv2

        out = []
        for p in sorted(d.glob("frame_*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return out

    # ------------------------------------------------------------------ labels
    def labels(self) -> list[Label]:
        if not self.labels_path.exists():
            return []
        return [Label.from_dict(r) for r in read_jsonl(self.labels_path)]

    def save_labels(self, labels: Iterable[Label]) -> None:
        write_jsonl(self.labels_path, (l.to_dict() for l in labels))

    def run_label(
        self,
        backend: LabelBackend | str = "rule_based",
        force: bool = False,
        limit: Optional[int] = None,
        progress=None,
        run_filter: bool = True,
        backend_kwargs: Optional[dict] = None,
    ) -> list[Label]:
        be = get_backend(backend, **(backend_kwargs or {})) if isinstance(backend, str) else backend
        labels = {(l.clip_id, l.index): l for l in self.labels()}
        qf = QualityFilter(FilterConfig(**self.cfg.filter)) if run_filter else None
        n_done = 0
        windows = self.windows()
        for w in windows:
            key = (w.clip_id, w.index)
            if key in labels and not force and labels[key].backend == be.name:
                continue
            if limit and n_done >= limit:
                break
            frames = self.window_frames(w) if be.needs_frames else None
            try:
                lab = be.label(w, frames)
            except Exception as exc:  # noqa: BLE001
                log.warning("label failed for %s: %s", key, exc)
                continue
            if qf is not None:
                sib = [l.coc for k, l in labels.items() if k[0] == w.clip_id and k != key]
                lab.filter = qf.check(lab.coc, w.keyframe_type, w, siblings=sib, head=lab.meta.get("head_decisions")).to_dict()
            labels[key] = lab
            n_done += 1
            if progress:
                progress(f"[{n_done}] {w.clip_id}#{w.index} {w.keyframe_type}: {lab.coc[:90]}")
            if n_done % 10 == 0:
                self.save_labels(labels.values())
        self.save_labels(labels.values())
        be.close()
        return list(labels.values())

    def run_filter(self, cfg: Optional[FilterConfig] = None) -> dict:
        qf = QualityFilter(cfg or FilterConfig(**self.cfg.filter))
        wins = {(w.clip_id, w.index): w for w in self.windows()}
        labels = self.labels()
        by_clip: dict[str, list[Label]] = {}
        for l in labels:
            by_clip.setdefault(l.clip_id, []).append(l)
        reports = []
        for l in labels:
            w = wins.get((l.clip_id, l.index))
            sib = [o.coc for o in by_clip[l.clip_id] if o is not l]
            rep = qf.check(l.coc, l.keyframe_type, w, siblings=sib, head=(l.meta or {}).get("head_decisions"))
            l.filter = rep.to_dict()
            reports.append(rep)
        self.save_labels(labels)
        return summarize_reports(reports)

    def set_review(self, clip_id: str, index: int, review: str, coc: Optional[str] = None) -> Optional[Label]:
        labels = self.labels()
        hit = None
        for l in labels:
            if l.clip_id == clip_id and l.index == index:
                l.review = review
                if coc is not None and coc.strip() and coc.strip() != l.coc:
                    l.meta.setdefault("history", []).append(l.coc)
                    l.coc = coc.strip()
                    l.review = "edited" if review != "rejected" else review
                hit = l
        if hit is not None:
            self.save_labels(labels)
        return hit

    # ------------------------------------------------------------------ export
    def export(self, name: str = "dataset", include: tuple[str, ...] = ("pass", "review"), reviewed_only: bool = False, copy_frames: bool = True) -> dict:
        """Write a training-ready dataset.

        * ``coc_results.jsonl``  - D3D-compatible {clip_id, coc_results:[...]}
        * ``samples.jsonl``      - flat rows with frame paths, prompt, label
        """
        out = self.exports_dir / name
        out.mkdir(parents=True, exist_ok=True)
        wins = {(w.clip_id, w.index): w for w in self.windows()}
        kept, dropped = [], 0
        for l in self.labels():
            if l.review == "rejected":
                dropped += 1
                continue
            verdict = (l.filter or {}).get("verdict", "pass")
            if l.review in ("approved", "edited"):
                pass
            elif reviewed_only or verdict not in include:
                dropped += 1
                continue
            kept.append(l)
        by_clip: dict[str, list[dict]] = {}
        rows = []
        for l in kept:
            w = wins.get((l.clip_id, l.index))
            by_clip.setdefault(l.clip_id, []).append(
                {"index": l.index, "type": l.keyframe_type, "first_frame_timestamp_us": l.first_frame_timestamp_us, "coc": l.coc}
            )
            frame_dir = self.frames_dir / l.clip_id / f"kf{l.index:03d}"
            frame_paths = sorted(str(p) for p in frame_dir.glob("frame_*.jpg")) if frame_dir.exists() else []
            if copy_frames and frame_paths:
                dst = out / "frames" / l.clip_id / f"kf{l.index:03d}"
                dst.mkdir(parents=True, exist_ok=True)
                new_paths = []
                for p in frame_paths:
                    q = dst / Path(p).name
                    if not q.exists():
                        shutil.copy2(p, q)
                    new_paths.append(str(q.relative_to(out)))
                frame_paths = new_paths
            rows.append(
                {
                    "clip_id": l.clip_id,
                    "index": l.index,
                    "keyframe_type": l.keyframe_type,
                    "keyframe_timestamp_us": l.keyframe_timestamp_us,
                    "first_frame_timestamp_us": l.first_frame_timestamp_us,
                    "frame_timestamps_us": w.frame_timestamps_us if w else [],
                    "history_frames": w.history_frames if w else self.cfg.history_frames,
                    "frames": frame_paths,
                    "coc": l.coc,
                    "backend": l.backend,
                    "filter_verdict": verdict,
                    "filter_score": (l.filter or {}).get("score"),
                    "review": l.review,
                }
            )
        write_jsonl(out / "coc_results.jsonl", ({"clip_id": c, "coc_results": r} for c, r in by_clip.items()))
        write_jsonl(out / "samples.jsonl", rows)
        summary = {"export_dir": str(out), "kept": len(kept), "dropped": dropped, "clips": len(by_clip)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        from collections import Counter

        labels = self.labels()
        wins = self.windows()
        return {
            "clips": len(self.clips()),
            "windows": len(wins),
            "keyframe_types": dict(Counter(w.keyframe_type for w in wins)),
            "labels": len(labels),
            "backends": dict(Counter(l.backend for l in labels)),
            "verdicts": dict(Counter((l.filter or {}).get("verdict", "n/a") for l in labels)),
            "review": dict(Counter(l.review for l in labels)),
        }
