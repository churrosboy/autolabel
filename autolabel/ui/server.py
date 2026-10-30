"""Tiny review / timeline UI (standard library only).

GET  /                       -> index.html      (label list)
GET  /timeline               -> timeline.html   (clip timeline: video + motion + keyframes)
GET  /api/labels             -> [{clip_id,index,keyframe_type,coc,filter,review,sheet}]
GET  /api/stats
GET  /api/clips              -> [{clip_id,duration_s,has_video,has_motion,n_windows,n_labels,verdicts}]
GET  /api/clip/<clip_id>     -> {clip, motion, candidates, windows[{..., label, frames, kinematics}], cfg, vocab}
GET  /video/<clip_id>        -> video file (HTTP range requests supported)
GET  /frames/<clip>/kfNNN/sheet.jpg | frame_XX.jpg
POST /api/review             {clip_id,index,review,coc?}
POST /api/keyframe           {clip_id,action:add|update|delete,index?,timestamp_us?,type?}
POST /api/label_one          {clip_id,index,backend?,backend_args?}
POST /api/export             {name?,reviewed_only?}
"""
from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from ..keyframes import KeyframeConfig, explain_keyframes
from ..motion import moving_average, summarize_window
from ..pipeline import Project
from ..schemas import EgoMotion, Window
from ..vocab import KEYFRAME_TYPES, LATERAL, LONGITUDINAL

HERE = Path(__file__).parent
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo", ".webm": "video/webm"}
MAX_POINTS = 6000


def _downsample(n: int) -> int:
    return max(1, int(np.ceil(n / MAX_POINTS)))


def _motion_payload(project: Project, clip_id: str, windows: list[Window]):
    """Signals the timeline draws, in the units the detector used (smoothed like the detector)."""
    ego = project.load_motion(clip_id)
    if ego is None:
        # imported datasets (D3D) carry only per-window ego slices: stitch them
        parts = sorted((w.ego for w in windows if w.ego.n), key=lambda e: int(e.timestamps_us[0]))
        if not parts:
            return None
        ts = np.concatenate([e.timestamps_us for e in parts])
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        keep = np.concatenate([[True], np.diff(ts) > 0])
        cat = lambda a: np.concatenate([getattr(e, a) for e in parts])[order][keep]  # noqa: E731
        ego = EgoMotion(ts[keep], cat("positions"), cat("velocities"), cat("accelerations"), cat("curvatures"), source=parts[0].source)
    kcfg = KeyframeConfig.for_source(ego.source)
    for k, v in project.cfg.keyframe.items():
        setattr(kcfg, k, v)
    kcfg.before_s, kcfg.after_s = project.cfg.before_s, project.cfg.after_s
    s = _downsample(ego.n)
    t = ego.timestamps_us / 1e6
    r = lambda a: [round(float(x), 4) for x in np.asarray(a)[::s]]  # noqa: E731
    return {
        "source": ego.source,
        "t": r(t),
        "speed": r(ego.speed),
        "ax": r(ego.ax),
        "ax_smooth": r(moving_average(ego.ax, kcfg.smooth_window)),
        "curv": r(ego.curvatures),
        "curv_abs_smooth": r(moving_average(np.abs(ego.curvatures), kcfg.smooth_window)),
        "x": r(ego.positions[:, 0]),
        "y": r(ego.positions[:, 1]),
        "t0": float(t[0]),
        "t1": float(t[-1]),
        "thresholds": {k: getattr(kcfg, k) for k in ("strong_accel", "gentle_accel", "gentle_decel", "strong_decel", "stop_speed", "steer", "sharp_steer")},
        "candidates": explain_keyframes(ego, kcfg),
    }


def _window_payload(project: Project, w: Window, label) -> dict:
    d = w.to_dict(include_ego=False)
    d["kinematics"] = summarize_window(w.ego, w.keyframe.timestamp_us, project.cfg.before_s, project.cfg.after_s)
    s = _downsample(w.ego.n)
    d["path"] = {"x": [round(float(v), 4) for v in w.ego.positions[::s, 0]], "y": [round(float(v), 4) for v in w.ego.positions[::s, 1]],
                 "t": [round(float(v) / 1e6, 3) for v in w.ego.timestamps_us[::s]], "speed": [round(float(v), 4) for v in w.ego.speed[::s]]}
    fdir = project.frames_dir / w.clip_id / f"kf{w.index:03d}"
    d["frames"] = [f"/frames/{w.clip_id}/kf{w.index:03d}/{p.name}" for p in sorted(fdir.glob("frame_*.jpg"))] if fdir.exists() else []
    d["sheet"] = f"/frames/{w.clip_id}/kf{w.index:03d}/sheet.jpg" if (fdir / "sheet.jpg").exists() else None
    # decision-moment snapshots (e.g. extracted from a D3D training cache) when the 16 window frames are not available
    d["snaps"], d["snap_timestamps_us"], d["snap_note"] = [], [], None
    if not d["frames"] and (fdir / "snaps.json").exists():
        try:
            meta = json.loads((fdir / "snaps.json").read_text())
            d["snaps"] = [f"/frames/{w.clip_id}/kf{w.index:03d}/{p.name}" for p in sorted(fdir.glob("snap_*.jpg"))]
            d["snap_timestamps_us"] = [int(t) for t in meta.get("timestamps_us", [])][: len(d["snaps"])]
            d["snap_history"] = int(meta.get("history", 4))
            d["snap_note"] = meta.get("note")
        except (ValueError, OSError):
            pass
    if label is not None:
        ld = label.to_dict()
        ld.pop("raw", None)
        d["label"] = ld
    else:
        d["label"] = None
    return d


_UNSET = object()


class _Cache:
    """Re-parse windows.jsonl / labels.jsonl only when the file changed (dataset projects hold 700+ windows)."""

    def __init__(self, project: Project):
        self.project = project
        self._w = (_UNSET, None)   # stamp is None when the file does not exist, so start from a sentinel
        self._l = (_UNSET, None)

    @staticmethod
    def _stamp(p: Path):
        try:
            st = p.stat()
            return (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            return None

    def windows(self) -> list[Window]:
        k = self._stamp(self.project.windows_path)
        if k != self._w[0]:
            self._w = (k, self.project.windows())
        return self._w[1]

    def labels(self):
        k = self._stamp(self.project.labels_path)
        if k != self._l[0]:
            self._l = (k, self.project.labels())
        return self._l[1]


def make_handler(project: Project):
    cache = _Cache(project)

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        # ------------------------------------------------------------ helpers
        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _bytes(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, p: Path, ctype: str):
            """Serve a file with HTTP range support (needed for <video> seeking)."""
            size = p.stat().st_size
            start, end, code = 0, size - 1, 200
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                a, _, b = rng[6:].partition("-")
                try:
                    if a:
                        start, end = int(a), (int(b) if b else size - 1)
                    else:
                        start = max(0, size - int(b))
                    end = min(end, size - 1)
                    code = 206
                except ValueError:
                    start, end, code = 0, size - 1, 200
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            length = end - start + 1
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with p.open("rb") as f:
                    f.seek(start)
                    left = length
                    while left > 0:
                        chunk = f.read(min(1 << 20, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _clip_payload(self, clip_id: str):
            clip = next((c for c in project.clips() if c.clip_id == clip_id), None)
            wins = [w for w in cache.windows() if w.clip_id == clip_id]
            vid = project.video_for(clip_id)
            if clip is None and not wins and vid is None:
                return None
            labels = {(l.clip_id, l.index): l for l in cache.labels()}
            motion = _motion_payload(project, clip_id, wins)
            has_video = vid is not None
            vmeta = vid[1] if vid else {}
            # dataset clips may carry absolute (epoch-like) timestamps: the UI then draws relative to t_offset_s;
            # a dataset video's sidecar gives the clip time of playback position 0 instead
            lo = min(([motion["t0"]] if motion else []) + [w.start_us / 1e6 for w in wins]) if (motion or wins) else 0.0
            hi = max(([motion["t1"]] if motion else []) + [w.end_us / 1e6 for w in wins]) if (motion or wins) else 0.0
            if has_video:
                t_off = float(vmeta.get("t0_us", 0)) / 1e6
            else:
                t_off = lo if lo > 3600.0 else 0.0
            duration = (clip.duration_s if clip and clip.duration_s else 0.0) or float(vmeta.get("duration_s") or 0.0) or (hi - t_off)
            return {
                "t_offset_s": t_off,
                "video_meta": vmeta,
                "clip": clip.to_dict() if clip else {"clip_id": clip_id, "path": None, "duration_s": duration},
                "video": f"/video/{clip_id}" if has_video else None,
                "duration_s": duration,
                "motion": motion,
                "windows": [_window_payload(project, w, labels.get((w.clip_id, w.index))) for w in sorted(wins, key=lambda w: w.keyframe.timestamp_us)],
                "cfg": {"before_s": project.cfg.before_s, "after_s": project.cfg.after_s, "n_frames": project.cfg.n_frames, "history_frames": project.cfg.history_frames},
                "vocab": {"longitudinal": LONGITUDINAL, "lateral": LATERAL, "keyframe_types": KEYFRAME_TYPES},
            }

        # ------------------------------------------------------------ GET
        def do_GET(self):
            u = urlparse(self.path)
            path = unquote(u.path)
            try:
                if path in ("/", "/index.html"):
                    self._bytes((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif path in ("/timeline", "/timeline.html"):
                    self._bytes((HERE / "timeline.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/labels":
                    rows = []
                    for l in project.labels():
                        sheet = project.frames_dir / l.clip_id / f"kf{l.index:03d}" / "sheet.jpg"
                        d = l.to_dict()
                        d["sheet"] = f"/frames/{l.clip_id}/kf{l.index:03d}/sheet.jpg" if sheet.exists() else None
                        d.pop("raw", None)
                        rows.append(d)
                    self._json(rows)
                elif path == "/api/stats":
                    self._json(project.stats())
                elif path == "/api/clips":
                    wins = cache.windows()
                    labels = cache.labels()
                    clips = {c.clip_id: c for c in project.clips()}
                    ids = list(clips) + sorted({w.clip_id for w in wins} - set(clips))
                    rows = []
                    for cid in ids:
                        c = clips.get(cid)
                        ls = [l for l in labels if l.clip_id == cid]
                        verdicts: dict[str, int] = {}
                        for l in ls:
                            v = (l.filter or {}).get("verdict", "n/a")
                            verdicts[v] = verdicts.get(v, 0) + 1
                        rows.append({
                            "clip_id": cid,
                            "path": c.path if c else None,
                            "duration_s": c.duration_s if c else 0.0,
                            "has_video": project.video_for(cid) is not None,
                            "has_motion": project.motion_path(cid).exists(),
                            "has_frames": (project.frames_dir / cid).exists(),
                            "n_windows": sum(1 for w in wins if w.clip_id == cid),
                            "n_labels": len(ls),
                            "verdicts": verdicts,
                        })
                    self._json(rows)
                elif path.startswith("/api/clip/"):
                    payload = self._clip_payload(path[len("/api/clip/"):])
                    self._json(payload) if payload else self.send_error(404)
                elif path.startswith("/video/"):
                    cid = path[len("/video/"):]
                    vid = project.video_for(cid)
                    if vid is None:
                        self.send_error(404)
                        return
                    p = vid[0]
                    self._file(p, VIDEO_TYPES.get(p.suffix.lower(), "application/octet-stream"))
                elif path.startswith("/frames/"):
                    p = (project.frames_dir / path[len("/frames/"):]).resolve()
                    if project.frames_dir.resolve() in p.parents and p.exists():
                        self._file(p, "image/jpeg")
                    else:
                        self.send_error(404)
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json({"error": str(exc)}, 500)

        # ------------------------------------------------------------ POST
        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n) or b"{}")
            try:
                if u.path == "/api/review":
                    lab = project.set_review(data["clip_id"], int(data["index"]), data.get("review", "approved"), data.get("coc"))
                    self._json({"ok": lab is not None, "label": lab.to_dict() if lab else None})
                elif u.path == "/api/keyframe":
                    w = project.edit_window(
                        data["clip_id"], data["action"],
                        index=int(data["index"]) if data.get("index") is not None else None,
                        timestamp_us=int(data["timestamp_us"]) if data.get("timestamp_us") is not None else None,
                        type=data.get("type"),
                    )
                    if w is None:
                        self._json({"ok": True, "window": None})
                    else:
                        lab = next((l for l in cache.labels() if l.clip_id == w.clip_id and l.index == w.index), None)
                        self._json({"ok": True, "window": _window_payload(project, w, lab)})
                elif u.path == "/api/label_one":
                    key = (data["clip_id"], int(data["index"]))
                    labels = project.run_label(data.get("backend", "rule_based"), force=True, only=[key],
                                               backend_kwargs=data.get("backend_args") or None)
                    lab = next((l for l in labels if (l.clip_id, l.index) == key), None)
                    if lab is None:
                        self._json({"ok": False, "error": "backend produced no label"}, 500)
                    else:
                        d = lab.to_dict()
                        d.pop("raw", None)
                        self._json({"ok": True, "label": d})
                elif u.path == "/api/export":
                    self._json(project.export(data.get("name", "dataset"), reviewed_only=bool(data.get("reviewed_only"))))
                else:
                    self.send_error(404)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json({"ok": False, "error": str(exc)}, 400)

    return H


def serve(project: Project, port: int = 8765, open_browser: bool = True, page: str = "timeline"):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(project))
    base = f"http://127.0.0.1:{port}/"
    print(f"timeline UI: {base}timeline\nreview list: {base}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(base + ("timeline" if page == "timeline" else ""))).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
