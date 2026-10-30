"""Timeline UI support: keyframe explanations, hand-edited windows, per-window labelling, HTTP API."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from autolabel.keyframes import KeyframeConfig, detect_keyframes, explain_keyframes
from autolabel.pipeline import Project
from autolabel.schemas import Keyframe
from autolabel.windows import make_windows
from conftest import synth_ego


def test_explain_keyframes_marks_kept_and_dropped(ego_brake):
    cfg = KeyframeConfig()
    kept = {(k.timestamp_us, k.type) for k in detect_keyframes(ego_brake, cfg)}
    rows = explain_keyframes(ego_brake, cfg)
    assert {(r["timestamp_us"], r["type"]) for r in rows if r["status"] == "kept"} == kept
    assert {r["status"] for r in rows} <= {"kept", "unconfirmed", "cooldown", "window_cap", "margin"}
    # without the speed check nothing can be "unconfirmed"
    rows2 = explain_keyframes(ego_brake, KeyframeConfig(confirm_with_speed=False))
    assert all(r["status"] != "unconfirmed" for r in rows2)
    # a decel transition that the speed trace does not confirm is reported as unconfirmed
    ego = synth_ego("cruise")
    ego.accelerations[50:60, 0] = -2.0            # fake braking without any speed drop
    rows3 = explain_keyframes(ego, KeyframeConfig())
    assert any(r["type"] == "gentle_decel" and r["status"] == "unconfirmed" for r in rows3)
    assert not any(r["type"] == "gentle_decel" and r["status"] == "kept" for r in rows3)


@pytest.fixture
def sensor_project(tmp_path, ego_brake):
    p = Project.create(tmp_path / "proj")
    cid = "clip_a"
    p.motion_dir.mkdir(exist_ok=True)
    p.motion_path(cid).write_text(json.dumps(ego_brake.to_dict()))
    kfs = detect_keyframes(ego_brake)
    p.add_windows(make_windows(cid, ego_brake, kfs))
    p.run_label("rule_based")
    return p


def test_edit_window_add_update_delete(sensor_project):
    p = sensor_project
    n0 = len(p.windows())
    w = p.edit_window("clip_a", "add", timestamp_us=5_000_000, type="go_straight")
    assert w.index == n0 and w.keyframe.type == "go_straight" and w.keyframe.category == "lateral"
    assert len(p.windows()) == n0 + 1 and w.ego.n > 0 and len(w.frame_timestamps_us) == 16
    # label it, then move it: the label survives but goes back to unreviewed + re-filtered
    p.run_label("rule_based", only=[("clip_a", w.index)])
    lab = next(l for l in p.labels() if l.index == w.index)
    p.set_review("clip_a", w.index, "approved")
    w2 = p.edit_window("clip_a", "update", index=w.index, timestamp_us=6_000_000, type="gentle_decel")
    assert w2.keyframe.timestamp_us == 6_000_000 and w2.start_us == 4_000_000 and w2.keyframe.category == "longitudinal"
    lab2 = next(l for l in p.labels() if l.index == w.index)
    assert lab2.coc == lab.coc and lab2.review == "unreviewed" and lab2.meta.get("window_edited")
    assert lab2.keyframe_type == "gentle_decel" and lab2.keyframe_timestamp_us == 6_000_000 and lab2.filter.get("verdict")
    # other windows untouched
    assert sum(1 for l in p.labels() if l.review == "unreviewed") == len(p.labels())
    p.edit_window("clip_a", "delete", index=w.index)
    assert len(p.windows()) == n0 and all(l.index != w.index for l in p.labels())
    with pytest.raises(KeyError):
        p.edit_window("clip_a", "update", index=999, timestamp_us=1)
    with pytest.raises(ValueError):
        p.edit_window("clip_a", "add", timestamp_us=19_900_000)   # window would leave the clip


def test_run_label_only_touches_requested_windows(sensor_project):
    p = sensor_project
    before = {l.index: l.coc for l in p.labels()}
    target = p.windows()[0].index
    p.run_label("rule_based", force=True, only=[("clip_a", target)])
    after = {l.index: l.coc for l in p.labels()}
    assert set(after) == set(before)


def test_http_api(sensor_project):
    from autolabel.ui.server import make_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(sensor_project))
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return r.status, r.headers, r.read()

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    try:
        st, _, body = get("/timeline")
        assert st == 200 and b"CoCaKiLa Timeline" in body
        clips = json.loads(get("/api/clips")[2])
        assert clips[0]["clip_id"] == "clip_a" and clips[0]["n_windows"] > 0 and clips[0]["has_motion"]
        d = json.loads(get("/api/clip/clip_a")[2])
        assert d["motion"]["source"] == "sensor" and len(d["motion"]["t"]) == 200
        assert set(d["motion"]["thresholds"]) >= {"gentle_decel", "steer"}
        assert d["windows"] and d["windows"][0]["label"]["coc"] and "heading_change_deg" in d["windows"][0]["kinematics"]
        assert d["video"] is None
        # decision-moment snapshots are exposed when the window frames are absent
        w0 = d["windows"][0]
        sd = sensor_project.frames_dir / "clip_a" / f"kf{w0['index']:03d}"
        sd.mkdir(parents=True)
        for i in range(3):
            (sd / f"snap_{i:02d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (sd / "snaps.json").write_text(json.dumps({"timestamps_us": [1, 2, 3], "history": 2, "note": "n"}))
        d2 = json.loads(get("/api/clip/clip_a")[2])
        w2 = next(w for w in d2["windows"] if w["index"] == w0["index"])
        assert len(w2["snaps"]) == 3 and w2["snap_timestamps_us"] == [1, 2, 3] and w2["snap_history"] == 2 and w2["frames"] == []
        r = post("/api/keyframe", {"clip_id": "clip_a", "action": "add", "timestamp_us": 5_000_000, "type": "steer_l"})
        assert r["ok"] and r["window"]["keyframe"]["type"] == "steer_l" and r["window"]["label"] is None
        idx = r["window"]["index"]
        r = post("/api/label_one", {"clip_id": "clip_a", "index": idx})
        assert r["ok"] and r["label"]["filter"]["verdict"] in ("pass", "review", "reject")
        r = post("/api/review", {"clip_id": "clip_a", "index": idx, "review": "approved"})
        assert r["ok"] and r["label"]["review"] == "approved"
        r = post("/api/keyframe", {"clip_id": "clip_a", "action": "delete", "index": idx})
        assert r["ok"] and r["window"] is None
        with pytest.raises(urllib.error.HTTPError):
            get("/api/clip/nope")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dataset_video_sidecar(tmp_path):
    """A dataset clip (no registered input file) gets its player from video/<clip>.mp4 + .video.json."""
    import json
    import sys
    sys.path.insert(0, str(tmp_path.parent))
    from autolabel.ui.server import make_handler

    p = Project.create(tmp_path / "proj")
    assert p.video_for("nope") is None
    p.video_dir.mkdir()
    (p.video_dir / "c1.mp4").write_bytes(b"\0" * 100)
    (p.video_dir / "c1.video.json").write_text(json.dumps({"t0_us": -88725, "duration_s": 20.133, "camera": "camera_front_wide_120fov"}))
    path, meta = p.video_for("c1")
    assert path.name == "c1.mp4" and meta["t0_us"] == -88725
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(p))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/video/c1", headers={"Range": "bytes=0-9"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and len(r.read()) == 10
        with urllib.request.urlopen(base + "/api/clip/c1") as r:
            d = json.loads(r.read())
        assert d["video"] == "/video/c1" and abs(d["t_offset_s"] + 0.088725) < 1e-9 and d["duration_s"] == 20.133
        assert d["video_meta"]["camera"] == "camera_front_wide_120fov"
    finally:
        httpd.shutdown()


def test_video_range_requests(tmp_path):
    """Range requests on /video/<clip> (what <video> seeking relies on)."""
    import sys
    sys.path.insert(0, str(tmp_path.parent))
    from autolabel.ui.server import make_handler
    from autolabel.schemas import Clip

    p = Project.create(tmp_path / "proj")
    vid = tmp_path / "v.mp4"
    vid.write_bytes(bytes(range(256)) * 4)
    p._write_clips([Clip(clip_id="v", path=str(vid), duration_s=1.0)])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(p))
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/video/v", headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and r.headers["Content-Range"] == "bytes 10-19/1024" and r.read() == bytes(range(10, 20))
        req = urllib.request.Request(base + "/video/v", headers={"Range": "bytes=-4"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and r.read() == bytes([252, 253, 254, 255])
        with urllib.request.urlopen(base + "/video/v") as r:
            assert r.status == 200 and r.headers["Accept-Ranges"] == "bytes" and len(r.read()) == 1024
    finally:
        httpd.shutdown()
        httpd.server_close()
