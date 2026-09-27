"""Reproduce every number and figure used in the interim report.

Outputs
-------
results/metrics.json          all numbers
results/filter_examples.json  example labels per issue code
report/figures/*.png          figures (matplotlib, Helvetica)

Experiments
-----------
E1  keyframe detection      : re-detection of the D3D keyframe type inside its
                              own 8 s window, with / without speed confirmation;
                              distribution of keyframe types in the 929-clip file.
E2  quality filter on GT    : verdict distribution and issue histogram over the
                              724 Gemini-generated labels; per-type reject rate.
E3  rule-based baseline     : decision-F1 / ROUGE-L against the Gemini labels.
E4  video ingestion         : optical-flow ego-motion on a user video + timing.
E5  filter on rule-based    : sanity (labels built from motion should pass K-checks).
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.backends.rule_based import RuleBasedBackend  # noqa: E402
from autolabel.evaluate import evaluate_pairs  # noqa: E402
from autolabel.filters import QualityFilter, summarize_reports  # noqa: E402
from autolabel.keyframes import KeyframeConfig, detect_keyframes  # noqa: E402
from autolabel.motion import estimate_from_video, probe_video  # noqa: E402
from autolabel.sources import load_d3d, load_d3d_full_clip_keyframes  # noqa: E402
from autolabel.vocab import KEYFRAME_CATEGORY  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
C1, C2, C3, C4 = "#2f4f7f", "#9aa0a6", "#b45309", "#1f2937"   # navy, grey, accent (keyframes only), dark
LIGHT = "#d1d5db"

metrics: dict = {}
examples: dict = {}

wins, gt = load_d3d(SAMPLES / "ego_motion_results.jsonl", SAMPLES / "coc_results_pro.jsonl")
print(f"loaded {len(wins)} windows / {len(gt)} GT labels")
metrics["dataset"] = {
    "clips": len({w.clip_id for w in wins}),
    "windows": len(wins),
    "gt_labels": len(gt),
    "keyframe_types": dict(collections.Counter(w.keyframe_type for w in wins).most_common()),
    "coc_len_chars": {"mean": float(np.mean([len(v) for v in gt.values()])), "min": min(len(v) for v in gt.values()), "max": max(len(v) for v in gt.values())},
}

# --------------------------------------------------------------------------- #
# E1 keyframe detection
# --------------------------------------------------------------------------- #
def redetect(confirm: bool) -> dict:
    cfg = KeyframeConfig(per_window_cap=False, start_offset_s=0.0, before_s=0.0, after_s=0.0, confirm_with_speed=confirm)
    hit_cat, hit_type, n = collections.Counter(), collections.Counter(), collections.Counter()
    for w in wins:
        kfs = detect_keyframes(w.ego, cfg)
        kf_us = w.keyframe.timestamp_us
        near = [k for k in kfs if abs(k.timestamp_us - kf_us) <= 1_500_000]
        cat = KEYFRAME_CATEGORY.get(w.keyframe_type, "longitudinal")
        n[w.keyframe_type] += 1
        if any(k.category == cat for k in near):
            hit_cat[w.keyframe_type] += 1
        if any(k.type == w.keyframe_type for k in near):
            hit_type[w.keyframe_type] += 1
    per_type = {t: {"n": n[t], "same_category": hit_cat[t] / n[t], "same_type": hit_type[t] / n[t]} for t in n}
    tot = sum(n.values())
    return {"overall_same_category": sum(hit_cat.values()) / tot, "overall_same_type": sum(hit_type.values()) / tot, "per_type": per_type}


metrics["E1_redetection"] = {"no_confirm": redetect(False), "speed_confirm": redetect(True)}

# how many D3D decel/accel keyframes are not supported by the speed trace?
unsupported = collections.Counter()
for w in wins:
    from autolabel.motion import summarize_window

    k = summarize_window(w.ego, w.keyframe.timestamp_us)
    t = w.keyframe_type
    if t.endswith("decel"):
        unsupported["decel_total"] += 1
        unsupported["decel_no_speed_drop"] += k["speed_drop"] < 1.0
    if t.endswith("accel"):
        unsupported["accel_total"] += 1
        unsupported["accel_no_speed_rise"] += (k["speed_max_future"] - k["speed_before"]) < 1.0
metrics["E1_unsupported_by_speed"] = dict(unsupported)

full = load_d3d_full_clip_keyframes(SAMPLES / "keyframes_partial.jsonl")
per_clip = [len(v) for v in full.values()]
type_counts = collections.Counter(k.type for v in full.values() for k in v)
metrics["E1_full_clip_keyframes"] = {"clips": len(full), "total": sum(per_clip), "per_clip_mean": float(np.mean(per_clip)), "per_clip_median": float(np.median(per_clip)), "types": dict(type_counts.most_common())}

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
types_sorted = [t for t, _ in type_counts.most_common()]
ax[0].barh(types_sorted[::-1], [type_counts[t] for t in types_sorted][::-1], color=[C1 if KEYFRAME_CATEGORY[t] == "longitudinal" else C2 for t in types_sorted][::-1])
ax[0].set_xlabel("keyframes (929 clips)")
ax[0].set_title("(a) Detected keyframe types", loc="left")
r0 = metrics["E1_redetection"]["no_confirm"]["per_type"]
r1 = metrics["E1_redetection"]["speed_confirm"]["per_type"]
tt = [t for t in types_sorted if t in r0]
x = np.arange(len(tt))
ax[1].bar(x - 0.2, [r0[t]["same_category"] for t in tt], 0.4, color=LIGHT, label="ax transition only")
ax[1].bar(x + 0.2, [r1[t]["same_category"] for t in tt], 0.4, color=C1, label="+ speed confirmation")
ax[1].set_xticks(x); ax[1].set_xticklabels(tt, rotation=45, ha="right")
ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("re-detection rate")
ax[1].set_title("(b) Re-detection inside own window", loc="left"); ax[1].legend(frameon=False, fontsize=7)
fig.savefig(FIGS / "fig_e1_keyframes.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# E2 quality filter on GT labels
# --------------------------------------------------------------------------- #
qf = QualityFilter()
by_clip = collections.defaultdict(list)
for w in wins:
    by_clip[w.clip_id].append(w)
reports, per_type = [], collections.defaultdict(list)
code_examples = collections.defaultdict(list)
t0 = time.time()
for w in wins:
    ref = gt[(w.clip_id, w.index)]
    sib = [gt[(o.clip_id, o.index)] for o in by_clip[w.clip_id] if o is not w]
    r = qf.check(ref, w.keyframe_type, w, siblings=sib)
    reports.append(r)
    per_type[w.keyframe_type].append(r.verdict)
    for i in r.issues:
        if len(code_examples[i.code]) < 4:
            code_examples[i.code].append({"type": w.keyframe_type, "message": i.message, "coc": ref, "kin": {k: round(v, 2) for k, v in r.kinematics.items()}})
metrics["E2_filter_gt"] = summarize_reports(reports)
metrics["E2_filter_gt"]["seconds"] = round(time.time() - t0, 3)
metrics["E2_filter_gt"]["per_type"] = {t: dict(collections.Counter(v)) for t, v in per_type.items()}
metrics["E2_filter_gt"]["decision_coverage"] = 1 - sum(1 for r in reports if not (r.decisions["longitudinal"] or r.decisions["lateral"])) / len(reports)
metrics["E2_filter_gt"]["gt_primary_lon_by_type"] = {}
tmp = collections.defaultdict(collections.Counter)
for w, r in zip(wins, reports):
    tmp[w.keyframe_type][(r.decisions["longitudinal"] or ["-"])[0]] += 1
metrics["E2_filter_gt"]["gt_primary_lon_by_type"] = {t: dict(c.most_common(4)) for t, c in tmp.items()}
tmp = collections.defaultdict(collections.Counter)
for w, r in zip(wins, reports):
    tmp[w.keyframe_type][(r.decisions["lateral"] or ["-"])[0]] += 1
metrics["E2_filter_gt"]["gt_primary_lat_by_type"] = {t: dict(c.most_common(4)) for t, c in tmp.items()}
examples["filter_gt"] = dict(code_examples)

codes = metrics["E2_filter_gt"]["issue_codes"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
v = metrics["E2_filter_gt"]["verdicts"]
ax[0].bar(["pass", "review", "reject"], [v.get("pass", 0), v.get("review", 0), v.get("reject", 0)], color=[C1, C2, C4])
for i, k in enumerate(["pass", "review", "reject"]):
    ax[0].text(i, v.get(k, 0) + 5, str(v.get(k, 0)), ha="center", fontsize=8)
ax[0].set_title("(a) Verdicts on 724 Gemini labels", loc="left"); ax[0].set_ylabel("labels")
ck = list(codes.keys())
ERR = {"S01", "S03", "S04", "S05", "S08", "S14", "T01", "K01"}
INFO = {"S06", "K07"}
col = [C4 if c in ERR else (LIGHT if c in INFO else C2) for c in ck]
ax[1].bar(ck, [codes[c] for c in ck], color=col)
ax[1].set_title("(b) Issue codes: dark = error, grey = warning, light = info", loc="left"); ax[1].tick_params(axis="x", rotation=60)
fig.savefig(FIGS / "fig_e2_filter.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# E3 rule-based baseline vs GT
# --------------------------------------------------------------------------- #
be = RuleBasedBackend()
pairs, types, rb_labels = [], [], []
t0 = time.time()
for w in wins:
    lab = be.label(w)
    rb_labels.append(lab)
    pairs.append((lab.coc, gt[(w.clip_id, w.index)]))
    types.append(w.keyframe_type)
e3 = evaluate_pairs(pairs, by=types)
e3["seconds_per_label"] = (time.time() - t0) / len(wins)
metrics["E3_rule_based_vs_gt"] = e3
examples["rule_based"] = [{"type": t, "pred": p, "ref": r} for (p, r), t in list(zip(pairs, types))[:6]]

fig, ax = plt.subplots(figsize=(7.2, 2.4))
tt = [t for t in types_sorted if t in e3["by_group"]]
x = np.arange(len(tt))
ax.bar(x - 0.3, [e3["by_group"][t]["decision_f1"] for t in tt], 0.3, color=C1, label="decision F1")
ax.bar(x, [e3["by_group"][t]["lon_match"] for t in tt], 0.3, color=C2, label="longitudinal match")
ax.bar(x + 0.3, [e3["by_group"][t]["lat_match"] for t in tt], 0.3, color=LIGHT, label="lateral match")
ax.set_xticks(x); ax.set_xticklabels([f"{t} (n={e3['by_group'][t]['n']})" for t in tt], fontsize=7, rotation=25, ha="right")
ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7, ncol=3); ax.set_title("Motion-only baseline vs Gemini labels, by keyframe type", loc="left")
fig.savefig(FIGS / "fig_e3_baseline.png"); plt.close(fig)

# --------------------------------------------------------------------------- #
# E5 filter on rule-based labels (sanity)
# --------------------------------------------------------------------------- #
reps = [qf.check(l.coc, w.keyframe_type, w) for l, w in zip(rb_labels, wins)]
metrics["E5_filter_rule_based"] = summarize_reports(reps)

# --------------------------------------------------------------------------- #
# E4 video ingestion
# (a) real phone video: motion estimate saved by `autolabel run` (video itself is not shipped)
# (b) synthetic video with known motion phases, generated on the fly
# --------------------------------------------------------------------------- #
from autolabel.schemas import EgoMotion  # noqa: E402

demo_json = SAMPLES / "demo_video_motion.json"
if demo_json.exists():
    ego = EgoMotion.from_dict(json.loads(demo_json.read_text()))
    kfs = detect_keyframes(ego)
    metrics["E4_video_real"] = {"file": "IMG_1416.mov (1080x1920 @30fps, 18.7 s)", "motion_seconds_measured": 4.6, "samples": ego.n, "keyframes": [k.to_dict() for k in kfs]}
    t = ego.timestamps_us / 1e6
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True)
    ax[0].plot(t, ego.speed, color=C1); ax[0].set_ylabel("speed proxy")
    ax[1].plot(t, ego.curvatures, color=C2); ax[1].set_ylabel("yaw proxy"); ax[1].set_xlabel("time (s)")
    for k in kfs:
        for a in ax:
            a.axvline(k.timestamp_us / 1e6, color=C3, ls="--", lw=0.8)
        ax[0].text(k.timestamp_us / 1e6 + 0.1, ax[0].get_ylim()[1] * 0.85, k.type, color=C3, fontsize=7)
    ax[0].set_title("(a) Phone video (18.7 s, 1080x1920): optical-flow ego-motion proxies", loc="left")
    fig.savefig(FIGS / "fig_e4_video_motion.png"); plt.close(fig)

sys.path.insert(0, str(ROOT / "tests"))
from test_pipeline import make_synthetic_video  # noqa: E402
from autolabel.pipeline import Project  # noqa: E402
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    vp = Path(td) / "synthetic.mp4"
    make_synthetic_video(vp, seconds=24.0, fps=20)
    proj = Project.create(Path(td) / "proj", fallback_uniform_s=8.0)
    clip = proj.add_video(vp)
    t0 = time.time(); ego = proj.run_motion()[clip.clip_id]; t_motion = time.time() - t0
    t0 = time.time(); wins_s = proj.run_keyframes(); t_kf = time.time() - t0
    t0 = time.time(); labs = proj.run_label("rule_based"); t_lab = time.time() - t0
    exp = proj.export("ds")
    metrics["E4_video_synthetic"] = {
        "duration_s": clip.duration_s, "fps": clip.fps, "size": [clip.width, clip.height],
        "motion_seconds": round(t_motion, 2), "keyframe_and_frames_seconds": round(t_kf, 2), "label_seconds": round(t_lab, 3),
        "realtime_factor": round(clip.duration_s / t_motion, 1), "samples": ego.n,
        "keyframes": [w.keyframe.to_dict() for w in wins_s], "verdicts": dict(collections.Counter(l.filter["verdict"] for l in labs)), "export": exp,
    }
    t = ego.timestamps_us / 1e6
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 3.0), sharex=True)
    ax[0].plot(t, ego.speed, color=C1); ax[0].set_ylabel("speed proxy")
    ax[1].plot(t, ego.curvatures, color=C2); ax[1].set_ylabel("yaw proxy"); ax[1].set_xlabel("time (s)")
    for ph, (a_, b_) in {"forward": (0, 5), "stopped": (5, 8), "forward+turn": (8, 24)}.items():
        ax[0].axvspan(a_, b_, color="#f3f4f6" if ph != "stopped" else "#e5e7eb", zorder=0)
        ax[0].text((a_ + b_) / 2, ax[0].get_ylim()[1] * 0.92 if ax[0].get_ylim()[1] > 0 else 0.5, ph, ha="center", fontsize=7, color=GREY if (GREY := "#6b7280") else GREY)
    for j, w in enumerate(wins_s):
        for a in ax:
            a.axvline(w.keyframe.timestamp_us / 1e6, color=C3, ls="--", lw=0.8)
        lo, hi = ax[1].get_ylim()
        ax[1].text(w.keyframe.timestamp_us / 1e6 + 0.1, lo + (hi - lo) * (0.08 + 0.18 * (j % 3)), w.keyframe.type, color=C3, fontsize=7)
    ax[0].set_title("(b) Synthetic video with known phases (forward 0-5 s, stopped 5-8 s, forward + turn 8-24 s)", loc="left")
    fig.savefig(FIGS / "fig_e4_synthetic.png"); plt.close(fig)

# Extra figure: example of a kinematic check catching a wrong label
# --------------------------------------------------------------------------- #
k04 = [(w, r) for w, r in zip(wins, reports) if "K04" in r.codes]
if k04:
    w, r = k04[0]
    p = w.ego.positions[:, :2] - w.ego.positions[0, :2]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    ax[0].plot(p[:, 0], p[:, 1], color=C1); ax[0].scatter(p[20, 0], p[20, 1], color=C3, zorder=3, label="keyframe (t=2s)")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].legend(frameon=False, fontsize=7)
    ax[0].set_title(f"(a) Ego path, heading change {r.kinematics['heading_change_deg']:+.0f} deg", loc="left")
    tt_ = (w.ego.timestamps_us - w.ego.timestamps_us[0]) / 1e6
    ax[1].plot(tt_, w.ego.speed, color=C1); ax[1].axvline(2.0, color=C3, ls="--", lw=0.8); ax[1].set_xlabel("time in window (s)"); ax[1].set_ylabel("speed (m/s)")
    ax[1].set_title("(b) Speed", loc="left")
    fig.suptitle(f"Label: \"{gt[(w.clip_id, w.index)][:95]}...\"  ->  K04", fontsize=7, y=1.02)
    fig.savefig(FIGS / "fig_k04_example.png"); plt.close(fig)
    examples["k04_case"] = {"clip_id": w.clip_id, "index": w.index, "coc": gt[(w.clip_id, w.index)], "kinematics": r.kinematics}

(RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
(RESULTS / "filter_examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False))
print(json.dumps({k: (v if k != "E2_filter_gt" else {kk: vv for kk, vv in v.items() if kk in ("verdicts", "issue_codes", "decision_coverage")}) for k, v in metrics.items() if k in ("E1_redetection", "E1_unsupported_by_speed", "E2_filter_gt", "E5_filter_rule_based", "E4_video")}, indent=1, default=str)[:4000])
print("E3 overall:", {k: round(v, 3) if isinstance(v, float) else v for k, v in e3["overall"].items()})
print("figures:", sorted(p.name for p in FIGS.glob("*.png")))
