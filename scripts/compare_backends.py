"""Compare label sets (rule-based / zero-shot VLM / fine-tuned VLM) against GT.

    python scripts/compare_backends.py results/eval_*.json [--val runs/x/val_samples.jsonl --project projects/d3d]

Each input is the JSON written by ``autolabel eval --save``.  With ``--val`` and
``--project`` the metrics are recomputed on the validation clips only (the
clips the fine-tuned model never saw), which is the number to report.
Writes results/compare_backends.json and report/figures/fig_backends.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.evaluate import evaluate_pairs  # noqa: E402
from autolabel.filters import QualityFilter  # noqa: E402
from autolabel.sources import read_jsonl  # noqa: E402

KEYS = ["decision_f1", "decision_precision", "decision_recall", "lon_match", "lat_match", "component_jaccard", "rouge_l", "bleu4"]


def val_only(project: Path, labels_file: str, val_clips: set[str]) -> dict:
    gt = {tuple(k.rsplit("#", 1)): v for k, v in json.loads((project / "gt.json").read_text()).items()}
    qf = QualityFilter()
    pairs, types, verdicts = [], [], []
    for l in read_jsonl(project / labels_file):
        if l["clip_id"] not in val_clips:
            continue
        ref = gt.get((l["clip_id"], str(l["index"])))
        if not ref:
            continue
        pairs.append((l["coc"], ref)); types.append(l["keyframe_type"])
        verdicts.append(qf.check(l["coc"], l["keyframe_type"]).verdict)
    res = evaluate_pairs(pairs, by=types)
    res["overall"]["filter_pass_rate"] = sum(v == "pass" for v in verdicts) / max(len(verdicts), 1)
    res["overall"]["filter_reject_rate"] = sum(v == "reject" for v in verdicts) / max(len(verdicts), 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--val", help="val_samples.jsonl from a training run")
    ap.add_argument("--project")
    ap.add_argument("--out", default=str(ROOT / "results" / "compare_backends.json"))
    a = ap.parse_args()

    rows = {}
    val_clips = {s["clip_id"] for s in read_jsonl(a.val)} if a.val else None
    for f in a.evals:
        d = json.loads(Path(f).read_text())
        name = Path(f).stem.replace("eval_", "")
        if val_clips and a.project:
            d = val_only(Path(a.project), d.get("labels_file", "labels.jsonl"), val_clips)
            name += " (val)"
        rows[name] = {k: d["overall"].get(k) for k in KEYS + ["filter_pass_rate", "filter_reject_rate", "n"] if k in d["overall"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"{'backend':28s} " + " ".join(f"{k[:10]:>10s}" for k in KEYS))
    for n, r in rows.items():
        print(f"{n:28s} " + " ".join(f"{(r.get(k) or 0):10.3f}" for k in KEYS))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
        names = list(rows)
        metrics = ["decision_f1", "lon_match", "lat_match", "rouge_l"]
        cols = ["#2f4f7f", "#9aa0a6", "#d1d5db", "#1f2937"]
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            ax.bar([j + i * w for j in range(len(metrics))], [rows[n].get(m) or 0 for m in metrics], w, color=cols[i % len(cols)], label=n)
        ax.set_xticks([j + w * (len(names) - 1) / 2 for j in range(len(metrics))]); ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7); ax.set_title("Auto-labeler backends vs GT labels", loc="left")
        out = ROOT / "report" / "figures" / "fig_backends.png"
        fig.savefig(out, dpi=200, bbox_inches="tight"); print("figure:", out)
    except Exception as exc:  # noqa: BLE001
        print("no figure:", exc)


if __name__ == "__main__":
    main()
