"""Command line interface.

    autolabel init  PROJECT [--name N]
    autolabel add   PROJECT VIDEO_OR_FOLDER [--csv sensor.csv] [--copy]
    autolabel run   PROJECT [--backend rule_based|gemini|qwen|openai] [--force] [--limit N]
    autolabel label PROJECT --backend ...            (labels only)
    autolabel filter PROJECT
    autolabel review PROJECT [--port 8765]           (browser UI)
    autolabel export PROJECT [--name dataset] [--reviewed-only]
    autolabel eval  PROJECT [--gt coc_results_pro.jsonl] [--labels labels_qwen.jsonl]
    autolabel import-d3d PROJECT --ego ego_motion_results.jsonl --coc coc_results_pro.jsonl
    autolabel fetch-frames PROJECT [--layout grid|front]       (physical_ai_av, needs HF_TOKEN)
    autolabel train PROJECT --out runs/qwen2b [--base-model ...] (LoRA + decision head)
    autolabel stats PROJECT
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .pipeline import Project


def _p(args) -> Project:
    return Project.open(args.project, labels_file=getattr(args, "labels", None) or "labels.jsonl")


def cmd_init(args):
    p = Project.create(args.project, name=args.name, fallback_uniform_s=args.uniform)
    print(f"created project at {p.root}")


def cmd_add(args):
    p = _p(args)
    src = Path(args.source)
    if src.is_dir():
        clips = p.add_folder(src, copy=args.copy)
    else:
        clips = [p.add_video(src, copy=args.copy, sensor_csv=args.csv)]
    for c in clips:
        print(f"+ {c.clip_id}  {c.duration_s:.1f}s  {c.path}")


def cmd_run(args):
    p = _p(args)
    p.run_motion(force=args.force, progress=print)
    wins = p.run_keyframes(force=args.force, extract_frames=not args.no_frames, progress=print)
    print(f"{len(wins)} windows")
    kw = json.loads(args.backend_args) if args.backend_args else {}
    labels = p.run_label(args.backend, force=args.force, limit=args.limit, progress=print, backend_kwargs=kw)
    print(json.dumps(p.stats(), indent=2, ensure_ascii=False))


def cmd_label(args):
    p = _p(args)
    kw = json.loads(args.backend_args) if args.backend_args else {}
    p.run_label(args.backend, force=args.force, limit=args.limit, progress=print, backend_kwargs=kw)
    print(json.dumps(p.stats(), indent=2, ensure_ascii=False))


def cmd_filter(args):
    p = _p(args)
    print(json.dumps(p.run_filter(), indent=2, ensure_ascii=False))


def cmd_review(args):
    from .ui.server import serve

    serve(_p(args), port=args.port, open_browser=not args.no_browser)


def cmd_export(args):
    p = _p(args)
    print(json.dumps(p.export(args.name, reviewed_only=args.reviewed_only), indent=2))


def cmd_stats(args):
    print(json.dumps(_p(args).stats(), indent=2, ensure_ascii=False))


def cmd_import_d3d(args):
    from .sources import load_d3d

    p = _p(args)
    wins, gt = load_d3d(args.ego, args.coc, limit=args.limit)
    p.add_windows(wins)
    if gt:
        (p.root / "gt.json").write_text(json.dumps({f"{k[0]}#{k[1]}": v for k, v in gt.items()}, ensure_ascii=False, indent=1))
    print(f"imported {len(wins)} windows, {len(gt)} GT labels")


def cmd_fetch_frames(args):
    from .training.fetch_frames import fetch

    print(json.dumps(fetch(args.project, args.layout, args.limit, args.force), indent=2))


def cmd_train(args):
    from .training.train import build_parser as tp, train

    argv = ["--data", args.project, "--out", args.out] + (args.train_args or [])
    train(tp().parse_args(argv))


def cmd_eval(args):
    from .evaluate import evaluate_pairs

    p = _p(args)
    if args.gt:
        gt = {}
        for row in (json.loads(l) for l in open(args.gt) if l.strip()):
            for r in row["coc_results"]:
                gt[(row["clip_id"], int(r["index"]))] = r["coc"]
    else:
        raw = json.loads((p.root / "gt.json").read_text())
        gt = {(k.rsplit("#", 1)[0], int(k.rsplit("#", 1)[1])): v for k, v in raw.items()}
    pairs, types = [], []
    for l in p.labels():
        ref = gt.get((l.clip_id, l.index))
        if ref:
            pairs.append((l.coc, ref))
            types.append(l.keyframe_type)
    res = evaluate_pairs(pairs, by=types)
    res["labels_file"] = p.labels_file
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["overall"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="autolabel", description="CoC auto-labelling for dashcam videos")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init"); s.add_argument("project"); s.add_argument("--name"); s.add_argument("--uniform", type=float, default=0.0, help="fallback: one window every N s when no keyframe found"); s.set_defaults(fn=cmd_init)
    s = sub.add_parser("add"); s.add_argument("project"); s.add_argument("source"); s.add_argument("--csv"); s.add_argument("--copy", action="store_true"); s.set_defaults(fn=cmd_add)
    for name, fn in (("run", cmd_run), ("label", cmd_label)):
        s = sub.add_parser(name); s.add_argument("project"); s.add_argument("--backend", default="rule_based"); s.add_argument("--backend-args", help="JSON kwargs for the backend")
        s.add_argument("--labels", help="labels file name inside the project (default labels.jsonl)")
        s.add_argument("--force", action="store_true"); s.add_argument("--limit", type=int); s.add_argument("--no-frames", action="store_true"); s.set_defaults(fn=fn)
    s = sub.add_parser("filter"); s.add_argument("project"); s.add_argument("--labels"); s.set_defaults(fn=cmd_filter)
    s = sub.add_parser("review"); s.add_argument("project"); s.add_argument("--port", type=int, default=8765); s.add_argument("--no-browser", action="store_true"); s.set_defaults(fn=cmd_review)
    s = sub.add_parser("export"); s.add_argument("project"); s.add_argument("--labels"); s.add_argument("--name", default="dataset"); s.add_argument("--reviewed-only", action="store_true"); s.set_defaults(fn=cmd_export)
    s = sub.add_parser("stats"); s.add_argument("project"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("import-d3d"); s.add_argument("project"); s.add_argument("--ego", required=True); s.add_argument("--coc"); s.add_argument("--limit", type=int); s.set_defaults(fn=cmd_import_d3d)
    s = sub.add_parser("eval"); s.add_argument("project"); s.add_argument("--gt"); s.add_argument("--labels"); s.add_argument("--save", help="write full metrics JSON here"); s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("fetch-frames"); s.add_argument("project"); s.add_argument("--layout", choices=["grid", "front"], default="grid"); s.add_argument("--limit", type=int); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_fetch_frames)
    s = sub.add_parser("train"); s.add_argument("project"); s.add_argument("--out", required=True); s.set_defaults(fn=cmd_train, train_args=[])  # args after "--" go to autolabel.training.train
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = build_parser().parse_args(argv)
    if extra:
        if args.fn is not cmd_train:
            build_parser().error("arguments after -- are only accepted by train")
        args.train_args = extra
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
