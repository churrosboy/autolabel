"""Training loop + evaluation for the LoRA auto-labeler.

    python -m autolabel.training.train --data PROJECT_DIR --out runs/qwen2b-lora \
        --base-model Qwen/Qwen3-VL-2B-Instruct --epochs 3 --batch-size 1 --grad-accum 8

Writes to --out: adapter_model.safetensors (+ config), decision_head.pt,
autolabel_head.json, log.jsonl, metrics.json, best/ (best checkpoint by val
decision-F1), and val_predictions.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

from ..evaluate import aggregate, compare
from ..filters import QualityFilter
from ..prompt import parse_final_coc
from .collate import Collator, to_device
from .data import LAT_CLASSES, LON_CLASSES, TrainSample, class_weights, dump_samples, load_samples, split_by_clip, summarize
from .model import AutolabelModel


def pick_device(name: Optional[str] = None):
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device):
    import torch

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return None


def evaluate_model(model: AutolabelModel, processor, samples: list[TrainSample], device, batch_size: int = 1,
                   max_new_tokens: int = 160, limit: Optional[int] = None, image_max_side: int = 448, keyframe_types=None) -> tuple[dict, list[dict]]:
    """Generate on ``samples`` and score against their GT sentence."""
    import torch

    model.eval()
    gen_collate = Collator(processor, image_max_side=image_max_side, with_target=False)
    subset = samples[:limit] if limit else samples
    qf = QualityFilter()
    rows, metrics_rows = [], []
    head_lon_ok = head_lat_ok = 0
    for i in range(0, len(subset), batch_size):
        chunk = subset[i : i + batch_size]
        batch = to_device(gen_collate(chunk), device)
        with torch.no_grad():
            texts = model.generate(batch, processor, max_new_tokens=max_new_tokens)
            lon_h, lat_h = model.predict_head(batch)
        for s, raw, lh, lth in zip(chunk, texts, lon_h, lat_h):
            coc = parse_final_coc(raw) or ""
            m = compare(coc, s.coc)
            rep = qf.check(coc, s.keyframe_type)
            head_lon_ok += int(lh == LON_CLASSES[s.lon_label])
            head_lat_ok += int(lth == LAT_CLASSES[s.lat_label])
            metrics_rows.append(m)
            rows.append({"clip_id": s.clip_id, "index": s.index, "keyframe_type": s.keyframe_type, "pred": coc, "raw": raw, "ref": s.coc,
                         "head_lon": lh, "head_lat": lth, "gt_lon": LON_CLASSES[s.lon_label], "gt_lat": LAT_CLASSES[s.lat_label],
                         "filter_verdict": rep.verdict, "decision_f1": m["decision_f1"]})
    agg = aggregate(metrics_rows)
    n = max(len(subset), 1)
    agg.update({
        "head_lon_acc": head_lon_ok / n, "head_lat_acc": head_lat_ok / n,
        "filter_pass_rate": sum(r["filter_verdict"] == "pass" for r in rows) / n,
        "filter_reject_rate": sum(r["filter_verdict"] == "reject" for r in rows) / n,
    })
    model.train()
    return agg, rows


def train(args) -> dict:
    import torch
    from transformers import AutoProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    ac_dtype = autocast_dtype(device)
    log_f = open(out / "log.jsonl", "a")

    def log(**kw):
        kw["time"] = round(time.time(), 1)
        log_f.write(json.dumps(kw) + "\n"); log_f.flush()
        print(json.dumps(kw))

    # ---------------- data
    samples = load_samples(args.data, reviewed_only=args.reviewed_only)
    if args.max_samples:
        samples = samples[: args.max_samples]
    train_s, val_s = split_by_clip(samples, args.val_frac, args.seed)
    dump_samples(train_s, out / "train_samples.jsonl"); dump_samples(val_s, out / "val_samples.jsonl")
    log(stage="data", train=summarize(train_s), val=summarize(val_s))
    if not train_s:
        raise SystemExit("no training samples (are frames present in the project?)")
    lon_w, lat_w = class_weights(train_s) if args.class_weights else (None, None)

    # ---------------- model
    processor = AutoProcessor.from_pretrained(args.base_model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model_dtype = torch.float32 if device.type == "cpu" else (torch.bfloat16 if device.type == "cuda" else torch.float16)
    tiny_vlm = None
    if args.tiny:
        from .model import make_tiny_vlm

        tiny_vlm = make_tiny_vlm(args.base_model)
        model_dtype = torch.float32
    model = AutolabelModel.from_base(
        args.base_model, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(",") if args.target_modules else None, head_weight=args.head_weight,
        dtype=model_dtype, gradient_checkpointing=not args.no_grad_ckpt, lon_weights=lon_w, lat_weights=lat_w, vlm=tiny_vlm,
    ).to(device)
    log(stage="model", base=args.base_model, trainable_params=model.n_trainable(), device=str(device))

    collate = Collator(processor, max_length=args.max_length, image_max_side=args.image_max_side)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_s) / args.batch_size)
    total_updates = max(1, math.ceil(steps_per_epoch * args.epochs / args.grad_accum))
    warmup = max(1, int(total_updates * args.warmup_frac))

    def lr_at(u):
        if u < warmup:
            return args.lr * (u + 1) / warmup
        p = (u - warmup) / max(1, total_updates - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    # ---------------- zero-shot eval (optional baseline before training)
    best_f1, history = -1.0, []
    if args.eval_zero_shot and val_s:
        m0, rows0 = evaluate_model(model, processor, val_s, device, args.eval_batch_size, args.max_new_tokens, args.eval_limit, args.image_max_side)
        log(stage="eval", epoch=0, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m0.items()})
        history.append({"epoch": 0, **m0})
        (out / "val_predictions_epoch0.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows0))

    # ---------------- loop
    model.train()
    update, step = 0, 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_s)))
        random.shuffle(order)
        run = {"loss": 0.0, "loss_text": 0.0, "loss_lon": 0.0, "loss_lat": 0.0, "n": 0}
        t_epoch = time.time()
        for bi in range(0, len(order), args.batch_size):
            chunk = [train_s[j] for j in order[bi : bi + args.batch_size]]
            batch = to_device(collate(chunk), device)
            ctx = torch.autocast(device_type=device.type, dtype=ac_dtype) if ac_dtype is not None else torch.autocast(device_type="cpu", enabled=False)
            with ctx:
                res = model.forward(batch)
            (res["loss"] / args.grad_accum).backward()
            step += 1
            for k in ("loss", "loss_text", "loss_lon", "loss_lat"):
                run[k] += float(res[k].detach())
            run["n"] += 1
            if step % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(update)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True)
                update += 1
                if update % args.log_every == 0:
                    log(stage="train", epoch=epoch, update=update, lr=round(lr_at(update), 7),
                        **{k: round(run[k] / max(run["n"], 1), 4) for k in ("loss", "loss_text", "loss_lon", "loss_lat")})
                    run = {"loss": 0.0, "loss_text": 0.0, "loss_lon": 0.0, "loss_lat": 0.0, "n": 0}
            if args.max_steps and step >= args.max_steps:
                break
        if step % args.grad_accum:                      # flush a partial accumulation
            for g in opt.param_groups:
                g["lr"] = lr_at(update)
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
            opt.step(); opt.zero_grad(set_to_none=True); update += 1
        log(stage="epoch_done", epoch=epoch, seconds=round(time.time() - t_epoch, 1), updates=update)
        model.save(out, args.base_model, {"epoch": epoch, "args": vars(args)})
        if val_s and not args.skip_eval:
            m, rows = evaluate_model(model, processor, val_s, device, args.eval_batch_size, args.max_new_tokens, args.eval_limit, args.image_max_side)
            log(stage="eval", epoch=epoch, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
            history.append({"epoch": epoch, **m})
            (out / f"val_predictions_epoch{epoch}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
            if m["decision_f1"] > best_f1:
                best_f1 = m["decision_f1"]
                model.save(out / "best", args.base_model, {"epoch": epoch, "decision_f1": best_f1})
        if args.max_steps and step >= args.max_steps:
            break
    metrics = {"best_val_decision_f1": best_f1, "history": history, "train_n": len(train_s), "val_n": len(val_s), "updates": update, "args": vars(args)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    log(stage="done", best_val_decision_f1=best_f1)
    log_f.close()
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LoRA fine-tune a Qwen3-VL auto-labeler with a decision head")
    ap.add_argument("--data", required=True, help="project dir (windows.jsonl+labels.jsonl/gt.json+frames) or export dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default=None, help="comma list; default = all LM projections")
    ap.add_argument("--head-weight", type=float, default=0.5)
    ap.add_argument("--class-weights", action="store_true", help="inverse-frequency weights for the head losses")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--image-max-side", type=int, default=448)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--eval-limit", type=int, default=None, help="evaluate on the first N val samples only")
    ap.add_argument("--eval-zero-shot", action="store_true", help="evaluate the untrained adapter first (epoch 0)")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after N micro-steps (smoke tests)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (auto)")
    ap.add_argument("--tiny", action="store_true", help="smoke test: random tiny model with the base architecture (no weights download)")
    return ap


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
