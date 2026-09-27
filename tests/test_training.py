"""Training package: data layer (always) + tiny end-to-end dry run (needs torch/transformers + network for the processor)."""
import json
from pathlib import Path

import pytest

from autolabel.pipeline import Project
from autolabel.training.data import LAT_CLASSES, LON_CLASSES, head_labels, load_samples, split_by_clip, summarize

from test_pipeline import make_synthetic_video

BASE = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("train")
    p = Project.create(root / "proj", fallback_uniform_s=8.0)
    for i in range(2):
        v = root / f"clip{i}.mp4"
        make_synthetic_video(v, seconds=24.0, fps=20)
        p.add_video(v)
    p.run_motion(); p.run_keyframes(); p.run_label("rule_based")
    return p


def test_head_labels_map_to_vocab():
    lon, lat = head_labels("The vehicle performs stop for static constraints and lane keeping & centering because the light is red.")
    assert LON_CLASSES[lon] == "stop for static constraints" and LAT_CLASSES[lat] == "lane keeping & centering"
    lon, lat = head_labels("The vehicle performs a turn because of routing intent.")
    assert LON_CLASSES[lon] == "none" and LAT_CLASSES[lat] == "turn"


def test_samples_from_project(synthetic_project):
    samples = load_samples(synthetic_project.root)
    assert samples, "expected samples with frames"
    s = samples[0]
    assert len(s.frames) == 16 and Path(s.frames[0]).exists()
    assert s.target.startswith("FINAL_COC: ") and "FINAL_COC" in s.prompt and "THOUGHT_PROCESS" not in s.prompt
    assert 0 <= s.lon_label < len(LON_CLASSES) and 0 <= s.lat_label < len(LAT_CLASSES)
    tr, va = split_by_clip(samples, val_frac=0.5, seed=0)
    assert tr and va and not ({x.clip_id for x in tr} & {x.clip_id for x in va})
    summ = summarize(samples)
    assert summ["n"] == len(samples) and summ["clips"] == 2


def _deps_available():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(BASE)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _deps_available(), reason="torch/transformers/peft or processor download unavailable")
def test_tiny_train_and_finetuned_backend(synthetic_project, tmp_path):
    from autolabel.backends.qwen import QwenLoRABackend
    from autolabel.filters import QualityFilter
    from autolabel.training.train import build_parser, train

    out = tmp_path / "run"
    args = build_parser().parse_args([
        "--data", str(synthetic_project.root), "--out", str(out), "--base-model", BASE, "--tiny",
        "--epochs", "1", "--batch-size", "2", "--grad-accum", "1", "--max-steps", "2", "--val-frac", "0.5",
        "--eval-limit", "1", "--max-new-tokens", "6", "--image-max-side", "128", "--max-pixels", str(64 * 28 * 28),
        "--min-pixels", str(16 * 28 * 28), "--device", "cpu", "--no-grad-ckpt", "--log-every", "1",
    ])
    metrics = train(args)
    assert (out / "adapter_config.json").exists() and (out / "decision_head.pt").exists() and (out / "autolabel_head.json").exists()
    assert metrics["updates"] >= 1 and metrics["history"], metrics
    h = metrics["history"][-1]
    for k in ("decision_f1", "head_lon_acc", "head_lat_acc", "filter_pass_rate"):
        assert k in h
    assert (out / "best").exists()
    logs = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert any(l["stage"] == "train" for l in logs) and any(l["stage"] == "eval" for l in logs)

    # fine-tuned backend: adapter + head are loaded, head decisions land in label.meta, filter sees them
    be = QwenLoRABackend(base_model=BASE, adapter_path=str(out / "best"), device="cpu", max_new_tokens=6, image_max_side=128,
                         min_pixels=16 * 28 * 28, max_pixels=64 * 28 * 28, tiny=True)
    w = synthetic_project.windows()[0]
    lab = be.label(w, synthetic_project.window_frames(w))
    assert lab.backend == "qwen+lora"
    hd = lab.meta["head_decisions"]
    assert hd["longitudinal"] in LON_CLASSES and hd["lateral"] in LAT_CLASSES
    rep = QualityFilter().check(lab.coc or "x", w.keyframe_type, w, head=hd)
    assert rep.verdict in ("pass", "review", "reject")
