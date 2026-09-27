"""TrainSample -> tensors with the Qwen3-VL processor."""
from __future__ import annotations

from typing import Any

from .data import TrainSample


def build_messages(prompt: str, images: list[Any], target: str | None = None) -> list[dict]:
    msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]}]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


def load_images(paths: list[str], max_side: int = 448):
    from PIL import Image

    out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((max_side, max_side))
        out.append(im)
    return out


class Collator:
    """Right-padded batch; prompt tokens are masked (-100) in ``labels``.

    Also returns ``prompt_last_idx`` (index of the last prompt token, used to
    pool the hidden state for the decision head) and the head labels.
    With ``with_target=False`` it builds generation prompts only.
    """

    def __init__(self, processor, max_length: int = 4096, image_max_side: int = 448, with_target: bool = True):
        self.processor = processor
        self.max_length = max_length
        self.image_max_side = image_max_side
        self.with_target = with_target
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = processor.tokenizer.eos_token_id

    def encode_one(self, s: TrainSample) -> dict:
        import torch

        images = load_images(s.frames, self.image_max_side)
        prompt = self.processor.apply_chat_template(build_messages(s.prompt, images), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        p_len = int(prompt["input_ids"].shape[1])
        if self.with_target:
            full = self.processor.apply_chat_template(build_messages(s.prompt, images, s.target), tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
        else:
            full = prompt
        ids = full["input_ids"][0][: self.max_length]
        attn = full["attention_mask"][0][: self.max_length]
        p_len = min(p_len, len(ids))
        labels = ids.clone()
        labels[:p_len] = -100
        return {
            "input_ids": ids, "attention_mask": attn, "labels": labels,
            "pixel_values": full.get("pixel_values"), "image_grid_thw": full.get("image_grid_thw"),
            "prompt_last_idx": torch.tensor(p_len - 1), "lon_label": torch.tensor(s.lon_label), "lat_label": torch.tensor(s.lat_label),
        }

    def __call__(self, samples: list[TrainSample]) -> dict:
        import torch

        enc = [self.encode_one(s) for s in samples]
        L = max(e["input_ids"].shape[0] for e in enc)

        def pad(key, value):
            out = torch.full((len(enc), L), value, dtype=enc[0][key].dtype)
            for i, e in enumerate(enc):
                out[i, : e[key].shape[0]] = e[key]
            return out

        batch = {
            "input_ids": pad("input_ids", self.pad_id), "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100),
            "prompt_last_idx": torch.stack([e["prompt_last_idx"] for e in enc]),
            "lon_label": torch.stack([e["lon_label"] for e in enc]), "lat_label": torch.stack([e["lat_label"] for e in enc]),
        }
        if enc[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.cat([e["pixel_values"] for e in enc], dim=0)
            batch["image_grid_thw"] = torch.cat([e["image_grid_thw"] for e in enc], dim=0)
        return batch


def to_device(batch: dict, device) -> dict:
    import torch

    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
