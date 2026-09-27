"""Qwen3-VL + LoRA + decision head."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .data import LAT_CLASSES, LON_CLASSES

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_decision_head(hidden_size: int, dropout: float = 0.1):
    import torch.nn as nn

    class DecisionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.mlp = nn.Sequential(nn.Linear(hidden_size, max(hidden_size // 4, 32)), nn.GELU(), nn.Dropout(dropout))
            self.lon = nn.Linear(max(hidden_size // 4, 32), len(LON_CLASSES))
            self.lat = nn.Linear(max(hidden_size // 4, 32), len(LAT_CLASSES))

        def forward(self, pooled):
            h = self.mlp(self.norm(pooled))
            return self.lon(h), self.lat(h)

    return DecisionHead()


def make_tiny_vlm(base_model: str = "Qwen/Qwen3-VL-2B-Instruct", hidden: int = 64, layers: int = 2):
    """Randomly initialised, shrunken copy of the base architecture (config + processor only are
    downloaded).  Used by ``--tiny`` smoke tests to exercise the whole training loop on a CPU."""
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = AutoConfig.from_pretrained(base_model)
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = hidden, hidden * 2, layers
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, hidden // 4
    vc.hidden_size, vc.intermediate_size, vc.depth, vc.num_heads, vc.out_hidden_size = hidden, hidden * 2, layers, 4, hidden
    if hasattr(vc, "deepstack_visual_indexes"):
        vc.deepstack_visual_indexes = list(range(min(layers, len(vc.deepstack_visual_indexes))))
    return AutoModelForImageTextToText.from_config(cfg)


def text_hidden_size(vlm) -> int:
    cfg = vlm.config
    tc = getattr(cfg, "text_config", None)
    return int(getattr(tc, "hidden_size", None) or getattr(cfg, "hidden_size"))


def language_lora_targets(vlm, names=DEFAULT_TARGETS) -> list[str]:
    """Full names of language-model projections (vision tower and merger excluded)."""
    out = []
    for n, _ in vlm.named_modules():
        leaf = n.split(".")[-1]
        if leaf in names and "visual" not in n and "vision" not in n and "merger" not in n:
            out.append(n)
    return out or list(names)


class AutolabelModel:
    """PEFT-wrapped VLM + decision head + combined loss.

    Kept as a plain object (not nn.Module) so ``.vlm`` retains PEFT's
    save_pretrained / from_pretrained semantics.
    """

    def __init__(self, vlm, head, head_weight: float = 0.5, lon_weights=None, lat_weights=None):
        import torch
        import torch.nn as nn

        self.vlm = vlm
        self.head = head
        self.head_weight = head_weight
        self._ce_lon = nn.CrossEntropyLoss(weight=torch.tensor(lon_weights) if lon_weights else None)
        self._ce_lat = nn.CrossEntropyLoss(weight=torch.tensor(lat_weights) if lat_weights else None)

    # ------------------------------------------------------------------ build / io
    @classmethod
    def from_base(cls, base_model: str, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
                  target_modules: Optional[list[str]] = None, head_weight: float = 0.5, dtype=None,
                  gradient_checkpointing: bool = True, lon_weights=None, lat_weights=None, vlm=None):
        import torch
        from peft import LoraConfig, get_peft_model

        if vlm is None:
            from transformers import AutoModelForImageTextToText

            vlm = AutoModelForImageTextToText.from_pretrained(base_model, dtype=dtype or torch.bfloat16, low_cpu_mem_usage=True)
        if gradient_checkpointing and hasattr(vlm, "gradient_checkpointing_enable"):
            vlm.gradient_checkpointing_enable()
            if hasattr(vlm, "enable_input_require_grads"):
                vlm.enable_input_require_grads()
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                         target_modules=target_modules or language_lora_targets(vlm), task_type="CAUSAL_LM")
        vlm = get_peft_model(vlm, cfg)
        head = build_decision_head(text_hidden_size(vlm)).to(next(vlm.parameters()).device)
        return cls(vlm, head, head_weight, lon_weights, lat_weights)

    @classmethod
    def load(cls, adapter_dir: str | Path, base_model: Optional[str] = None, dtype=None, vlm=None):
        import torch
        from peft import PeftModel

        adapter_dir = Path(adapter_dir)
        meta_p = adapter_dir / "autolabel_head.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        base_model = base_model or meta.get("base_model")
        if vlm is None:
            from transformers import AutoModelForImageTextToText

            vlm = AutoModelForImageTextToText.from_pretrained(base_model, dtype=dtype or torch.bfloat16, low_cpu_mem_usage=True)
        vlm = PeftModel.from_pretrained(vlm, str(adapter_dir))
        head = build_decision_head(text_hidden_size(vlm))
        hp = adapter_dir / "decision_head.pt"
        if hp.exists():
            head.load_state_dict(torch.load(hp, map_location="cpu"))
        head.to(next(vlm.parameters()).device)
        return cls(vlm, head, float(meta.get("head_weight", 0.5)))

    def save(self, out_dir: str | Path, base_model: str, extra: Optional[dict] = None) -> None:
        import torch

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.vlm.save_pretrained(str(out))
        torch.save(self.head.state_dict(), out / "decision_head.pt")
        (out / "autolabel_head.json").write_text(json.dumps({
            "base_model": base_model, "head_weight": self.head_weight,
            "lon_classes": LON_CLASSES, "lat_classes": LAT_CLASSES, **(extra or {})}, indent=2))

    # ------------------------------------------------------------------ run
    def to(self, device):
        self.vlm.to(device); self.head.to(device); self._ce_lon.to(device); self._ce_lat.to(device)
        return self

    def train(self):
        self.vlm.train(); self.head.train()

    def eval(self):
        self.vlm.eval(); self.head.eval()

    def trainable_parameters(self):
        return [p for p in self.vlm.parameters() if p.requires_grad] + list(self.head.parameters())

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def pooled_hidden(self, out, prompt_last_idx):
        import torch

        h = out.hidden_states[-1]
        idx = prompt_last_idx.to(h.device)
        return h[torch.arange(h.shape[0], device=h.device), idx]

    def forward(self, batch: dict) -> dict:
        kwargs = {k: batch[k] for k in ("input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw") if k in batch}
        out = self.vlm(**kwargs, output_hidden_states=True, use_cache=False)
        pooled = self.pooled_hidden(out, batch["prompt_last_idx"]).float()
        lon_logits, lat_logits = self.head(pooled)
        l_lon = self._ce_lon(lon_logits, batch["lon_label"].to(lon_logits.device))
        l_lat = self._ce_lat(lat_logits, batch["lat_label"].to(lat_logits.device))
        loss = out.loss + self.head_weight * (l_lon + l_lat)
        return {"loss": loss, "loss_text": out.loss.detach(), "loss_lon": l_lon.detach(), "loss_lat": l_lat.detach(),
                "lon_pred": lon_logits.argmax(-1).detach(), "lat_pred": lat_logits.argmax(-1).detach()}

    def predict_head(self, batch: dict) -> tuple[list[str], list[str]]:
        """Decision-head prediction for prompt-only batches."""
        import torch

        with torch.no_grad():
            kwargs = {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw") if k in batch}
            out = self.vlm(**kwargs, output_hidden_states=True, use_cache=False)
            lon_logits, lat_logits = self.head(self.pooled_hidden(out, batch["prompt_last_idx"]).float())
        return [LON_CLASSES[i] for i in lon_logits.argmax(-1).tolist()], [LAT_CLASSES[i] for i in lat_logits.argmax(-1).tolist()]

    def generate(self, batch: dict, processor, max_new_tokens: int = 160) -> list[str]:
        """Greedy generation for prompt-only batches; returns decoded new text per row."""
        import torch

        kwargs = {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw") if k in batch}
        with torch.no_grad():
            out = self.vlm.generate(**kwargs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        new = out[:, batch["input_ids"].shape[1]:]
        return [t.strip() for t in processor.tokenizer.batch_decode(new, skip_special_tokens=True)]
