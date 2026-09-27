"""Backend registry."""
from __future__ import annotations

from typing import Any

from .base import LabelBackend
from .rule_based import RuleBasedBackend

BACKENDS = {
    "rule_based": "autolabel.backends.rule_based:RuleBasedBackend",
    "gemini": "autolabel.backends.gemini:GeminiBackend",
    "qwen": "autolabel.backends.qwen:QwenLoRABackend",
    "openai": "autolabel.backends.openai_compat:OpenAICompatBackend",
}


def get_backend(name: str, **kwargs: Any) -> LabelBackend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend '{name}', choose from {sorted(BACKENDS)}")
    module_name, cls_name = BACKENDS[name].split(":")
    import importlib

    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)


__all__ = ["LabelBackend", "RuleBasedBackend", "get_backend", "BACKENDS"]
