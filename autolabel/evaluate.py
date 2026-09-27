"""Metrics for comparing generated CoC labels with reference labels.

* decision-level  : precision / recall / F1 over canonical Table-1 decisions,
                    plus exact-match of the primary longitudinal / lateral
                    decision.
* component-level : Jaccard over Table-2 categories.
* text-level      : ROUGE-L F1 and smoothed BLEU-4 (pure python).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Optional

from .vocab import extract_components, extract_decisions

_TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[&/-]")


def tokenize(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


# --------------------------------------------------------------------------- #
def _lcs(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    l = _lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def bleu(pred: str, ref: str, max_n: int = 4) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        pn = Counter(tuple(p[i : i + n]) for i in range(len(p) - n + 1))
        rn = Counter(tuple(r[i : i + n]) for i in range(len(r) - n + 1))
        overlap = sum(min(c, rn[g]) for g, c in pn.items())
        total = max(sum(pn.values()), 0)
        # add-one smoothing (Lin & Och 2004) for n>1
        if n == 1:
            prec = overlap / total if total else 0.0
        else:
            prec = (overlap + 1) / (total + 1)
        logs.append(math.log(prec) if prec > 0 else -1e9)
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(len(p), 1))
    return bp * math.exp(sum(logs) / max_n)


def _prf(pred: set, ref: set) -> tuple[float, float, float]:
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    tp = len(pred & ref)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(ref) if ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def compare(pred: str, ref: str) -> dict[str, float]:
    dp, dr = extract_decisions(pred), extract_decisions(ref)
    sp = set(dp["longitudinal"] + dp["lateral"])
    sr = set(dr["longitudinal"] + dr["lateral"])
    prec, rec, f1 = _prf(sp, sr)
    lon_match = float((dp["longitudinal"][:1] == dr["longitudinal"][:1]))
    lat_match = float((dp["lateral"][:1] == dr["lateral"][:1]))
    cp, cr = set(extract_components(pred)), set(extract_components(ref))
    jacc = len(cp & cr) / len(cp | cr) if (cp | cr) else 1.0
    return {
        "decision_precision": prec,
        "decision_recall": rec,
        "decision_f1": f1,
        "lon_match": lon_match,
        "lat_match": lat_match,
        "any_decision_overlap": float(bool(sp & sr)),
        "component_jaccard": jacc,
        "rouge_l": rouge_l(pred, ref),
        "bleu4": bleu(pred, ref),
    }


def aggregate(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["n"] = len(rows)
    return out


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    import random

    if not values:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        s = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def evaluate_pairs(pairs: Iterable[tuple[str, str]], by: Optional[Iterable[str]] = None) -> dict:
    """pairs: (pred, ref). ``by``: optional group key per pair (e.g. keyframe type)."""
    pairs = list(pairs)
    groups = list(by) if by is not None else [None] * len(pairs)
    rows = [compare(p, r) for p, r in pairs]
    result = {"overall": aggregate(rows)}
    if rows:
        f1s = [r["decision_f1"] for r in rows]
        result["overall"]["decision_f1_ci95"] = list(bootstrap_ci(f1s))
    if by is not None:
        per: dict[str, list] = {}
        for g, r in zip(groups, rows):
            per.setdefault(str(g), []).append(r)
        result["by_group"] = {g: aggregate(rs) for g, rs in sorted(per.items())}
    return result
