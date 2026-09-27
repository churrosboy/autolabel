"""Quality filter for generated CoC labels.

Three families of checks:

* **S** structural / linguistic  - length, language, vocabulary, causal form,
  hedging, meta-text leakage, degenerate repetition, duplicates.
* **T** keyframe-type consistency - the decision must be compatible with the
  ego-motion event that triggered the window.
* **K** kinematic consistency    - the decision must be compatible with what
  the ego actually did in the 6 s after the keyframe (stop / turn / lane
  change / speed trend).

Each issue has a severity.  ``error`` -> reject, ``warning`` -> review,
otherwise pass.  ``score`` is a 0..1 cleanliness number for ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .motion import summarize_window
from .schemas import FilterReport, Issue, Window
from .vocab import (
    ACCEL_WORDS,
    CAUSAL_CONNECTIVES,
    CONTRADICTORY_PAIRS,
    DECEL_WORDS,
    FUTURE_EVIDENCE,
    HEDGE_WORDS,
    KEYFRAME_CATEGORY,
    KEYFRAME_CONTRADICTS,
    KEYFRAME_EXPECTED,
    META_LEAK,
    STOP_WORDS,
    extract_components,
    extract_decisions,
)

WEIGHTS = {"error": 0.4, "warning": 0.15, "info": 0.03}


@dataclass
class FilterConfig:
    min_chars: int = 25
    max_chars: int = 500
    max_non_ascii_ratio: float = 0.05
    require_component: bool = True
    require_causal: bool = True
    # kinematic thresholds (sensor units)
    stop_speed_mps: float = 1.0
    lane_change_min_m: float = 1.0
    turn_min_deg: float = 15.0
    lane_keep_max_deg: float = 45.0
    speed_trend_mps: float = 1.5
    # kinematic thresholds (video / robust units)
    video_stop_speed: float = 0.2
    video_turn_min_deg: float = 10.0
    video_lane_keep_max_deg: float = 40.0
    disabled: set[str] = field(default_factory=set)


_REPEAT = re.compile(r"\b(\w+(?:\s+\w+){1,3})\b(?:\s+\1\b){2,}", re.I)


class QualityFilter:
    def __init__(self, cfg: Optional[FilterConfig] = None):
        self.cfg = cfg or FilterConfig()

    # ------------------------------------------------------------------ #
    def check(
        self,
        coc: str,
        keyframe_type: Optional[str] = None,
        window: Optional[Window] = None,
        siblings: Optional[Iterable[str]] = None,
        head: Optional[dict] = None,
    ) -> FilterReport:
        """``head``: optional {"longitudinal": <decision|none>, "lateral": ...} predicted by the
        fine-tuned model's decision head; disagreement with the sentence is flagged (H01)."""
        issues: list[Issue] = []
        text = (coc or "").strip()
        add = lambda code, sev, msg: issues.append(Issue(code, sev, msg))  # noqa: E731

        # ---------------- S: structure ----------------
        if len(text) < self.cfg.min_chars:
            add("S01", "error", f"label too short ({len(text)} chars)")
        if len(text) > self.cfg.max_chars:
            add("S02", "warning", f"label too long ({len(text)} chars)")
        if "INSUFFICIENT_EVIDENCE" in text.upper():
            add("S03", "error", "model reported insufficient evidence")
        if text:
            non_ascii = sum(1 for ch in text if ord(ch) > 127) / len(text)
            if non_ascii > self.cfg.max_non_ascii_ratio:
                add("S04", "error", f"non-English characters ({non_ascii:.0%})")
        decisions = extract_decisions(text)
        n_lon, n_lat = len(decisions["longitudinal"]), len(decisions["lateral"])
        if n_lon + n_lat == 0 and text:
            add("S05", "error", "no Table-1 driving decision found")
        if n_lon > 2 or n_lat > 2:
            add("S06", "warning", f"too many decisions per category (lon={n_lon}, lat={n_lat})")
        elif n_lon > 1 or n_lat > 1:
            add("S06", "info", f"more than one decision per category (lon={n_lon}, lat={n_lat})")
        if self.cfg.require_causal and text and not CAUSAL_CONNECTIVES.search(text):
            add("S07", "warning", "no causal connective (because / due to / to ...)")
        flat = decisions["longitudinal"] + decisions["lateral"]
        for a, b in CONTRADICTORY_PAIRS:
            if a in flat and b in flat:
                add("S08", "error", f"contradictory decisions: '{a}' vs '{b}'")
        if HEDGE_WORDS.search(text):
            add("S09", "warning", f"hedging language: '{HEDGE_WORDS.search(text).group(0)}'")
        if FUTURE_EVIDENCE.search(text):
            add("S10", "warning", "cites future frames as evidence (causal locality)")
        if META_LEAK.search(text):
            add("S11", "warning", f"prompt/meta text leaked: '{META_LEAK.search(text).group(0)[:30]}'")
        components = extract_components(text)
        if self.cfg.require_component and text and not components:
            add("S12", "warning", "no Table-2 critical component mentioned")
        if siblings is not None:
            norm = re.sub(r"\W+", " ", text.lower()).strip()
            for s in siblings:
                if s and re.sub(r"\W+", " ", s.lower()).strip() == norm:
                    add("S13", "warning", "duplicate of another label in the same clip")
                    break
        if _REPEAT.search(text):
            add("S14", "error", "degenerate repetition")

        # ---------------- T: keyframe-type consistency ----------------
        if keyframe_type and keyframe_type in KEYFRAME_CATEGORY and flat:
            bad = KEYFRAME_CONTRADICTS.get(keyframe_type, set()) & set(flat)
            if bad:
                add("T01", "error", f"decision {sorted(bad)} contradicts keyframe type '{keyframe_type}'")
            cat = KEYFRAME_CATEGORY[keyframe_type]
            same_cat = decisions[cat]
            if not same_cat:
                add("T02", "warning", f"no {cat} decision although keyframe is {cat} ('{keyframe_type}')")
            elif not (set(same_cat) & KEYFRAME_EXPECTED.get(keyframe_type, set())):
                add("T03", "warning", f"{cat} decision {same_cat} unusual for keyframe '{keyframe_type}'")
            if keyframe_type in ("gentle_decel", "strong_decel", "stop") and ACCEL_WORDS.search(text) and not DECEL_WORDS.search(text):
                add("T04", "warning", "describes acceleration for a deceleration keyframe")
            if keyframe_type in ("gentle_accel", "strong_accel") and STOP_WORDS.search(text) and not ACCEL_WORDS.search(text):
                add("T05", "warning", "describes stopping for an acceleration keyframe")

        # ---------------- K: kinematic consistency ----------------
        kin: dict[str, float] = {}
        if window is not None and window.ego.n > 5 and text:
            kin = summarize_window(window.ego, window.keyframe.timestamp_us)
            if kin:
                video = window.ego.source == "video"
                c = self.cfg
                stop_thr = c.video_stop_speed if video else c.stop_speed_mps
                turn_thr = c.video_turn_min_deg if video else c.turn_min_deg
                keep_thr = c.video_lane_keep_max_deg if video else c.lane_keep_max_deg
                hc = abs(kin["heading_change_deg"])
                sev = "warning" if video else "error"
                decel_evident = kin.get("speed_drop", 0.0) > (0.3 if video else 1.5) or kin["ax_min_future"] < (-1.0 if video else -1.0)
                if "stop for static constraints" in flat and kin["speed_min_future"] > stop_thr:
                    if decel_evident:
                        add("K01", "info", f"claims stop; ego is decelerating but has not stopped within the window (vmin={kin['speed_min_future']:.1f})")
                    else:
                        add("K01", sev, f"claims stop but ego neither stops nor decelerates (vmin={kin['speed_min_future']:.1f}, drop={kin.get('speed_drop', 0):.1f})")
                if (not video) and "lane change" in flat and abs(kin["lateral_offset_m"]) < c.lane_change_min_m and hc < 10:
                    add("K02", "warning", f"claims lane change but lateral offset = {kin['lateral_offset_m']:.2f} m")
                if "turn" in flat and hc < turn_thr:
                    add("K03", "warning", f"claims turn but heading change = {hc:.0f} deg")
                curve_words = re.search(r"curv|bend|winding|corner|road geometry", text, re.I)
                if "lane keeping & centering" in flat and hc > keep_thr and "turn" not in flat and not curve_words and "speed adaptation" not in flat:
                    add("K04", "warning", f"claims lane keeping but heading change = {hc:.0f} deg")
                if not video:
                    d_speed = kin["speed_end"] - kin["speed_before"]
                    never_slowed = kin["speed_min_future"] > max(kin["speed_before"] - 0.5, c.stop_speed_mps) and kin["speed_before"] > 2.0
                    if DECEL_WORDS.search(text) and not ACCEL_WORDS.search(text) and d_speed > c.speed_trend_mps and never_slowed:
                        add("K05", "warning", f"describes slowing but speed only rose ({d_speed:+.1f} m/s)")
                    if (ACCEL_WORDS.search(text) or "acceleration for passing" in flat) and not DECEL_WORDS.search(text) and d_speed < -c.speed_trend_mps and kin["speed_max_future"] < kin["speed_before"] + 0.5:
                        add("K06", "warning", f"describes accelerating but speed only fell ({d_speed:+.1f} m/s)")
                    if STOP_WORDS.search(text) and "stop for static constraints" not in flat and "yield" not in flat and kin["speed_min_future"] > 3.0 and not decel_evident:
                        add("K07", "info", f"mentions stopping but ego never slowed below {kin['speed_min_future']:.1f} m/s")

        # ---------------- H: decision-head consistency ----------------
        if head and text:
            for cat in ("longitudinal", "lateral"):
                hp = head.get(cat)
                if hp is None:
                    continue
                txt = decisions[cat][0] if decisions[cat] else "none"
                if hp != txt:
                    add("H01", "warning", f"{cat}: head predicts '{hp}' but sentence says '{txt}'")

        issues = [i for i in issues if i.code not in self.cfg.disabled]
        score = max(0.0, 1.0 - sum(WEIGHTS[i.severity] for i in issues))
        if any(i.severity == "error" for i in issues):
            verdict = "reject"
        elif any(i.severity == "warning" for i in issues):
            verdict = "review"
        else:
            verdict = "pass"
        return FilterReport(verdict=verdict, score=score, issues=issues, decisions=decisions, components=components, kinematics=kin)


def summarize_reports(reports: Iterable[FilterReport]) -> dict:
    from collections import Counter

    verdicts, codes = Counter(), Counter()
    n = 0
    for r in reports:
        n += 1
        verdicts[r.verdict] += 1
        for c in set(r.codes):
            codes[c] += 1
    return {"n": n, "verdicts": dict(verdicts), "issue_codes": dict(codes.most_common())}
