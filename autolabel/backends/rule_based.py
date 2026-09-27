"""Motion-only rule-based labeler.

It never looks at pixels, so it cannot name the *cause* (a red light, a
pedestrian...).  It exists for three reasons:

1. an offline test double so the whole pipeline runs on a laptop,
2. a floor baseline for the evaluation section (what you get from kinematics
   alone),
3. a "physics prior": its decision is what the quality filter compares VLM
   labels against.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..motion import summarize_window
from ..schemas import Window
from .base import LabelBackend


def infer_decisions(window: Window) -> tuple[Optional[str], Optional[str], dict[str, float]]:
    """Return (longitudinal, lateral, kinematics) inferred from the ego-motion."""
    ego = window.ego
    k = summarize_window(ego, window.keyframe.timestamp_us)
    if not k:
        return None, None, {}
    video = ego.source == "video"
    kt = window.keyframe_type

    # ---- longitudinal -------------------------------------------------------
    stop_thr = 0.15 if video else 0.5
    slow_thr = 0.35 if video else 1.0
    lon: Optional[str]
    if k["speed_min_future"] < stop_thr and k["stopped_frac_future"] > 0.1:
        lon = "stop for static constraints"
    elif k["speed_end"] < k["speed_before"] - slow_thr or kt in ("gentle_decel", "strong_decel"):
        lon = "speed adaptation" if (k["curv_abs_max_future"] > (1.0 if video else 0.02)) else "lead obstacle following"
    elif k["speed_end"] > k["speed_before"] + slow_thr or kt in ("gentle_accel", "strong_accel"):
        lon = "set speed tracking"
    elif k["speed_before"] > (0.3 if video else 3.0):
        lon = "set speed tracking"
    else:
        lon = None
    # lateral keyframes: only report a longitudinal decision if speed really changed
    if kt.startswith(("steer", "sharp_steer", "go_straight")) and lon not in ("stop for static constraints",):
        d_speed = k["speed_end"] - k["speed_before"]
        if abs(d_speed) < (0.5 if video else 2.0):
            lon = None if kt != "go_straight" else "set speed tracking"

    # ---- lateral ------------------------------------------------------------
    hc = abs(k["heading_change_deg"])
    lat: Optional[str]
    if hc > (25.0 if video else 35.0):
        lat = "turn"
    elif (not video) and abs(k["lateral_offset_m"]) > 2.0 and hc < 20.0:
        lat = "lane change"
    elif kt.startswith(("steer", "sharp_steer")) and hc > (12.0 if video else 15.0):
        lat = "turn"
    else:
        lat = "lane keeping & centering"
    return lon, lat, k


class RuleBasedBackend(LabelBackend):
    name = "rule_based"
    needs_frames = False

    def generate(self, window: Window, frames: Optional[list[np.ndarray]] = None) -> str:
        lon, lat, k = infer_decisions(window)
        if lon is None and lat is None:
            return "FINAL_COC: INSUFFICIENT_EVIDENCE"
        kt = window.keyframe_type.replace("_", " ")
        parts = [d for d in (lon, lat) if d]
        decision = " and ".join(parts)
        video = window.ego.source == "video"
        if video:
            evidence = f"a {kt} event in the estimated ego-motion (heading change {k.get('heading_change_deg', 0):+.0f} deg)"
        else:
            evidence = (
                f"a {kt} event (speed {k.get('speed_before', 0):.1f} to {k.get('speed_end', 0):.1f} m/s, "
                f"heading change {k.get('heading_change_deg', 0):+.0f} deg)"
            )
        if lon == "stop for static constraints":
            outcome = "come to a controlled stop at the constraint ahead"
        elif lon in ("lead obstacle following", "speed adaptation"):
            outcome = "keep a safe speed for the situation ahead"
        elif lat == "turn":
            outcome = "follow the planned route through the turn"
        elif lat == "lane change":
            outcome = "complete the lane transition safely"
        else:
            outcome = "continue safely along the road"
        raw = (
            "THOUGHT_PROCESS:\n"
            f"- Stage 1: motion-only labeler; no visual components available.\n"
            f"- Stage 2: {decision}.\n"
            f"- Causal Link: {evidence}.\n"
            f"FINAL_COC: The vehicle performs {decision} because {evidence} on the road ahead required this action to {outcome}."
        )
        return raw
