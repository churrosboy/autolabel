"""Prompt construction and output parsing for VLM backends."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from .schemas import Window
from .vocab import DECISION_DEFINITIONS, LATERAL, LONGITUDINAL

SYSTEM_PROMPT = "You are an expert autonomous driving reasoning analyst."


def trajectory_text(window: Window, n_points: int = 16) -> str:
    ego = window.ego
    if ego.n == 0:
        return "Trajectory data not provided."
    idx = np.linspace(0, ego.n - 1, min(n_points, ego.n)).astype(int)
    if ego.source == "video":
        rows = [f"t={ego.timestamps_us[i]/1e6 - window.start_us/1e6:.1f}s speed={ego.speed[i]:.2f} yaw={ego.curvatures[i]:+.2f}" for i in idx]
        head = "Ego-motion estimated from the video (relative units: speed ~ scene expansion, yaw>0 = turning left):"
    else:
        rows = [f"t={ego.timestamps_us[i]/1e6 - window.start_us/1e6:.1f}s speed={ego.speed[i]:.1f}m/s ax={ego.ax[i]:+.2f} curv={ego.curvatures[i]:+.3f}" for i in idx]
        head = "Ego-vehicle motion synchronised with the frames (t=0 is frame 1, keyframe at t=2s):"
    return head + "\n" + "\n".join(rows)


def build_prompt(window: Window, include_trajectory: bool = True, n_frames: int = 16, history_frames: int = 4, concise: bool = False) -> str:
    """Two-stage CoC prompt. ``concise=True`` asks for the FINAL_COC line only (fine-tuned model)."""
    lon = "\n".join(f"- {k.title() if k != 'gap-searching' else 'Gap-searching'}: {DECISION_DEFINITIONS[k]}" for k in LONGITUDINAL)
    lat = "\n".join(f"- {k.title()}: {DECISION_DEFINITIONS[k]}" for k in LATERAL)
    traj = trajectory_text(window) if include_trajectory else ""
    fut_first = history_frames + 1
    return (
        f"{SYSTEM_PROMPT} Your task is to generate a structured 'Chain-of-Causation (CoC)' trace "
        f"based on the provided dashcam video frames.\n\n"
        f"### INPUT VIDEO STRUCTURE ({n_frames} frames total, sampled at 2Hz)\n"
        f"- HISTORY WINDOW (Stage I): Frames 1-{history_frames} (0-2s PRIOR to decision). Contains observable causal factors.\n"
        f"- FUTURE WINDOW (Stage II): Frames {fut_first}-{n_frames} (2-8s AFTER decision). Shows the outcome and driving decision.\n"
        f"- TARGET META ACTION (from ego-motion): [{window.keyframe_type}]\n\n"
        + (f"### EGO-VEHICLE MOTION\n{traj}\n\n" if traj else "")
        + "### STRICT REASONING PROCESS\n"
        f"1. STAGE I (Analyze History): Observe Frames 1-{history_frames} ONLY. Identify 'Critical Components'. Rank their importance and retain ONLY the factors that directly and inevitably cause the decision.\n"
        f"2. STAGE II (Resolve Decision): Observe Frames {fut_first}-{n_frames}. Identify the exact post-keyframe 'Driving Decision' using the strict definitions below. Select at most ONE Longitudinal and/or ONE Lateral decision.\n"
        f"3. SYNTHESIS: Verify causal locality. ALL evidence used MUST originate strictly from Frames 1-{history_frames}.\n\n"
        "### DEFINITIONS: DRIVING DECISIONS (Table 1)\nChoose ONLY from these exact terms if applicable:\n"
        f"[Longitudinal]:\n{lon}\n[Lateral]:\n{lat}\n\n"
        "### DEFINITIONS: CRITICAL COMPONENTS (Table 2)\nExtract only decision-relevant attributes from these categories:\n"
        "- Critical objects: Type (veh/ped/cyclist), relative pose (in-path, oncoming), motion (stopped, slowing).\n"
        "- Traffic lights: State (R/Y/G), visibility.\n"
        "- Yield/Stop control: Signs, stop/yield line.\n"
        "- Road events: Curvature, speed bump, narrowing.\n"
        "- Lane/lanelines: Lane count, line type.\n"
        "- Routing intent: Target lane/turn (L/R/through).\n"
        "- ODD constraints: Weather, construction, school bus rules.\n\n"
        + ("### OUTPUT FORMAT\nOutput exactly one line:\nFINAL_COC: The vehicle performs <Driving Decision> because <Critical Component(s)> required this action to <Safety or Goal Outcome>.\n"
           f"If the historical evidence (Frames 1-{history_frames}) is completely insufficient, output exactly:\nFINAL_COC: INSUFFICIENT_EVIDENCE\n" if concise else
        "### OUTPUT FORMAT\nGenerate the output EXACTLY in the following structure:\n"
        "THOUGHT_PROCESS:\n"
        f"- Stage 1 (Causal Factors in Frames 1-{history_frames}): <List relevant components & attributes>\n"
        f"- Stage 2 (Driving Decision in Frames {fut_first}-{n_frames}): <Identify explicit Longitudinal/Lateral decision>\n"
        "- Causal Link: <Explain how Stage 1 forced Stage 2>\n"
        "FINAL_COC: The vehicle performs <Driving Decision> because <Critical Component(s)> required this action to <Safety or Goal Outcome>.\n\n"
        f"If the historical evidence (Frames 1-{history_frames}) is completely insufficient, output exactly:\n"
        "FINAL_COC: INSUFFICIENT_EVIDENCE\n")
    )


_FINAL = re.compile(r"FINAL_COC\s*:\s*(.+?)(?:\n\s*\n|\Z)", re.S | re.I)


def parse_final_coc(text: str) -> Optional[str]:
    """Extract the one-sentence CoC from a full model response."""
    if not text:
        return None
    m = _FINAL.search(text)
    if m:
        coc = m.group(1).strip()
    else:
        # fall back: last non-empty line
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        coc = lines[-1] if lines else ""
    coc = re.sub(r"\s+", " ", coc).strip().strip("*").strip()
    coc = re.sub(r"^\[?this is a \[[^\]]*\]\s*scenario\.\s*", "", coc, flags=re.I)
    return coc or None
