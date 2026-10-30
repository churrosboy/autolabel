"""Hand-drawn (matplotlib) diagrams for the report: system architecture, window layout, filter flow."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGS = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "savefig.dpi": 200, "savefig.bbox": "tight"})
DARK, GREY, ACCENT, NAVY = "#1f2937", "#6b7280", "#b45309", "#2f4f7f"


def box(ax, x, y, w, h, title, sub="", color=DARK, fc="#ffffff", tsize=8.5, ssize=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", ec=color, fc=fc, lw=1.4))
    ty = y + h * (0.68 if sub else 0.5)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color="#111827")
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=ssize, color=GREY, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#374151"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.1, shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")
    W, H = 1.9, 1.05
    xs = [2.6, 5.0, 7.4]
    box(ax, 0.2, 4.2, 1.9, H, "Input", "video (mp4/mov)\nGPS/IMU csv (optional)", GREY, "#f3f4f6")
    box(ax, xs[0], 4.2, W, H, "1 Motion", "optical flow ->\nspeed / yaw proxies")
    box(ax, xs[1], 4.2, W, H, "2 Keyframes", "ax / curvature\ntransitions + caps")
    box(ax, xs[2], 4.2, W, H, "3 Windows", "[-2 s, +6 s], 16 frames\n@ 2 Hz + contact sheet")
    box(ax, 0.2, 2.5, 1.9, H, "Dataset import", "physical_ai_av windows\n+ Gemini labels", GREY, "#f3f4f6")
    box(ax, xs[2], 2.5, W, H, "4 Backend", "rule / Gemini /\nQwen3-VL (+LoRA)")
    box(ax, xs[1], 2.5, W, H, "5 Filter", "S structural, T type,\nK kinematic, H head")
    box(ax, xs[0], 2.5, W, H, "6 Review UI", "approve / edit / reject\nin the browser")
    box(ax, xs[0], 0.8, W, H, "7 Export", "coc_results.jsonl\nsamples.jsonl + frames")
    box(ax, xs[1], 0.8, W, H, "8 Evaluate", "decision-F1, Jaccard,\nROUGE-L, BLEU vs GT")
    box(ax, xs[2], 0.8, W, H, "9 Train", "Qwen3-VL + LoRA\n+ decision head", ACCENT, "#fff7ed")
    arrow(ax, 2.1, 4.72, xs[0], 4.72); arrow(ax, xs[0] + W, 4.72, xs[1], 4.72); arrow(ax, xs[1] + W, 4.72, xs[2], 4.72)
    arrow(ax, xs[2] + W / 2, 4.2, xs[2] + W / 2, 3.55)
    arrow(ax, xs[2], 3.02, xs[1] + W, 3.02); arrow(ax, xs[1], 3.02, xs[0] + W, 3.02)
    arrow(ax, xs[0] + W / 2, 2.5, xs[0] + W / 2, 1.85)
    arrow(ax, xs[0] + W, 1.32, xs[1], 1.32); arrow(ax, xs[1] + W, 1.32, xs[2], 1.32)
    # imported windows: run through the free corridor between rows 1 and 2, enter stage 3 from below
    ax.plot([2.1, 2.35, 2.35, xs[2] + 0.35, xs[2] + 0.35], [3.02, 3.02, 3.85, 3.85, 3.85], color="#374151", lw=1.1)
    arrow(ax, xs[2] + 0.35, 3.85, xs[2] + 0.35, 4.2)
    ax.text(4.9, 3.93, "imported windows enter at stage 3", fontsize=6.3, color=GREY, ha="center", va="bottom")
    # adapter loop: train -> backend
    ax.plot([xs[2] + W, 10.3, 10.3], [1.32, 1.32, 3.02], color="#374151", lw=1.1)
    arrow(ax, 10.3, 3.02, xs[2] + W, 3.02)
    ax.text(10.35, 2.2, "LoRA adapter\n+ head", fontsize=6.5, color=ACCENT, va="center", ha="left")
    ax.text(0.2, 1.75, "Project workspace", fontsize=7.5, fontweight="bold", color=GREY)
    ax.text(0.2, 1.5, "clips.jsonl\nmotion/*.json\nwindows.jsonl\nframes/**/*.jpg\nlabels*.jsonl\nexports/", fontsize=6.3, color=GREY, va="top", linespacing=1.35)
    fig.savefig(FIGS / "fig_architecture.png"); plt.close(fig)


def window_layout():
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(-2.7, 6.7); ax.set_ylim(0, 2.6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((-2, 0.9), 2, 0.7, boxstyle="square,pad=0", fc="#e5e7eb", ec=GREY))
    ax.add_patch(FancyBboxPatch((0, 0.9), 6, 0.7, boxstyle="square,pad=0", fc="#f9fafb", ec=GREY))
    for i in range(16):
        t = -2 + i * 8 / 15
        ax.plot([t, t], [0.9, 1.6], color="#ffffff", lw=0.8)
        ax.text(t, 1.25, str(i + 1), ha="center", va="center", fontsize=6.5, color=DARK)
    ax.plot([0, 0], [0.55, 1.95], color=ACCENT, lw=1.4, ls="--")
    ax.text(0, 2.35, "keyframe (decision, t = 0)", ha="center", fontsize=7.5, color=ACCENT)
    ax.text(-0.2, 1.78, "Stage I: history, frames 1-4", ha="right", fontsize=7.2, color=DARK)
    ax.text(0.2, 1.78, "Stage II: outcome, frames 5-16", ha="left", fontsize=7.2, color=DARK)
    ax.text(-2, 0.45, "-2 s", ha="center", fontsize=7, color=GREY); ax.text(6, 0.45, "+6 s", ha="center", fontsize=7, color=GREY)
    ax.text(-2.65, 0.05, "16 frames at 2 Hz; ego-motion at 10 Hz (81 samples) over the same window feeds the kinematic checks.", fontsize=6.5, color=GREY)
    fig.savefig(FIGS / "fig_window.png"); plt.close(fig)


def filter_flow():
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.set_xlim(0, 10.2); ax.set_ylim(0, 3.7); ax.axis("off")
    box(ax, 0.2, 1.35, 1.5, 1.0, "CoC text", "from backend", GREY, "#f9fafb")
    box(ax, 2.0, 1.35, 1.9, 1.0, "Extract", "decisions (Table 1)\ncomponents (Table 2)")
    box(ax, 4.3, 2.55, 2.2, 0.95, "S checks", "form, vocabulary, hedging,\ncontradiction, duplicates", ssize=6.2)
    box(ax, 4.3, 1.38, 2.2, 0.95, "T checks", "vs keyframe type", ssize=6.2)
    box(ax, 4.3, 0.2, 2.2, 0.95, "K / H checks", "vs ego kinematics\nvs decision head", ssize=6.2)
    box(ax, 6.9, 1.35, 1.5, 1.0, "Verdict", "error -> reject\nwarning -> review")
    box(ax, 8.7, 1.35, 1.3, 1.0, "Review UI", "human in\nthe loop")
    arrow(ax, 1.7, 1.85, 2.0, 1.85)
    for y in (3.02, 1.85, 0.68):
        arrow(ax, 3.9, 1.85, 4.3, y); arrow(ax, 6.5, y, 6.9, 1.85)
    arrow(ax, 8.4, 1.85, 8.7, 1.85)
    ax.text(2.05, 1.05, "history / intent mentions are ignored\n(\"after stopping at ...\",\n\"prepare for a turn\")", fontsize=6.2, color=GREY, ha="center", va="top", linespacing=1.3)
    fig.savefig(FIGS / "fig_filter_flow.png"); plt.close(fig)


if __name__ == "__main__":
    architecture(); window_layout(); filter_flow()
    print("diagrams written to", FIGS)
