"""Hand-drawn (matplotlib) diagrams for the report: system architecture and window layout."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGS = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "savefig.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, RED, GREY = "#1f2937", "#1f2937", "#1f2937", "#b45309", "#6b7280"   # monochrome boxes, one accent
NAVY = "#2f4f7f"


def box(ax, x, y, w, h, title, sub="", color=BLUE, fc="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", ec=color, fc=fc, lw=1.4))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#111827")
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=6.5, color=GREY)


def arrow(ax, x0, y0, x1, y1, color="#374151"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.1))


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    # inputs
    box(ax, 0.2, 3.6, 1.7, 1.0, "Input", "video (mp4/mov)\nGPS/IMU csv (opt.)", GREY, "#f3f4f6")
    box(ax, 0.2, 2.1, 1.7, 1.0, "Dataset import", "physical_ai_av /\nD3D jsonl", GREY, "#f3f4f6")
    # stages
    box(ax, 2.5, 3.6, 1.6, 1.0, "1 Motion", "optical flow -> speed,\nyaw proxies", BLUE)
    box(ax, 4.5, 3.6, 1.6, 1.0, "2 Keyframes", "ax / curvature\ntransitions + caps", BLUE)
    box(ax, 6.5, 3.6, 1.6, 1.0, "3 Windows", "[-2s, +6s], 16 frames\n@ 2 Hz, sheet.jpg", BLUE)
    box(ax, 6.5, 2.1, 1.6, 1.0, "4 Backend", "rule / Gemini /\nQwen3-VL + LoRA", ORANGE)
    box(ax, 4.5, 2.1, 1.6, 1.0, "5 Filter", "S structural, T type,\nK kinematic", ORANGE)
    box(ax, 2.5, 2.1, 1.6, 1.0, "6 Review UI", "approve / edit /\nreject in browser", GREEN)
    box(ax, 2.5, 0.5, 1.6, 1.0, "7 Export", "coc_results.jsonl\nsamples.jsonl + frames", GREEN)
    box(ax, 4.5, 0.5, 1.6, 1.0, "8 Evaluate", "decision-F1,\nROUGE-L, BLEU", GREEN)
    box(ax, 6.5, 0.5, 1.6, 1.0, "9 Train", "Qwen3-VL + LoRA\n+ decision head", RED, "#fff7ed")
    # arrows
    arrow(ax, 1.9, 4.1, 2.5, 4.1); arrow(ax, 4.1, 4.1, 4.5, 4.1); arrow(ax, 6.1, 4.1, 6.5, 4.1)
    arrow(ax, 7.3, 3.6, 7.3, 3.1)                 # windows -> backend
    arrow(ax, 6.5, 2.6, 6.1, 2.6); arrow(ax, 4.5, 2.6, 4.1, 2.6)
    arrow(ax, 3.3, 2.1, 3.3, 1.5)                 # review -> export
    # imported windows skip stages 1-2 and enter at stage 3
    ax.plot([1.9, 2.2, 2.2, 7.3], [2.6, 2.6, 3.35, 3.35], color="#374151", lw=1.1)
    arrow(ax, 7.3, 3.35, 7.3, 3.6)
    ax.text(4.6, 3.42, "imported windows (skip 1-2)", fontsize=6.3, color=GREY, ha="center")
    arrow(ax, 4.1, 1.0, 4.5, 1.0); arrow(ax, 6.1, 1.0, 6.5, 1.0)
    arrow(ax, 8.1, 1.0, 9.0, 1.0); ax.text(9.05, 1.0, "adapter\nweights", fontsize=6.5, va="center", color=RED)
    arrow(ax, 9.3, 1.4, 9.3, 2.6); arrow(ax, 9.3, 2.6, 8.1, 2.6)
    # workspace
    ax.text(0.2, 1.62, "Project workspace", fontsize=7.5, fontweight="bold", color=GREY)
    ax.text(0.2, 0.2, "clips.jsonl  motion/*.json\nwindows.jsonl  labels.jsonl\nframes/**/*.jpg  exports/", fontsize=6.3, color=GREY, va="bottom", linespacing=1.5)
    fig.savefig(FIGS / "fig_architecture.png"); plt.close(fig)


def window_layout():
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.set_xlim(-2.6, 6.6); ax.set_ylim(0, 2.2); ax.axis("off")
    ax.add_patch(FancyBboxPatch((-2, 0.9), 2, 0.7, boxstyle="square,pad=0", fc="#e5e7eb", ec=GREY))
    ax.add_patch(FancyBboxPatch((0, 0.9), 6, 0.7, boxstyle="square,pad=0", fc="#f9fafb", ec=GREY))
    for i in range(16):
        t = -2 + i * 8 / 15
        ax.plot([t, t], [0.9, 1.6], color="#ffffff", lw=0.8)
        ax.text(t, 0.72, str(i + 1), ha="center", fontsize=6, color=GREY)
    ax.plot([0, 0], [0.6, 1.9], color=RED, lw=1.4, ls="--")
    ax.text(0, 1.95, "keyframe (decision)", ha="center", fontsize=7.5, color=RED)
    ax.text(-1, 1.25, "Stage I  history  (frames 1-4)", ha="center", fontsize=7.5, color="#1f2937")
    ax.text(3, 1.25, "Stage II  outcome  (frames 5-16)", ha="center", fontsize=7.5, color="#1f2937")
    ax.text(-2, 0.35, "-2 s", ha="center", fontsize=7); ax.text(6, 0.35, "+6 s", ha="center", fontsize=7); ax.text(0, 0.35, "0", ha="center", fontsize=7)
    ax.text(-2.5, 0.05, "Ego-motion sampled at 10 Hz over the same window (81 samples) feeds the kinematic checks.", fontsize=6.5, color=GREY)
    fig.savefig(FIGS / "fig_window.png"); plt.close(fig)


def filter_flow():
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    box(ax, 0.2, 1.0, 1.5, 1.0, "CoC text", "from backend", GREY, "#f9fafb")
    box(ax, 2.1, 1.0, 1.8, 1.0, "Extract", "decisions (Table 1)\ncomponents (Table 2)", BLUE)
    box(ax, 4.3, 2.0, 1.7, 0.85, "S checks", "form, vocab, hedging,\ncontradiction, dup", ORANGE)
    box(ax, 4.3, 1.05, 1.7, 0.85, "T checks", "vs keyframe type", ORANGE)
    box(ax, 4.3, 0.1, 1.7, 0.85, "K checks", "vs ego kinematics\n(stop/turn/lane/speed)", ORANGE)
    box(ax, 6.5, 1.0, 1.6, 1.0, "Verdict", "error->reject\nwarning->review", GREEN)
    box(ax, 8.4, 1.0, 1.4, 1.0, "Review UI", "human in the loop", GREEN)
    arrow(ax, 1.7, 1.5, 2.1, 1.5)
    for y in (2.42, 1.47, 0.52):
        arrow(ax, 3.9, 1.5, 4.3, y); arrow(ax, 6.0, y, 6.5, 1.5)
    arrow(ax, 8.1, 1.5, 8.4, 1.5)
    ax.text(2.1, 0.3, "history / intent mentions ignored\n(\"after stopping...\", \"prepare for a turn\")", fontsize=6.3, color=GREY)
    fig.savefig(FIGS / "fig_filter_flow.png"); plt.close(fig)


if __name__ == "__main__":
    architecture(); window_layout(); filter_flow()
    print("diagrams written to", FIGS)
