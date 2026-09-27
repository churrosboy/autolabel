"""Autolabel: Chain-of-Causation (CoC) auto-labelling for your own dashcam videos.

Stages
------
1. ingest     : register raw video files (mp4/mov) as clips.
2. motion     : estimate ego-motion from the video (optical flow) or load
                sensor ego-motion (GPS/IMU CSV, physical_ai_av JSONL).
3. keyframes  : detect high-level driving decisions from the motion signals.
4. windows    : cut an 8-second window ([-2s, +6s]) with 16 frames @ 2 Hz.
5. backends   : generate a CoC label (rule-based / Gemini / Qwen-LoRA).
6. filters    : reject or flag labels that are malformed, contradictory or
                physically inconsistent with the motion.
7. review     : browser UI to approve / edit / reject labels.
8. export     : write a training-ready dataset (JSONL + frames).
"""

__version__ = "0.3.0"
