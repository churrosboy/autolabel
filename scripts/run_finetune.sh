#!/usr/bin/env bash
# End-to-end fine-tuning of the CoC auto-labeler on a GPU machine.
#
#   export HF_TOKEN=hf_...            # gated NVIDIA Physical AI AV dataset (frames)
#   ./scripts/run_finetune.sh [BASE_MODEL] [PROJECT_DIR] [RUN_DIR]
#
# Steps: import D3D windows+labels -> fetch frames -> zero-shot eval ->
#        LoRA+head training -> fine-tuned eval -> comparison table/figure.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="${1:-Qwen/Qwen3-VL-2B-Instruct}"
PROJ="${2:-projects/d3d}"
RUN="${3:-runs/$(basename "$BASE_MODEL" | tr '[:upper:]' '[:lower:]')-lora}"
PY="${PYTHON:-python}"

echo "== 0. deps"
$PY -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$PY -c "import physical_ai_av" 2>/dev/null || pip install physical-ai-av

echo "== 1. project from D3D outputs (282 clips / 724 windows / 724 Gemini labels)"
[ -f "$PROJ/project.json" ] || $PY -m autolabel init "$PROJ"
$PY -m autolabel import-d3d "$PROJ" --ego data/samples/ego_motion_results.jsonl --coc data/samples/coc_results_pro.jsonl

echo "== 2. frames (16 per window, 4-camera 2x2 grid as seen by Gemini)"
$PY -m autolabel fetch-frames "$PROJ" --layout grid

echo "== 3. zero-shot baseline"
$PY -m autolabel label "$PROJ" --backend qwen --backend-args "{\"base_model\": \"$BASE_MODEL\"}" --labels labels_qwen_zeroshot.jsonl
$PY -m autolabel eval  "$PROJ" --labels labels_qwen_zeroshot.jsonl --save results/eval_qwen_zeroshot.json
$PY -m autolabel label "$PROJ" --backend rule_based --labels labels_rule_based.jsonl
$PY -m autolabel eval  "$PROJ" --labels labels_rule_based.jsonl --save results/eval_rule_based.json

echo "== 4. LoRA + decision head"
$PY -m autolabel train "$PROJ" --out "$RUN" -- --base-model "$BASE_MODEL" \
    --epochs "${EPOCHS:-3}" --batch-size "${BATCH:-1}" --grad-accum "${ACCUM:-8}" --lr "${LR:-1e-4}" \
    --lora-r 16 --lora-alpha 32 --head-weight 0.5 --class-weights --eval-zero-shot --eval-limit "${EVAL_LIMIT:-100}"

echo "== 5. fine-tuned labels on every window (val clips are listed in $RUN/val_samples.jsonl)"
$PY -m autolabel label "$PROJ" --backend qwen --backend-args "{\"adapter_path\": \"$RUN/best\"}" --labels labels_qwen_lora.jsonl
$PY -m autolabel eval  "$PROJ" --labels labels_qwen_lora.jsonl --save results/eval_qwen_lora.json

echo "== 6. comparison"
$PY scripts/compare_backends.py results/eval_rule_based.json results/eval_qwen_zeroshot.json results/eval_qwen_lora.json --val "$RUN/val_samples.jsonl" --project "$PROJ"
echo "done: $RUN, results/compare_backends.json, report/figures/fig_backends.png"
