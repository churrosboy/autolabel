# Qwen3-VL-2B LoRA 파인튜닝 결과 보고 (CoC 오토라벨러)

- 실행일: 2026-09-27
- 실행 환경: RTX 4090 (24 GB), CUDA 13.0 드라이버, Python 3.11 (conda env `autolabel`)
- 주요 패키지: torch 2.14.0+cu130, torchvision 0.29.0, transformers 4.57.6, peft 0.21.0, physical-ai-av 0.2.2
- 그림: `report/figures/fig_backends.png` (백엔드 비교 막대그래프)

---

## 1. 요약

| 항목 | 결과 |
|---|---|
| 파인튜닝 효과 (zero-shot 대비) | decision-F1 **0.112 → 0.632**, 필터 탈락률 46% → 0% |
| 규칙 기반 대비 decision-F1 | **0.632 vs 0.587 (+0.046)**, 클립 단위 부트스트랩 95% CI [−0.035, +0.130] → 통계적으로 유의하지 않음 |
| 규칙 기반 대비 문장 품질 | ROUGE-L 0.402 vs 0.208 (약 2배), BLEU-4 0.239 vs 0.077 (약 3배) |
| 약점 | 완만한 감속(gentle_decel, 검증 최다 유형)에서 규칙 기반보다 낮음 (0.48 vs 0.54), 결정 헤드 정확도 낮음 (종 26%, 횡 37%) |

결론: 파인튜닝은 확실히 효과가 있었고, 원인까지 서술하는 CoC 라벨 생성기로는 규칙 기반보다 우수하다. 다만 주행 결정 자체의 정확도는 규칙 기반과 통계적으로 구분되지 않는 수준이다.

> 주의: 정답(GT)은 사람 라벨이 아니라 D3D 파이프라인의 **Gemini 생성 CoC 라벨**이다. 따라서 모든 수치는 "Gemini 라벨 재현도"이다.

---

## 2. 데이터

- 출처: NVIDIA Physical AI AV 데이터셋 (`nvidia/PhysicalAI-Autonomous-Vehicles`, HF gated), D3D 파이프라인 산출물(`data/samples/ego_motion_results.jsonl`, `data/samples/coc_results_pro.jsonl`)
- 규모: 282 클립, 724 윈도우, 윈도우당 16프레임 (8초, 2 Hz)
- 입력 이미지: 4카메라 2×2 그리드 (front wide 120°, front tele 30°, cross left, cross right), Gemini 라벨 생성 시와 동일 구성
- 분할: **클립 단위** 80/20 (seed 0), 같은 클립이 학습·검증에 동시에 들어가지 않음

| 분할 | 샘플 | 클립 |
|---|---|---|
| 학습 | 579 | 226 |
| 검증 | 145 | 56 |

### 키프레임 유형 분포

| 유형 | 학습 | 검증 |
|---|---|---|
| gentle_decel | 249 | 60 |
| gentle_accel | 136 | 28 |
| go_straight | 65 | 21 |
| stop | 36 | 8 |
| steer_l | 33 | 7 |
| steer_r | 30 | 15 |
| strong_accel | 24 | 5 |
| sharp_steer_l | 4 | 0 |
| sharp_steer_r | 2 | 1 |

### 결정 헤드 라벨 분포 (GT 문장에서 추출한 첫 번째 결정)

| 종방향 | 학습 | 검증 |
|---|---|---|
| set speed tracking | 186 | 37 |
| stop for static constraints | 135 | 27 |
| lead obstacle following | 96 | 33 |
| none | 64 | 22 |
| speed adaptation | 51 | 19 |
| yield | 45 | 7 |
| gap-searching | 2 | 0 |

| 횡방향 | 학습 | 검증 |
|---|---|---|
| lane keeping & centering | 234 | 54 |
| none | 211 | 40 |
| turn | 126 | 48 |
| merge / split, out-of-lane nudge, in-lane nudge, lane change | 각 2 | 0–2 |

클래스 불균형이 심하다 (횡방향 희소 클래스는 학습 샘플 2개).

---

## 3. 모델

- 베이스: **Qwen/Qwen3-VL-2B-Instruct** (가중치 고정, bf16)
- **LoRA**: 언어 모델의 `q/k/v/o_proj`, `gate/up/down_proj`에만 적용 (비전 인코더·merger 제외), r=16, alpha=32, dropout 0.05
- **결정 헤드**: 언어 모델 마지막 층에서 프롬프트 마지막 토큰의 hidden state(2048차원) 입력
  - `LayerNorm → Linear(2048→512) → GELU → Dropout(0.1)` 뒤 두 출력층
  - 종방향 8클래스 (Table-1 7개 + none), 횡방향 9클래스 (Table-1 8개 + none)
  - 추론 시 헤드 예측과 문장의 결정이 다르면 품질 필터가 H01 경고
- 학습 파라미터: 18,494,481개 (LoRA + 헤드, 전체의 1% 미만)
- 손실: `CoC 텍스트 CE + 0.5 × (종방향 CE + 횡방향 CE)`, 헤드 CE에는 역빈도 클래스 가중치

---

## 4. 학습 설정

| 항목 | 값 |
|---|---|
| epoch | 2 |
| batch / grad accum | 1 / 8 (유효 배치 8) |
| 학습률 | 1e-4, warmup 5%, weight decay 0.01, grad clip 1.0 |
| 이미지 | 최대 변 448 px, max_pixels 256×28×28 |
| 최대 시퀀스 | 4096 |
| gradient checkpointing | 사용 |
| 체크포인트 선택 | epoch별 검증 50개의 decision-F1 최고 → epoch 1 |
| 소요 시간 | epoch당 약 6분 50초 (업데이트 73회), GPU 메모리 약 16.7 GB |

### 손실 추이

| epoch | update | 전체 | 텍스트 | 종방향 헤드 | 횡방향 헤드 |
|---|---|---|---|---|---|
| 1 | 5 | 4.256 | 2.190 | 2.055 | 2.075 |
| 1 | 15 | 2.875 | 1.268 | 1.863 | 1.351 |
| 1 | 30 | 2.541 | 1.045 | 1.809 | 1.183 |
| 1 | 45 | 2.245 | 0.836 | 1.650 | 1.169 |
| 1 | 60 | 2.151 | 0.782 | 1.629 | 1.109 |
| 2 | 75 | 2.061 | 0.784 | 1.557 | 0.996 |
| 2 | 90 | 2.045 | 0.747 | 1.563 | 1.033 |
| 2 | 105 | 2.106 | 0.684 | 1.706 | 1.138 |
| 2 | 120 | 2.197 | 0.644 | 1.822 | 1.284 |
| 2 | 135 | 2.276 | 0.682 | 1.761 | 1.427 |

텍스트 손실은 계속 감소했지만 헤드 손실은 epoch 2 후반에 다시 증가했다.

### epoch별 검증 (검증 50개, 학습 스크립트 내부 평가)

| epoch | decision-F1 | 종방향 일치 | 횡방향 일치 | ROUGE-L | BLEU-4 | 헤드 종 정확도 | 헤드 횡 정확도 | 필터 통과 |
|---|---|---|---|---|---|---|---|---|
| 0 (학습 전) | 0.110 | 0.12 | 0.20 | 0.271 | 0.087 | 0.00 | 0.36 | 38% |
| **1 (best)** | **0.573** | 0.62 | 0.56 | 0.391 | 0.235 | 0.34 | 0.36 | 100% |
| 2 | 0.498 | 0.56 | 0.46 | 0.359 | 0.207 | 0.34 | 0.20 | 80% |

epoch 2에서 성능 하락 → 과적합 징후 (표본 50개라 변동 큼).

---

## 5. 최종 평가 (검증 145개, 학습에 쓰지 않은 56 클립)

| 백엔드 | decision-F1 | precision | recall | 종방향 일치 | 횡방향 일치 | component Jaccard | ROUGE-L | BLEU-4 | 필터 통과 | 필터 탈락 |
|---|---|---|---|---|---|---|---|---|---|---|
| 규칙 기반 (ego-motion만) | 0.587 | 0.555 | 0.662 | 0.586 | 0.490 | 0.539 | 0.208 | 0.077 | 100% | 0% |
| Qwen3-VL-2B zero-shot | 0.112 | 0.112 | 0.120 | 0.145 | 0.290 | 0.459 | 0.150 | 0.048 | 19% | 46% |
| **Qwen3-VL-2B + LoRA** | **0.632** | **0.671** | 0.623 | **0.676** | **0.614** | **0.634** | **0.402** | **0.239** | 100% | 0% |

- LoRA − 규칙 기반 decision-F1 차이: **+0.046**, 클립 단위 부트스트랩(1000회) 95% CI **[−0.035, +0.130]**, P(차이 ≤ 0) = 0.158
- 결정 헤드 단독 정확도 (검증 145개): 종방향 **25.5%**, 횡방향 **37.2%**

### 724개 전체 (참고용, 학습 데이터 포함)

| 백엔드 | n | decision-F1 | 95% CI |
|---|---|---|---|
| 규칙 기반 | 724 | 0.548 | [0.521, 0.576] |
| Qwen3-VL-2B + LoRA | 724 | 0.640 | [0.612, 0.671] |
| Qwen3-VL-2B zero-shot | 145 (검증만) | 0.112 | [0.067, 0.160] |

LoRA의 724개 수치는 학습 샘플을 포함하므로 보고용 성능은 5절 검증 결과를 사용한다.

### 키프레임 유형별 decision-F1 (검증)

| 유형 | n | 규칙 기반 | LoRA |
|---|---|---|---|
| gentle_decel | 60 | **0.54** | 0.48 |
| gentle_accel | 28 | 0.68 | **0.69** |
| go_straight | 21 | 0.40 | **0.69** |
| steer_r | 15 | 0.72 | **0.82** |
| stop | 8 | 0.64 | **0.77** |
| steer_l | 7 | 0.67 | **1.00** |
| strong_accel | 5 | **0.73** | 0.53 |
| sharp_steer_r | 1 | 0.67 | **1.00** |

n이 10 미만인 유형은 해석에 주의.

---

## 6. 정성 예시 (검증)

**go_straight (오답: 선행차 누락)**
- GT (Gemini): The ego vehicle maintains lane keeping & centering and lead obstacle following due to the presence of a lead vehicle ahead and a green traffic light.
- LoRA: The ego vehicle executes set speed tracking and lane keeping & centering because the traffic light is green and the road is clear.

**gentle_accel (부분 정답: 원인 서술 축약)**
- GT: Since the lead vehicle ahead begins to move forward under a green traffic light, the ego vehicle gently accelerates to maintain lead obstacle following and lane keeping & centering.
- LoRA: The ego vehicle accelerates to track the lead vehicle and maintain lane centering.

**steer_r (정답)**
- GT: Because the ego vehicle is approaching an intersection with a green traffic light and has a routing intent to turn right, it executes a planned turn onto the new road segment.
- LoRA: The ego vehicle executes a turn to the right because the traffic light is green and the routing intent is to turn right.

**zero-shot의 전형적 실패**: 2단계 THOUGHT_PROCESS를 길게 쓰다 FINAL_COC 형식을 지키지 못하거나, 같은 문장을 반복 → 필터 탈락 (S 계열 오류).

---

## 7. 분석

1. **파인튜닝 효과는 명확하다.** zero-shot 2B 모델은 Table-1 결정 어휘와 출력 형식을 거의 따르지 못하며(F1 0.11), LoRA 1 epoch만으로 F1 0.63, 필터 통과 100%에 도달했다.
2. **결정 정확도는 규칙 기반과 비슷하다.** 이 데이터의 키프레임은 ego-motion 전이로 검출되므로, 운동 정보만으로도 결정의 상당 부분이 결정된다. 모델도 프롬프트로 동일한 운동 정보와 키프레임 유형을 받기 때문에 이미지 이해가 추가로 기여한 부분은 제한적이다.
3. **설명력은 뚜렷하게 우수하다.** 규칙 기반은 픽셀을 보지 않아 원인(신호등, 선행차, 보행자 등)을 서술할 수 없다. LoRA 모델은 ROUGE-L 2배, component Jaccard 0.634로 원인 구성요소를 더 잘 맞힌다.
4. **완만한 감속이 약점이다.** 선행차 추종 / 속도 조절 / 정지 준비 중 무엇인지는 영상과 GT 모두에서 모호하며, Gemini GT 자체의 일관성 문제일 가능성도 있다.
5. **결정 헤드는 사실상 학습 부족.** 헤드 정확도(종 26%)가 문장 내 결정 일치(68%)보다 크게 낮아 현재 H01 교차검증 용도로는 신뢰하기 어렵다. 헤드 가중치 0.5와 희소 클래스가 원인으로 추정된다.
6. **과적합이 빠르다.** 579 샘플에서 epoch 2에 검증 성능이 떨어졌다.

### 한계

- GT가 사람 라벨이 아닌 Gemini 라벨 (라벨 잡음·상한 존재)
- 검증 145개, 56 클립으로 작음 → 백엔드 간 차이의 신뢰구간이 넓음
- 체크포인트 선택을 검증 50개로 했고 같은 검증 세트로 최종 평가 (별도 test 세트 없음) → 약간의 낙관적 편향 가능
- zero-shot은 2단계 긴 프롬프트 + max_new_tokens 640, LoRA는 학습용 짧은 프롬프트 사용 (조건이 다름). 기본값 200 토큰에서는 zero-shot이 FINAL_COC까지 도달하지 못해 늘렸음

---

## 8. 개선 방향

| 우선순위 | 방법 | 기대 효과 | 비용 |
|---|---|---|---|
| 1 | lr 5e-5, 3 epoch, 검증 전체로 best 선택 | 과적합 완화, 결정 정확도 소폭 상승 | 약 30분 |
| 2 | 헤드 손실 가중치 0.5 → 1.0 | 헤드 정확도 상승, H01 유용성 확보 | 1과 동시 가능 |
| 3 | Qwen3-VL-8B (4-bit QLoRA) | 이미지 이해 향상, 감속 원인 구분 | 학습 3–4배 |
| 4 | train / val / test 3분할 | 선택 편향 제거 | 데이터 추가 필요 |

---

## 9. 재현 방법

```bash
# 환경
conda create -n autolabel python=3.11 -y && conda activate autolabel
pip install numpy opencv-python-headless pillow matplotlib pytest lark \
    torch torchvision 'transformers>=4.57,<5' 'peft>=0.13' accelerate physical-ai-av
hf auth login            # nvidia/PhysicalAI-Autonomous-Vehicles 접근 승인된 계정
unset PYTHONPATH         # ROS 등 다른 워크스페이스의 pytest 플러그인 충돌 방지

# 데이터
python -m autolabel init projects/d3d
python -m autolabel import-d3d projects/d3d --ego data/samples/ego_motion_results.jsonl --coc data/samples/coc_results_pro.jsonl
python -m autolabel fetch-frames projects/d3d --layout grid      # 단일 프로세스 약 4시간, 이번에는 8프로세스 병렬로 약 40분

# 기준선
python -m autolabel label projects/d3d --backend rule_based --labels labels_rule_based.jsonl
python -m autolabel eval  projects/d3d --labels labels_rule_based.jsonl --save results/eval_rule_based.json

# 학습
python -m autolabel train projects/d3d --out runs/qwen2b-lora -- --base-model Qwen/Qwen3-VL-2B-Instruct \
    --epochs 2 --batch-size 1 --grad-accum 8 --lr 1e-4 --lora-r 16 --lora-alpha 32 --head-weight 0.5 \
    --class-weights --eval-zero-shot --eval-limit 50

# zero-shot (검증 클립만; runs/qwen2b-lora/val_samples.jsonl의 clip_id로 윈도우를 거른 뒤 실행)
python -m autolabel label projects/d3d --backend qwen --backend-args '{"max_new_tokens": 640}' --labels labels_qwen_zeroshot.jsonl
python -m autolabel eval  projects/d3d --labels labels_qwen_zeroshot.jsonl --save results/eval_qwen_zeroshot.json

# 파인튜닝 모델
python -m autolabel label projects/d3d --backend qwen --backend-args '{"adapter_path": "runs/qwen2b-lora/best"}' --labels labels_qwen_lora.jsonl
python -m autolabel eval  projects/d3d --labels labels_qwen_lora.jsonl --save results/eval_qwen_lora.json

# 비교 (검증 클립 기준)
python scripts/compare_backends.py results/eval_rule_based.json results/eval_qwen_zeroshot.json results/eval_qwen_lora.json \
    --val runs/qwen2b-lora/val_samples.jsonl --project projects/d3d
```

### 실행 중 수정한 레포 문제

- `autolabel/cli.py`: `train` 서브커맨드의 `argparse.REMAINDER`가 `--out`까지 삼켜 README 명령과 `scripts/run_finetune.sh`가 `--out is required`로 실패하던 버그 수정. 이제 `--` 뒤 인자만 학습 스크립트로 전달.
- `transformers` 5.x에서는 학습 시 `mm_token_type_ids is missing` 오류 → `<5`로 고정 필요 (requirements 미반영).
- `pip install -e .`는 setuptools flat-layout 탐지 오류로 실패 (`pyproject.toml`에 패키지 명시 필요).
- `physical-ai-av`는 Python ≥ 3.11 필요.

---

## 10. 산출물 위치 (학습 머신 `~/autolabel`)

| 내용 | 경로 | 비고 |
|---|---|---|
| 파인튜닝 라벨 (724개) | `projects/d3d/labels_qwen_lora.jsonl` | gitignore됨 |
| 규칙 기반 라벨 (724개) | `projects/d3d/labels_rule_based.jsonl` | gitignore됨 |
| zero-shot 라벨 (검증 145개) | `projects/d3d/labels_qwen_zeroshot.jsonl` | gitignore됨 |
| 모델 (LoRA + 헤드, 71 MB) | `runs/qwen2b-lora/best/` | gitignore됨 |
| 학습 로그 / epoch별 검증 예측 | `runs/qwen2b-lora/log.jsonl`, `val_predictions_epoch*.jsonl` | gitignore됨 |
| 학습/검증 분할 | `runs/qwen2b-lora/train_samples.jsonl`, `val_samples.jsonl` | gitignore됨 |
| 백엔드 비교표 | `results/compare_backends.json` | |
| 백엔드별 평가 (by_group 포함) | `results/eval_rule_based.json`, `eval_qwen_zeroshot.json`, `eval_qwen_lora.json` | |
| 비교 그림 | `report/figures/fig_backends.png` | |
