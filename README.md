# Autolabel — 내가 찍은 주행 영상에 CoC 라벨 자동 부여

Chain-of-Causation(CoC) 오토라벨링 파이프라인. 대시캠/폰으로 찍은 영상을 넣으면
(1) 영상에서 ego-motion을 추정하고 (2) 주행 결정 시점(keyframe)을 찾아
(3) 8초 윈도우(16프레임)를 잘라 (4) VLM(Gemini / Qwen3-VL+LoRA / OpenAI 호환 서버)
또는 규칙 기반 백엔드로 CoC 문장을 생성한 뒤 (5) 품질 필터로 걸러 (6) 브라우저에서
검수하고 (7) 학습용 데이터셋(JSONL + 프레임)으로 내보낸다.

```
video ─▶ motion ─▶ keyframes ─▶ windows/frames ─▶ backend ─▶ filter ─▶ review UI ─▶ export
        (optical  (transition   (16 @ 2Hz)       (rule /    (S/T/K    (approve/   (D3D-compatible
         flow or   + cooldown                     gemini /   checks)    edit/       coc_results.jsonl
         GPS/IMU)  + 8s cap)                      qwen-lora)            reject)     + samples.jsonl)
```

## 설치

```bash
cd Autolabel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv, pillow, matplotlib, pytest
# 선택: pip install google-genai        (gemini 백엔드)
# 선택: pip install torch transformers peft accelerate   (qwen 백엔드 / 파인튜닝, GPU 권장)
```

## 빠른 시작 (내 영상 라벨링)

```bash
python -m autolabel init  myproj --uniform 8        # 이벤트가 없으면 8초마다 윈도우
python -m autolabel add   myproj ~/Videos/drive1.mp4   # 폴더를 주면 안의 모든 영상 등록
python -m autolabel run   myproj --backend rule_based  # 모션 추정 → keyframe → 프레임 → 라벨 → 필터
python -m autolabel review myproj                      # http://127.0.0.1:8765 검수 UI
python -m autolabel export myproj --name v1            # myproj/exports/v1/
```

VLM 백엔드:

```bash
export GEMINI_API_KEY=...
python -m autolabel label myproj --backend gemini --force
python -m autolabel label myproj --backend qwen                                             # zero-shot Qwen3-VL
python -m autolabel label myproj --backend qwen --backend-args '{"adapter_path": "runs/qwen2b/best"}'   # fine-tuned (LoRA + head)
python -m autolabel label myproj --backend openai --backend-args '{"base_url": "http://localhost:8000/v1", "model": "Qwen/Qwen3-VL-8B-Instruct"}'
```

GPS/IMU 로그가 있으면 `add ... --csv log.csv` (time, speed, ax, yaw_rate/curvature 열 인식).

## 기존 데이터셋(D3D / physical_ai_av) 평가

```bash
python -m autolabel import-d3d myproj --ego data/samples/ego_motion_results.jsonl --coc data/samples/coc_results_pro.jsonl
python -m autolabel label  myproj --backend rule_based
python -m autolabel filter myproj
python -m autolabel eval   myproj          # decision-F1, ROUGE-L, BLEU vs GT
python scripts/run_experiments.py         # 보고서용 실험 전체 재현 (results/, report/figures/)
```

## 오토라벨러 파인튜닝 (Qwen3-VL + 결정 헤드 + LoRA)

```bash
pip install torch transformers peft accelerate physical-ai-av
export HF_TOKEN=...                                   # physical_ai_av (gated) 프레임 다운로드용

python -m autolabel init d3d && python -m autolabel import-d3d d3d \
    --ego data/samples/ego_motion_results.jsonl --coc data/samples/coc_results_pro.jsonl
python -m autolabel fetch-frames d3d --layout grid    # 724 윈도우 × 16프레임 (4카메라 2×2)

python -m autolabel label d3d --backend qwen --labels labels_qwen_zeroshot.jsonl   # 파인튜닝 전
python -m autolabel eval  d3d --labels labels_qwen_zeroshot.jsonl --save results/eval_zeroshot.json

python -m autolabel train d3d --out runs/qwen2b -- --base-model Qwen/Qwen3-VL-2B-Instruct \
    --epochs 3 --batch-size 1 --grad-accum 8 --eval-zero-shot          # LoRA + 헤드 학습, epoch마다 val 결정-F1

python -m autolabel label d3d --backend qwen --backend-args '{"adapter_path": "runs/qwen2b/best"}' --labels labels_qwen_lora.jsonl
python -m autolabel eval  d3d --labels labels_qwen_lora.jsonl --save results/eval_lora.json
python scripts/compare_backends.py results/eval_*.json        # 비교 표·그림
```

전체 순서는 `scripts/run_finetune.sh` 하나로 실행된다. 학습 목적함수는 CoC 텍스트 CE + 0.5 × (종방향 결정 CE + 횡방향 결정 CE)이고,
결정 헤드는 마지막 프롬프트 토큰의 hidden state에서 Table-1 결정(각 범주 + none)을 분류한다. 추론 시 헤드 예측과
문장의 결정이 다르면 필터가 H01 경고를 낸다.

## 품질 필터 코드

| 코드 | 내용 | 심각도 |
|---|---|---|
| S01–S14 | 길이, 언어, Table-1 결정 부재, 결정 개수, 인과 연결어, 모순 쌍, 헤징, 미래 프레임 인용, 프롬프트 누출, 구성요소 부재, 중복, 반복 | error/warning/info |
| T01–T05 | keyframe 타입(ego-motion 이벤트)과 결정의 정합성 | error/warning |
| K01–K07 | 결정과 실제 운동(정지/회전/차로변경/속도추세)의 물리적 정합성 | error/warning/info |
| H01 | 파인튜닝 모델의 결정 헤드 예측과 문장의 결정 불일치 | warning |

`error` 하나라도 있으면 `reject`, `warning`이 있으면 `review`, 아니면 `pass`.

## 테스트

```bash
pytest -q
```

## 구조

```
autolabel/
  schemas.py     데이터 구조 (EgoMotion, Keyframe, Window, Label, FilterReport)
  motion.py      optical-flow ego-motion 추정, GPS/IMU CSV, 운동학 요약
  keyframes.py   keyframe 검출 (전이 검출 + 속도 확인 + cooldown + 8초 cap)
  windows.py     윈도우 생성, 프레임 추출, 컨택트 시트
  vocab.py       Table-1/2 통제 어휘, 결정/구성요소 추출 (history/intent 제외)
  prompt.py      2단계(Stage I/II) CoC 프롬프트, FINAL_COC 파싱
  backends/      rule_based, gemini, qwen(zero-shot / LoRA+헤드), openai_compat
  training/      data(샘플·split·헤드 라벨), collate, model(Qwen3-VL+LoRA+헤드), train(루프·평가), fetch_frames
  filters.py     품질 필터 (S/T/K)
  evaluate.py    decision-F1, component Jaccard, ROUGE-L, BLEU
  pipeline.py    Project 워크스페이스, 단계 실행, 내보내기
  ui/            검수 웹 UI (표준 라이브러리 http.server)
  cli.py         명령줄
scripts/run_experiments.py   보고서 실험
tests/                       pytest
data/samples/                D3D 파이프라인 산출물 샘플 (282 clips / 724 labels)
```
