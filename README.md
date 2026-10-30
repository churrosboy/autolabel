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
python -m autolabel review myproj                      # http://127.0.0.1:8765/timeline 타임라인 UI (목록형 검수: /)
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

### 타임라인 UI

`autolabel review`는 Final Cut Pro 식 편집기 화면(`/timeline`)을 연다. 위쪽은 Index(윈도우·탈락 후보 목록) / Viewer(영상) / Inspector(선택 윈도우의 keyframe·kinematics·경로·라벨·프레임), 아래쪽은 전체 폭 타임라인이다.

- 타임라인: 눈금자 아래에 윈도우가 썸네일이 들어간 클립 막대로 놓인다(겹치면 위로 쌓임). 클립 안 흰 마커 = keyframe, 파란 틴트 = history(−2 s), 노란 테두리 = 선택. 그 아래 speed·accel·curv 세 레인이 오디오 파형처럼 그려지고 검출기 임계값은 점선이다.
- 세로선: 흰색 = playhead(재생 위치), 빨간색 = skimmer. 스키밍은 타임라인 위에서 마우스만 움직여도 뷰어가 그 시점을 보여 주는 기능이며 `s` 또는 툴바 버튼으로 끄고 켠다. 클릭·드래그 = playhead 이동(스크럽), 클립 안 마커 드래그 = keyframe 이동(윈도우·프레임·필터 재계산), `⌘ + 휠` = 확대.
- 눈금자의 회색 표식 = 검출기가 버린 후보. 마우스를 올리면 이유(speed confirmation / cooldown / window cap)가 표시되고 Index의 Candidates 탭에도 목록이 있다.
- 툴바: 왼쪽 `+ keyframe`(재생 위치에 윈도우 추가), 가운데 타임코드(HH:MM:SS:FF), 오른쪽 스키밍 토글과 목록 보기. 뷰어 아래 트랜스포트 = 이전/다음 keyframe, ±1 s, 재생.
- Inspector는 IDE처럼 오른쪽 세로 툴바에서 패널을 골라 연다(기본은 Label만). Label = 범주별 controlled vocabulary 칩(클릭 = 문장에 넣기/빼기, 같은 범주는 치환)·CoC 문장·필터 이슈·Approve/Save/Reject(현재 상태가 켜짐)·Relabel·Delete, Keyframe = 결정 순간 프레임과 유형 변경, Frames = 16프레임(마우스를 올리면 뷰어 위에 크게 미리보기), Kinematics, Path. 단축키 `1`–`5`로도 토글되고 선택은 브라우저에 기억된다.
- 단축키: `space` 재생, `←/→` 1프레임(`shift` = 1 s), `[`/`]` 윈도우 이동, `a`/`e`/`r` 검수, `s` 스키밍, `Home`/`End`. 타임라인 바 오른쪽의 `?` 버튼을 누르면 전체 목록이 뜬다.
- 데이터셋 클립(import-d3d)은 영상 파일이 없으므로 프레임만 표시된다. `fetch-frames --video --clips <id>`로 HF chunk zip에서 해당 클립의 mp4만 꺼내 `myproj/video/<clip_id>.mp4`(+ `.video.json` 사이드카: 재생 위치 0의 클립 시각 `t0_us`)에 두면 플레이어가 켜진다. 원본은 HEVC라 H.264 변환(ffmpeg 있으면 자동)이 필요하다. 사용자 영상도 브라우저가 코덱을 못 읽으면 `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`로 변환한다.
- 뷰어와 타임라인 사이 경계선을 드래그하면 타임라인 높이가, Index·Viewer·Inspector 사이 세로 경계선을 드래그하면 각 패널 폭이 바뀐다. 세로 경계선 더블클릭은 기본 폭으로 되돌리고, 크기는 브라우저에 저장된다.

## 기존 데이터셋(D3D / physical_ai_av) 평가

```bash
python -m autolabel import-d3d myproj --ego data/samples/ego_motion_results.jsonl --coc data/samples/coc_results_pro.jsonl
python -m autolabel review myproj             # --as-labels 를 붙여 import 하면 Gemini CoC가 라벨로 등록되어 타임라인에서 바로 검수 가능
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
python -m autolabel fetch-frames d3d --clips 01d3588e --video   # 한 클립만: 16프레임 + 전방 광각 mp4 (타임라인 플레이어용)

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
