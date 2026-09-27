/* Builds the interim report (.docx + .md) from results/metrics.json and report/figures.
 *   node report/gen/build_report.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, Footer, PageNumber, LevelFormat,
  TableLayoutType, VerticalAlign,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "report", "figures");
const M = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "metrics.json"), "utf8"));
const EX = JSON.parse(fs.readFileSync(path.join(ROOT, "results", "filter_examples.json"), "utf8"));

const KO = "맑은 고딕";
const EN = "Arial";
const MONO = "Consolas";
const BODY = 21; // half-points (10.5 pt)
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const f2 = (x, d = 2) => Number(x).toFixed(d);

// --------------------------------------------------------------------------- content model
const blocks = [];
const H1 = (t) => blocks.push({ k: "h1", t });
const H2 = (t) => blocks.push({ k: "h2", t });
const H3 = (t) => blocks.push({ k: "h3", t });
const P = (t, o = {}) => blocks.push({ k: "p", t, ...o });
const UL = (items) => blocks.push({ k: "ul", items });
const TABLE = (cap, header, rows, widths) => blocks.push({ k: "table", cap, header, rows, widths });
const FIGURE = (file, cap, cm = 15) => blocks.push({ k: "fig", file, cap, cm });
const CODE = (title, file) => blocks.push({ k: "code", title, file });
const BR = () => blocks.push({ k: "br" });
let figN = 0, tabN = 0;
const figRef = () => ++figN;
const tabRef = () => ++tabN;

// --------------------------------------------------------------------------- numbers
const D = M.dataset;
const E1 = M.E1_redetection, E1u = M.E1_unsupported_by_speed, E1f = M.E1_full_clip_keyframes;
const E2 = M.E2_filter_gt, E3 = M.E3_rule_based_vs_gt, E4r = M.E4_video_real, E4s = M.E4_video_synthetic, E5 = M.E5_filter_rule_based;
const V = E2.verdicts, N2 = E2.n;
const O = E3.overall;
const typeOrder = Object.keys(E1f.types);

// =========================================================================== COVER
blocks.push({ k: "cover" });

// =========================================================================== 요약
H1("■ 요약");
P(`본 작품은 사용자가 직접 촬영한 주행 영상(대시캠·스마트폰)에 Alpamayo-R1이 제안한 Chain-of-Causation(CoC) 형식의 주행 인과 설명을 자동으로 부여하는 오토라벨링 애플리케이션 “Autolabel”을 구현한 것이다. 신청서의 진행계획 중 Ⅲ(CoC 오토라벨링 파이프라인 구현: 자동 추론 엔진과 품질 필터링)과 Ⅳ(정량적 평가)를 중간 단계에서 동작하는 소프트웨어로 완성하였고, Ⅱ(VLM LoRA 파인튜닝)는 오픈 VLM(Qwen3-VL)에 결정 분류 헤드 하나를 붙여 LoRA로 학습하는 파이프라인으로 구현하여 소형 모델 dry-run까지 검증하였다. 실제 모델 학습과 비교 평가는 GPU 환경에서 수행할 예정이다.`);
P(`파이프라인은 (1) 영상에서 광학 흐름(optical flow)으로 ego-motion을 추정하고, (2) 종·횡방향 가속 및 곡률 전이로부터 주행 결정 시점(keyframe)을 검출하며, (3) keyframe 기준 [-2 s, +6 s] 윈도우의 16프레임을 추출하고, (4) 규칙 기반·Gemini·Qwen3-VL(+LoRA)·OpenAI 호환 서버 중 선택한 백엔드로 CoC 문장을 생성한 뒤, (5) 구조(S)·keyframe 정합(T)·운동학 정합(K)의 세 축 26개 규칙으로 품질을 검사하고, (6) 브라우저 검수 UI에서 승인·수정·기각한 뒤, (7) 학습용 데이터셋(JSONL + 프레임)으로 내보낸다.`);
P(`정량 평가는 NVIDIA Physical AI AV 데이터셋 ${D.clips}개 클립에서 Gemini로 생성한 ${D.gt_labels}개 CoC 라벨을 기준으로 수행하였다. 품질 필터는 라벨당 ${(E2.seconds / N2 * 1000).toFixed(2)} ms 만에 ${pct(V.pass / N2)}를 통과, ${pct(V.review / N2)}를 검수 대상, ${pct(V.reject / N2)}를 기각으로 분류하였고, 운동학 검사(K04)는 차량이 70° 이상 회전했음에도 “직진”이라고 서술된 VLM 라벨 오류 12건을 실제로 검출하였다. 또한 기존 파이프라인의 가속도 기반 keyframe 중 감속 유형의 ${pct(E1u.decel_no_speed_drop / E1u.decel_total, 0)}가 속도 감소를 동반하지 않는 잡음이며, 이 잘못된 메타 정보가 프롬프트를 통해 VLM 라벨에 그대로 전이되는 현상을 확인하고 속도 확인(speed confirmation) 단계를 추가하였다. 운동 정보만 사용하는 규칙 기반 백엔드는 결정 F1 ${f2(O.decision_f1, 3)}으로 파인튜닝 모델이 넘어야 할 하한 기준선을 제공한다. 27개 단위·통합 테스트가 모두 통과하며, 합성 영상 기준 실시간 대비 약 ${E4s.realtime_factor}배 속도로 처리된다.`);

// =========================================================================== 서론
H1("■ 서론");
H2("1.1 제안 배경 및 필요성");
P(`자율주행 인공지능은 “무엇이 보이는가”를 인식하는 단계에서 “왜 그렇게 행동하는가”를 설명하는 단계로 진화하고 있다. NVIDIA의 Alpamayo-R1 [1]은 주행 궤적과 함께 결정의 인과 근거를 자연어로 서술하는 Chain-of-Causation(CoC) 데이터를 제안하였고, 이러한 추론 데이터로 학습한 비전-언어-행동(VLA) 모델이 롱테일 상황에서 계획 정확도를 크게 높인다는 것을 보였다. DriveLM [4] 역시 그래프 형태의 시각 질의응답으로 인식-예측-계획의 추론 사슬을 데이터화하였다. 즉 “판단의 이유”가 담긴 데이터가 설명 가능하고 안전한 자율주행의 핵심 자산이 되고 있다.`);
P(`문제는 비용이다. CoC 한 문장을 만들려면 8초 남짓의 영상을 보고 결정 이전 구간에서 원인이 되는 객체를 찾고, 결정 이후 구간에서 실제로 취해진 종·횡방향 결정을 통제 어휘로 분류한 뒤, 둘을 인과적으로 연결해야 한다. Alpamayo-R1 논문도 이를 사람이 모두 작성하지 않고, VLM 자동 라벨링과 사람 검수를 결합한 하이브리드 파이프라인으로 구축하였다 [1]. 국내에서도 학습 데이터 자동 생성 시스템 [12], 자동 레이블링 기반 영상 학습데이터 제작 시스템 [13], 자율주행 학습 데이터 수집·전처리 환경 [14] 등 라벨링 자동화 연구가 이어지고 있으나, 대부분 객체 검출용 박스·세그먼트 라벨을 대상으로 하며 자연어 인과 설명을 다루는 사례는 드물다.`);
P(`범용 VLM(Gemini, GPT, Qwen 등)에 프롬프트만 주어 CoC를 생성하는 방식에는 두 가지 한계가 있다. 첫째, 주행 도메인 지식이 부족하여 “정지선”과 “차선”을 혼동하거나 통제 어휘(Table 1의 15개 결정)를 벗어난 표현을 사용한다. 둘째, 생성된 문장이 실제 차량의 움직임과 모순되어도 이를 스스로 걸러내지 못한다. 본 연구에서 기존 라벨을 분석한 결과, 차량이 70° 이상 회전하는 구간을 “직진을 유지한다”고 서술하거나, 프롬프트에 주어진 잘못된 메타 정보(감속 이벤트)를 영상보다 우선시하여 실제로는 가속하는 장면을 감속으로 설명하는 사례가 실제로 존재하였다(4.3절).`);
P(`한 가지 더 중요한 배경은 데이터의 출처이다. 공개 데이터셋은 촬영 지역, 차량, 센서 구성이 고정되어 있어 국내 도로나 특정 환경을 반영하기 어렵고, ego-motion·LiDAR 같은 센서 정보가 함께 제공된다는 전제 위에 설계되어 있다. 반면 연구자나 개발자가 직접 확보할 수 있는 데이터는 대개 스마트폰이나 블랙박스로 찍은 “영상만 있는” 데이터이다. 따라서 오토라벨러가 실제로 쓸모 있으려면, 센서 없이 영상 파일만 넣어도 결정 시점을 찾고 CoC를 붙여 학습용 데이터셋으로 만들어 주는 애플리케이션 형태여야 한다.`);
P(`요약하면 본 작품의 필요성은 (i) CoC 라벨 구축 비용의 절감, (ii) 범용 VLM 라벨의 신뢰도 문제 해결, (iii) 센서가 없는 사용자 촬영 영상까지 라벨링 대상으로 확장하는 것에 있다.`);

H2("1.2 연구논문/작품의 목표");
P(`신청서에 제시한 진행계획(Ⅰ 데이터 전처리, Ⅱ LoRA 파인튜닝, Ⅲ 오토라벨링 파이프라인, Ⅳ 성능 검증)에 따라 본 중간 단계의 목표를 다음과 같이 설정하였다.`);
UL([
  "목표 1 (파이프라인 애플리케이션): 사용자가 촬영한 영상 파일을 입력으로 받아 ego-motion 추정, keyframe 검출, 윈도우 추출, CoC 생성, 품질 필터, 검수, 데이터셋 내보내기를 명령줄과 브라우저 UI로 수행하는 소프트웨어를 구현한다.",
  "목표 2 (품질 필터링): 신청서 Ⅲ의 “문법적 오류나 주행 상식에 어긋나는 논리적 모순을 자동으로 검출”하는 후처리를 규칙 기반으로 구현하되, 텍스트만이 아니라 차량의 실제 운동(정지·회전·차로 변경·속도 추세)과 대조하는 물리적 정합성 검사를 포함한다.",
  "목표 3 (정량적 평가): 기준(GT) 라벨과 생성 라벨을 통제 어휘 수준(결정 F1)과 문장 수준(ROUGE-L, BLEU)에서 비교하는 평가 도구를 만들고, 파인튜닝 전 기준선을 확보한다.",
  "목표 4 (LoRA 오토라벨러): 오픈 VLM(Qwen3-VL)에 결정 분류 헤드를 붙여 CoC 라벨만 잘 생성하도록 LoRA로 학습하는 파이프라인을 구현하고, zero-shot 모델·규칙 기반 기준선과 같은 지표로 비교한다.",
]);

H2("1.3 작품 전체 overview");
P(`그림 ${figRef()}은 작품의 전체 구성이다. 입력은 두 갈래이다. 사용자 영상(mp4/mov, 선택적으로 GPS/IMU CSV)은 1단계 모션 추정부터 시작하고, 이미 ego-motion과 윈도우가 정의된 공개 데이터셋(physical_ai_av, D3D 산출물)은 3단계 윈도우로 바로 들어온다. 이후 4단계 백엔드가 CoC를 생성하고 5단계 필터가 검사한 뒤 6단계 검수 UI를 거쳐 7단계로 내보내며, 8단계 평가 도구는 내보낸 라벨을 GT와 비교한다. 9단계 학습 모듈은 프로젝트의 윈도우·프레임·라벨로 Qwen3-VL에 LoRA와 결정 헤드를 학습시키고, 그 어댑터는 다시 4단계 Qwen 백엔드에 장착된다. 이 순환 구조가 “한정된 CoC 데이터 → 오토라벨러 → 더 많은 데이터 → 더 나은 오토라벨러”라는 신청서의 기대효과를 구현한다.`);
FIGURE("fig_architecture.png", `그림 ${figN}. Autolabel 시스템 구성도. 1–3은 영상 처리, 4–5는 라벨 생성·검사, 6–8은 검수·산출·평가, 9는 오토라벨러 파인튜닝 단계이다.`, 15.5);
P(`중간 결과를 요약하면 다음과 같다. 첫째, 전체 파이프라인이 GPU 없이 노트북에서 끝까지 동작한다(합성 24초 영상 기준 모션 추정 ${E4s.motion_seconds}초, 실시간 대비 ${E4s.realtime_factor}배). 둘째, 품질 필터가 기존 Gemini 라벨 ${N2}개 중 ${V.reject}개를 기각, ${V.review}개를 검수 대상으로 분류하였고, 그중 운동학 검사는 사람이 보아도 명백한 라벨 오류(회전 중 “직진”)를 잡아냈다. 셋째, 규칙 기반 기준선의 결정 F1은 ${f2(O.decision_f1, 3)}(95% 신뢰구간 ${f2(O.decision_f1_ci95[0], 3)}–${f2(O.decision_f1_ci95[1], 3)})로, 향후 LoRA 파인튜닝 모델이 넘어야 할 하한을 제공한다. 넷째, 기존 데이터의 keyframe 검출 잡음(감속 유형의 ${pct(E1u.decel_no_speed_drop / E1u.decel_total, 0)})을 발견하고 개선하였다. 다섯째, Qwen3-VL + 결정 헤드 + LoRA 학습 파이프라인이 데이터 구성부터 학습 스텝, 저장, 재로드, 추론, 필터 연동까지 소형 모델로 끝까지 동작함을 확인하였다.`);

H2("1.4 보고서의 구성");
P(`2장은 추론 기반 자율주행, VLM과 LoRA, 자동 라벨링, 영상 기반 ego-motion 추정, 텍스트 평가 지표에 관한 관련 연구를 정리한다. 3장은 CoC의 구조와 결정 윈도우, keyframe 검출, 광학 흐름 기반 모션 추정, 품질 필터의 세 축이라는 이론적 배경과 시스템 구성·모듈·데이터 형식, 그리고 오토라벨러 파인튜닝 설계를 상세히 소개한다. 4장은 구현 환경과 여섯 가지 실험(E1 keyframe 검출, E2 품질 필터, E3 규칙 기반 기준선, E4 영상 입력, E5 필터 자기 일관성, E6 학습 파이프라인 dry-run)의 결과와 분석, 그리고 발견된 한계를 다룬다. 5장은 결론과 소감, 6장은 참고문헌이며, 부록에 사용법과 원천코드를 수록한다.`);

// =========================================================================== 관련연구
H1("■ 관련연구");
H2("2.1 추론 기반 자율주행과 Chain-of-Causation");
P(`대규모 언어모델에서 중간 추론 과정을 명시적으로 생성하게 하면 복잡한 문제의 정확도가 높아진다는 Chain-of-Thought 프롬프팅 [11]은 자율주행에도 이식되었다. DriveLM [4]은 nuScenes 위에 인식→예측→계획으로 이어지는 질의응답 그래프를 구축하여 VLM이 주행 결정을 단계적으로 설명하도록 하였다. Alpamayo-R1 [1]은 한 걸음 더 나아가 “원인 요소(critical component) → 주행 결정(driving decision)”의 인과 사슬을 8초 결정 윈도우 단위로 서술하는 CoC 데이터를 정의하고, 이를 Cosmos-Reason 기반 VLM과 확산(flow matching) 궤적 디코더로 구성된 10B 규모 VLA 모델의 지도학습·강화학습에 사용하였다. 특히 결정 어휘를 종방향 7종, 횡방향 8종의 통제 어휘(Table 1)로, 원인 요소를 7개 범주(Table 2)로 제한한 점이 라벨의 일관성과 자동 검증 가능성을 높였으며, 본 작품의 필터와 평가 지표는 이 통제 어휘를 그대로 채택한다. 후속 연구인 WorkDrive는 공사 구간에 특화된 CoC를, Cognitive Dual-Process Planning은 추론-행동 일관성의 검증 가능성을 다루어 CoC 계열 데이터가 하나의 흐름을 형성하고 있음을 보여 준다.`);
H2("2.2 비전-언어 모델과 효율적 파인튜닝");
P(`Qwen3-VL [2]은 2B부터 235B까지의 밀집·MoE 변형을 제공하는 개방형 VLM으로, 이미지·비디오·텍스트가 섞인 256K 토큰 문맥을 처리한다. Alpamayo-R1의 공개 구현도 Qwen3-VL 계열 백본을 사용하며, 본 작품은 같은 계열의 Instruct 모델을 오토라벨러의 기반으로 삼는다. Gemini [10]는 본 연구에서 기준 라벨 생성에 사용한 폐쇄형 멀티모달 모델이다. LoRA [3]는 사전학습 가중치를 고정한 채 저차원 행렬 쌍만 학습하여 전체 파라미터의 1% 미만으로 도메인 적응을 달성하는 기법으로, 10B 규모 VLM을 단일 GPU에서 파인튜닝할 수 있게 한다. 국내에서도 LoRA를 교육용 LLM에 적용하여 BLEU를 크게 개선한 사례 [15]와, 커리큘럼 러닝으로 LoRA 미세조정 성능을 높이는 방법론 [16]이 보고되어, 소규모 도메인 데이터로 대형 모델을 특화하는 접근이 국내외에서 검증되고 있다.`);
H2("2.3 자동 라벨링과 약지도 학습");
P(`사람이 규칙(labeling function)을 작성하고 그 출력을 확률적으로 결합하여 대량의 학습 라벨을 만드는 데이터 프로그래밍 [7]은 자동 라벨링의 이론적 토대이다. 본 작품의 규칙 기반 백엔드와 품질 필터는 이 관점에서 “운동학 규칙”을 라벨링 함수이자 검증 함수로 사용한다. 국내 연구로는 시뮬레이터에서 날씨·조도·차량 구성을 바꾸어 자율주행 학습 데이터를 자동 생성하는 시스템 [12], 반복적 수작업 레이블링을 자동화하는 영상 학습데이터 제작 시스템 [13], 자율주행 차량의 인공지능 학습용 데이터 수집 환경과 전처리 데이터 구축 연구 [14]가 있다. 이들은 주로 검출·분할 라벨을 다루며, 본 작품은 같은 문제의식을 자연어 인과 설명 라벨로 확장한다. 라벨링과 별개로, 사람이 최종 승인하는 human-in-the-loop 구조는 Alpamayo-R1 [1]의 데이터 구축과 동일한 원칙이다.`);
H2("2.4 영상 기반 ego-motion 추정");
P(`센서 없이 영상만으로 차량의 움직임을 추정하는 문제는 시각 주행거리계(visual odometry)로 오래 연구되었다. 본 작품은 실시간성과 의존성 최소화를 위해 Farnebäck의 다항식 전개 기반 조밀 광학 흐름 [5]을 사용한다. 전진 운동은 소실점을 중심으로 한 흐름장의 방사형 팽창(divergence)으로, 회전은 원거리 영역의 수평 흐름으로 나타난다는 기하학적 사실을 이용하여 속도와 요(yaw) 대리 신호를 얻는다. 절대 스케일은 얻을 수 없으므로 신호를 강건 척도(MAD)로 정규화하고 임계값을 “표준편차 단위”로 두는 방식을 택하였다.`);
H2("2.5 텍스트 생성 평가 지표");
P(`BLEU [8]는 n-gram 정밀도의 기하평균에 길이 페널티를 곱한 지표이고, ROUGE-L [6]은 최장 공통 부분수열 기반의 F-measure이다. 두 지표는 표현이 다양할 수 있는 CoC 문장의 “의미” 일치를 과소평가하므로, 본 작품은 통제 어휘로 정규화한 결정 집합의 정밀도·재현율·F1과 원인 요소 범주의 Jaccard를 주 지표로 하고 BLEU/ROUGE-L을 보조 지표로 사용한다.`);
H2("2.6 데이터셋");
P(`NVIDIA Physical AI Autonomous Vehicles 데이터셋 [9]은 7개 카메라, 10 Hz ego-motion(위치·속도·가속도·곡률)이 포함된 20초 클립 모음으로, Alpamayo-R1 공개 모델의 추론 예제가 이 데이터에 맞추어 제공된다. 본 연구의 사전 작업(Gemini 라벨링 파이프라인)에서 이 데이터셋 929개 클립의 keyframe을 검출하고 282개 클립, 724개 윈도우에 Gemini로 CoC를 생성하였으며, 본 작품은 그 산출물을 평가 기준이자 파인튜닝 데이터로 사용한다.`);

// =========================================================================== 제안 작품
H1("■ 제안 작품 소개");
H2("3.1 이론적 배경");
H3("3.1.1 CoC 라벨의 구조와 결정 윈도우");
P(`CoC 한 건은 하나의 결정 윈도우에 대응한다(그림 ${figRef()}). keyframe(결정 시점) 이전 2초는 원인이 관찰되는 Stage I, 이후 6초는 결정이 실행되는 Stage II이며, 2 Hz로 총 16프레임을 샘플링한다. 라벨은 “The vehicle performs [결정] because [원인 요소] required this action to [결과]” 형식의 한 문장으로, 결정은 표 ${tabRef()}의 통제 어휘에서 종·횡 각 최대 1개를 고른다. 인과 국소성(causal locality) 원칙에 따라 원인은 Stage I에서만 찾아야 하며, 이 원칙은 필터 규칙 S10(미래 프레임 인용 금지)으로 검사된다.`);
FIGURE("fig_window.png", `그림 ${figN}. 결정 윈도우의 구조. keyframe 기준 [-2 s, +6 s] 구간을 2 Hz 16프레임으로 샘플링하고, 같은 구간의 10 Hz ego-motion(81개 샘플)이 운동학 검사에 사용된다.`, 15);
TABLE(`표 ${tabN}. CoC 결정 통제 어휘(Alpamayo-R1 Table 1)와 keyframe 유형별 기대 결정`, ["범주", "결정", "정의(요약)"], [
  ["종방향", "set speed tracking", "제약이 없을 때 목표 속도를 유지·도달"],
  ["", "lead obstacle following", "같은 흐름의 선행 차량과 안전 간격 유지"],
  ["", "speed adaptation", "곡선·과속방지턱 등 도로 특성에 맞춘 속도 조절"],
  ["", "gap-searching", "횡방향 기동을 위해 목표 흐름에 속도 맞춤"],
  ["", "acceleration for passing", "추월을 위한 가속"],
  ["", "yield", "보행자·교차 교통 등에 우선권 양보"],
  ["", "stop for static constraints", "정지선·적신호 등 통제 지점에서 감속·정지"],
  ["횡방향", "lane keeping & centering", "차로 내 위치 유지"],
  ["", "merge / split", "진입로·분기 구간 전환"],
  ["", "out-of-lane / in-lane nudge", "장애물 회피를 위한 차로 밖/안 편향"],
  ["", "lane change", "인접 차로로 완전 이동"],
  ["", "pull-over / curb approach", "갓길·정차 구역 접근"],
  ["", "turn", "다른 도로 구간으로 진행 방향 변경"],
  ["", "lateral maneuver abort", "진행 중인 횡방향 기동 취소"],
], [1500, 3300, 4560]);

H3("3.1.2 ego-motion으로부터의 keyframe 검출");
P(`주행 결정은 ego-motion의 상태 전이로 나타난다. 종방향은 이동평균한 종가속도 aₓ가 임계값을 넘는 순간(강가속 aₓ>3, 완가속 1.5<aₓ≤3, 완감속 −7≤aₓ<−1, 강감속 aₓ<−7 m/s²)과 0.5초 이상 속도 0.1 m/s 미만인 정지 시작을, 횡방향은 |곡률|의 전이(조향 0.02<|κ|≤0.15, 급조향 |κ|>0.15 m⁻¹, 직진 복귀 |κ|≤0.02)를 이벤트로 잡고, 결정이 전이보다 앞선다는 점을 반영해 keyframe을 전이 0.5초 전에 둔다. 여기에 세 가지 필터를 더한다. (i) 같은 유형이 3초 안에 반복되면 제거하는 cooldown, (ii) Alpamayo-R1의 데이터 구성 원칙대로 8초 윈도우당 종·횡 각 최대 1개만 남기는 cap, (iii) 윈도우가 클립 밖으로 나가는 keyframe 제거. 본 작품은 추가로 (iv) 속도 확인(speed confirmation)을 도입하였다. 가속도 신호가 감속 전이를 보고하더라도 이후 2초 안에 속도가 0.5 m/s 이상 실제로 줄지 않으면 이벤트를 버린다. 4.2절에서 보듯 기존 데이터의 감속 keyframe 중 3분의 1이 이 확인을 통과하지 못하는 잡음이었다.`);
H3("3.1.3 광학 흐름 기반 ego-motion 추정");
P(`센서가 없는 영상에서는 Farnebäck 조밀 광학 흐름 [5]으로 대리 신호를 만든다. 분석 해상도 320픽셀, 10 Hz로 연속 프레임 쌍의 흐름장 F(u,v)를 구한 뒤, 팬(pan)에 의한 전역 이동을 제거하기 위해 흐름의 중앙값을 뺀다. 속도 대리 신호는 하늘·보닛을 제외한 관심 영역에서 소실점 기준 방사 단위벡터와의 내적 평균, 즉 팽창률 s = mean⟨F − median(F), r̂⟩이고, 요 대리 신호는 원거리 띠(화면 높이 35–65%)의 수평 흐름 중앙값 y = median(Fₓ)이다. 우회전 시 장면이 왼쪽으로 흐르므로 y<0이 되어 곡률 부호 규약(좌회전 양수)과 일치한다. 두 신호를 0.5초 이동평균하고 강건 척도 1.4826·MAD로 나누어 무차원화하며, 가속도는 정규화 속도의 시간 미분이다. 영상 소스에는 이 단위에 맞춘 별도 임계값(강가속 2.5, 완가속 1.0, 완감속 −1.0, 조향 1.0, 급조향 2.5)을 적용한다. 정규화된 속도와 요를 적분한 추측 항법 경로는 회전각 검사에 쓰인다.`);
H3("3.1.4 품질 필터의 세 축");
P(`생성된 문장은 그림 ${figRef()}과 같이 세 축으로 검사된다. 먼저 통제 어휘의 정규식 사전으로 결정과 원인 요소를 추출하는데, 이때 “after stopping at the stop sign, it accelerates…”처럼 과거 맥락(history)에 등장한 결정과 “prepare for a planned right turn”처럼 의도(intent)만 언급된 회전은 결정으로 세지 않는다. 이 두 예외 처리가 없으면 정상 라벨의 15%가 모순으로 오판된다(4.3절). S 검사는 길이·언어·결정 부재·결정 개수·인과 연결어·상호 배타 쌍(예: 가속 추월 vs 정지)·헤징 표현·미래 프레임 인용·프롬프트 문구 누출·원인 요소 부재·중복·퇴화 반복을 본다. T 검사는 keyframe 유형과 결정의 정합성으로, 감속 이벤트에 “acceleration for passing”이 붙으면 오류, 기대 집합 밖이면 경고이다. K 검사는 윈도우의 ego-motion 요약(정지 여부, 최소·최종 속도, 회전각, 횡변위, 가속도 추세)과 결정을 대조한다. 정지를 주장했는데 감속조차 없으면 오류, 회전을 주장했는데 회전각이 15° 미만이거나 직진을 주장했는데 45°를 넘으면 경고를 낸다. 오류가 하나라도 있으면 기각, 경고만 있으면 검수, 아니면 통과이며 가중 감점으로 0–1 점수를 계산해 검수 우선순위를 정한다.`);
FIGURE("fig_filter_flow.png", `그림 ${figN}. 품질 필터의 흐름. 통제 어휘 추출 후 구조(S), keyframe 정합(T), 운동학 정합(K) 검사를 병렬로 수행하고 판정을 검수 UI로 넘긴다.`, 15);
H3("3.1.5 평가 지표");
P(`생성 라벨 p와 기준 라벨 r에서 추출한 결정 집합을 각각 Dₚ, Dᵣ라 할 때 결정 정밀도 |Dₚ∩Dᵣ|/|Dₚ|, 재현율 |Dₚ∩Dᵣ|/|Dᵣ|, F1을 계산하고, 종·횡 첫 번째 결정의 일치 여부(lon/lat match), 원인 요소 범주의 Jaccard, ROUGE-L F1 [6], 평활화된 BLEU-4 [8]를 함께 보고한다. F1의 95% 신뢰구간은 1,000회 부트스트랩으로 구한다.`);

H2("3.2 시스템 구성");
H3("3.2.1 모듈 구성");
P(`작품은 Python 패키지로 구현되었으며(파이프라인은 numpy, OpenCV, Pillow만 필수, 파인튜닝은 torch, transformers, peft) 표 ${tabRef()}의 모듈로 나뉜다. 백엔드·학습·UI를 포함해 약 4,800행이고, 27개의 단위·통합 테스트가 있다.`);
TABLE(`표 ${tabN}. 모듈 구성`, ["모듈", "역할", "핵심 구성"], [
  ["schemas.py", "데이터 구조", "EgoMotion, Clip, Keyframe, Window, Label, FilterReport (numpy 기반 dataclass)"],
  ["motion.py", "ego-motion 소스", "영상 광학 흐름 추정, GPS/IMU CSV 로더, physical_ai_av 변환, 운동학 요약(회전각·횡변위·속도 추세)"],
  ["keyframes.py", "결정 시점 검출", "전이 검출, 속도 확인, cooldown, 8초 cap, 균등 분할 대체"],
  ["windows.py", "윈도우·프레임", "[-2, +6] s 윈도우, OpenCV 프레임 추출, 16장 컨택트 시트"],
  ["vocab.py", "통제 어휘", "Table 1/2 정규식 사전, history/intent 예외, 결정·요소 추출"],
  ["prompt.py", "프롬프트", "2단계(Stage I/II) CoC 프롬프트, 궤적 텍스트, FINAL_COC 파싱"],
  ["backends/", "라벨 생성기", "rule_based(모션 전용), gemini, qwen(+LoRA, PEFT), openai_compat(vLLM/Ollama)"],
  ["filters.py", "품질 필터", "S/T/K 26개 규칙, 판정·점수, 통계 요약"],
  ["evaluate.py", "평가", "결정 P/R/F1, 요소 Jaccard, ROUGE-L, BLEU-4, 부트스트랩"],
  ["pipeline.py", "워크스페이스", "Project(클립 등록, 단계 실행, 재개, 검수 상태, 내보내기, 통계)"],
  ["ui/", "검수 UI", "표준 라이브러리 HTTP 서버 + 단일 HTML(필터·검색·승인·수정·기각·내보내기)"],
  ["training/", "파인튜닝", "data(샘플·클립 분할·헤드 라벨), collate, model(Qwen3-VL+LoRA+헤드), train(루프·평가·체크포인트), fetch_frames"],
  ["cli.py", "명령줄", "init/add/run/label/filter/review/export/eval/import-d3d/fetch-frames/train/stats"],
], [1700, 1900, 5760]);
H3("3.2.2 프로젝트 워크스페이스와 데이터 흐름");
P(`사용자는 프로젝트 디렉터리 하나로 작업한다. autolabel init으로 만든 디렉터리에 add로 영상(또는 폴더)을 등록하면 clips.jsonl에 경로·해상도·fps가 기록되고, run이 motion/<clip>.json(ego-motion), windows.jsonl(윈도우와 ego 조각), frames/<clip>/kfNNN/frame_XX.jpg(16프레임)와 sheet.jpg(컨택트 시트), labels.jsonl(라벨·필터 결과·검수 상태)을 차례로 만든다. 모든 단계는 재개 가능하여 이미 처리된 클립은 건너뛰고, --force로 다시 만든다. 라벨은 10건마다 저장되어 API 호출 중 중단되어도 손실이 없다. export는 검수에서 기각된 것과 필터 기각을 제외하고(옵션으로 승인된 것만) D3D 학습 코드가 읽는 coc_results.jsonl과, 프레임 경로·타임스탬프·백엔드·필터 점수·검수 상태를 담은 samples.jsonl을 exports/<이름>/에 쓴다.`);
H3("3.2.3 백엔드");
P(`모든 백엔드는 Window와 프레임 목록을 받아 원문 응답을 돌려주는 하나의 인터페이스를 구현하며, FINAL_COC 한 문장은 공통 파서가 추출한다(표 ${tabRef()}). 규칙 기반 백엔드는 픽셀을 보지 않고 운동학 요약만으로 결정을 정하고 문장을 조립한다. 원인 객체를 알 수 없으므로 “a gentle decel event (speed 10.9 to 7.1 m/s, heading change −10 deg)”처럼 관측된 운동을 원인 자리에 적는다. 이 백엔드는 (i) GPU·API 없이 파이프라인 전체를 시험하는 대역, (ii) 운동학만으로 얻을 수 있는 성능의 하한 기준선, (iii) 필터의 K 검사가 참조하는 물리 사전이라는 세 역할을 한다.`);
TABLE(`표 ${tabN}. 라벨 생성 백엔드`, ["백엔드", "입력", "요구 사항", "용도"], [
  ["rule_based", "ego-motion", "없음", "오프라인 테스트, 기준선, 물리 사전"],
  ["gemini", "16프레임 + 프롬프트", "GEMINI_API_KEY", "기준 라벨 생성(D3D와 동일 프롬프트)"],
  ["qwen", "16프레임 + 프롬프트", "torch, transformers, peft, GPU", "Qwen3-VL zero-shot, 또는 LoRA 어댑터 + 결정 헤드(3.3절) 장착"],
  ["openai", "16프레임 + 프롬프트", "vLLM/Ollama 등 호환 서버", "GPU 서버에 띄운 파인튜닝 모델 원격 사용"],
], [1500, 2200, 2800, 2860]);
H3("3.2.4 검수 UI와 내보내기");
P(`review 명령은 127.0.0.1의 로컬 HTTP 서버를 띄우고 브라우저를 연다. 각 라벨 카드는 컨택트 시트(과거 4장과 미래 12장을 테두리 색으로 구분), keyframe 유형, 백엔드, 필터 판정과 점수, 추출된 결정·원인 요소, 회전각·속도 요약, 이슈 목록, 편집 가능한 문장을 보여 주며 승인·수정 저장·기각 버튼으로 상태를 바꾼다. 판정·검수 상태·검색어로 필터링할 수 있어 “review로 분류된 것만” 빠르게 훑는 워크플로를 지원한다. 수정 이력은 라벨의 meta에 남는다. 그림 ${figRef()}은 실제 스마트폰 영상에서 추출된 컨택트 시트의 예이다.`);
FIGURE("fig_demo_sheet.jpg", `그림 ${figN}. 스마트폰 영상(1080×1920, 30 fps)에서 추출된 결정 윈도우 컨택트 시트. 1–4번은 결정 이전(Stage I), 5–16번은 결정 이후(Stage II) 프레임이다.`, 11);

H2("3.3 오토라벨러 파인튜닝: Qwen3-VL + 결정 헤드 + LoRA");
P(`신청서 Ⅱ의 파인튜닝은 자율주행 정책 모델이 아니라 “라벨을 잘 쓰는 모델”을 목표로 하므로, 오픈 VLM에 라벨링에 필요한 최소한의 구조만 더한다. 기반 모델은 Qwen3-VL Instruct(노트북에서는 2B, GPU에서는 4B/8B)이고, 언어 모델의 어텐션·MLP 투영(q/k/v/o, gate/up/down)에 LoRA(r=16, α=32)를 붙이며 비전 인코더와 병합기는 고정한다. 여기에 결정 헤드(decision head) 하나를 추가한다. 프롬프트의 마지막 토큰 hidden state를 LayerNorm–MLP를 거쳐 종방향 결정 8클래스(Table 1의 7종 + none)와 횡방향 결정 9클래스(8종 + none)로 분류하는 작은 네트워크이다. 헤드의 정답은 CoC 문장에서 3.1.4절의 통제 어휘 추출기로 자동 생성되므로 추가 라벨링이 필요 없다.`);
P(`목적함수는 L = CE_text + λ·(CE_lon + CE_lat), λ = 0.5이다. CE_text는 “FINAL_COC: …” 한 문장에 대한 토큰 교차 엔트로피(프롬프트 토큰은 마스킹)이고, 두 헤드 손실은 클래스 불균형을 완화하기 위해 역빈도의 제곱근으로 가중한다. 헤드는 두 가지 역할을 한다. 학습 시에는 모델이 문장을 쓰기 전에 하나의 통제 어휘 결정에 “약속”하도록 만드는 보조 신호이고, 추론 시에는 문장에서 추출한 결정과 헤드 예측이 다르면 필터가 H01 경고를 내는 자기 일관성 검사의 근거가 된다.`);
TABLE(`표 ${tabRef()}. 학습 설정(기본값)`, ["항목", "값"], [
  ["기반 모델", "Qwen/Qwen3-VL-2B-Instruct (GPU: 4B/8B)"],
  ["LoRA", "r=16, α=32, dropout 0.05, 언어 모델 투영층 전체"],
  ["결정 헤드", "LayerNorm → Linear(h, h/4) → GELU → Linear(8) / Linear(9)"],
  ["목적함수", "CE_text + 0.5 × (CE_lon + CE_lat), 헤드 손실 클래스 가중"],
  ["입력", "16프레임(최대 448px, 64–256 시각 토큰/장) + 2단계 프롬프트(FINAL_COC만 출력)"],
  ["최적화", "AdamW lr 1e-4, wd 0.01, 5% warmup 후 cosine, grad clip 1.0, 배치 1 × 누적 8, bf16"],
  ["데이터", "프로젝트의 windows/labels(또는 gt.json)/frames; 클립 단위 8:2 분할"],
  ["평가", "epoch마다 val 클립에서 greedy 생성 → 결정 F1, 종·횡 일치, 헤드 정확도, 필터 통과율; 최고 F1 체크포인트 저장"],
], [2200, 7160]);
P(`데이터는 프로젝트 디렉터리에서 그대로 읽는다. 기존 Gemini 라벨을 학습에 쓰려면 import-d3d로 윈도우와 GT 문장을 들여온 뒤 fetch-frames가 Physical AI AV 데이터셋에서 각 윈도우의 16프레임을 Gemini가 보았던 것과 같은 4카메라 2×2 그리드로 내려받는다. 사용자가 촬영한 영상은 run으로 만든 프레임과 검수를 마친 라벨이 그대로 학습 샘플이 된다. 학습 스크립트는 zero-shot 평가(epoch 0)부터 시작해 epoch마다 검증 지표를 기록하고, 학습이 끝나면 label --backend qwen --backend-args '{"adapter_path": …}'로 어댑터와 헤드를 장착한 모델이 라벨러가 된다. scripts/run_finetune.sh는 import → 프레임 → zero-shot 평가 → 학습 → 파인튜닝 평가 → 비교 표·그림까지를 한 번에 수행한다.`);

// =========================================================================== 구현 및 결과
H1("■ 구현 및 결과분석");
H2("4.1 구현 환경과 실험 설계");
P(`구현은 macOS(Apple Silicon), Python 3.9, numpy 2.0, OpenCV 4.x 환경에서 이루어졌고 GPU와 외부 API를 사용하지 않았다. 평가 데이터는 사전 작업의 Gemini 라벨링 파이프라인이 NVIDIA Physical AI AV 데이터셋에서 만든 산출물이다. 929개 클립의 keyframe 목록(13,531개), 그중 ${D.clips}개 클립 ${D.windows}개 윈도우의 10 Hz ego-motion, 그리고 같은 윈도우에 Gemini(gemini-3.1-pro-preview, temperature 0.2, 4카메라 2×2 그리드 16프레임 입력)가 생성한 ${D.gt_labels}개 CoC 문장(평균 ${Math.round(D.coc_len_chars.mean)}자)이다. 표 ${tabRef()}에 여섯 실험을 정리하였다. E1–E5의 수치는 scripts/run_experiments.py 한 번으로, E6은 pytest tests/test_training.py로 재현된다.`);
TABLE(`표 ${tabN}. 실험 구성`, ["실험", "질문", "데이터"], [
  ["E1 keyframe 검출", "검출기가 기존 keyframe을 재검출하는가, 속도 확인이 무엇을 거르는가", "724 윈도우 ego-motion, 929 클립 keyframe"],
  ["E2 품질 필터", "기존 Gemini 라벨을 어떻게 판정하며 무엇을 잡아내는가", "724 GT 라벨 + ego-motion"],
  ["E3 규칙 기반 기준선", "운동학만으로 GT와 얼마나 일치하는가", "724 윈도우"],
  ["E4 영상 입력", "센서 없는 영상에서 결정 시점을 찾는가, 처리 속도는", "스마트폰 영상 18.7 s, 합성 영상 24 s"],
  ["E5 필터 자기 일관성", "운동학으로 만든 라벨이 K 검사를 통과하는가", "724 규칙 기반 라벨"],
  ["E6 학습 파이프라인 dry-run", "데이터→학습→저장→재로드→추론→필터가 끝까지 도는가", "합성 영상 2클립, 소형 무작위 모델"],
], [2000, 4700, 2660]);

H2("4.2 E1: keyframe 검출");
P(`그림 ${figRef()}(a)는 929개 클립에서 검출된 13,531개 keyframe의 유형 분포이다. 클립당 평균 ${f2(E1f.per_clip_mean, 1)}개(중앙값 ${E1f.per_clip_median}개)로, 완감속(${E1f.types.gentle_decel})과 완가속(${E1f.types.gentle_accel})이 전체의 ${pct((E1f.types.gentle_decel + E1f.types.gentle_accel) / E1f.total, 0)}를 차지하고 정지·급조향은 드물다. 이 불균형은 라벨 데이터에도 그대로 이어져 GT 724개 중 gentle_decel이 ${D.keyframe_types.gentle_decel}개이다.`);
P(`(b)는 각 GT 윈도우의 ego-motion 8초 구간만 넣었을 때 본 작품의 검출기가 keyframe 시각(t = 2 s) ±1.5 s 안에서 같은 범주의 이벤트를 다시 찾는 비율이다. 가속도 전이만 쓰면 ${pct(E1.no_confirm.overall_same_category)}(같은 유형 ${pct(E1.no_confirm.overall_same_type)})로 기존 구현과 사실상 동일함을 확인하였다. 속도 확인을 켜면 전체 ${pct(E1.speed_confirm.overall_same_category)}로 떨어지는데, 감소분은 거의 전부 gentle_decel(${pct(E1.speed_confirm.per_type.gentle_decel.same_category)})과 gentle_accel(${pct(E1.speed_confirm.per_type.gentle_accel.same_category)})에서 나온다. 직접 세어 보면 감속 keyframe ${E1u.decel_total}개 중 ${E1u.decel_no_speed_drop}개(${pct(E1u.decel_no_speed_drop / E1u.decel_total)})는 이후 6초 동안 속도가 1 m/s도 줄지 않았고, 가속은 ${E1u.accel_total}개 중 ${E1u.accel_no_speed_rise}개(${pct(E1u.accel_no_speed_rise / E1u.accel_total)})만 그러하다. 즉 데이터셋의 종가속도 채널은 속도 채널과 일치하지 않는 순간적 잡음을 포함하며, 이를 그대로 쓰면 “감속하지 않은 감속 이벤트”가 대량으로 라벨링 대상이 된다. 속도 확인은 이 잡음을 제거하는 대신 재검출률을 낮추는 것이 아니라, 잘못된 keyframe을 걸러내는 것이므로 기본값으로 켜 두었다.`);
FIGURE("fig_e1_keyframes.png", `그림 ${figN}. (a) 929개 클립의 keyframe 유형 분포(진한 색 종방향, 연한 색 횡방향). (b) 윈도우 내 재검출률. 속도 확인은 완감속·완가속의 잡음 이벤트를 걸러낸다.`, 15.5);

H2("4.3 E2: 품질 필터");
H3("4.3.1 판정 결과");
P(`그림 ${figRef()}은 필터를 ${N2}개 GT 라벨에 적용한 결과이다. 통과 ${V.pass}개(${pct(V.pass / N2)}), 검수 ${V.review}개(${pct(V.review / N2)}), 기각 ${V.reject}개(${pct(V.reject / N2)})이며 ${N2}개를 처리하는 데 ${E2.seconds}초가 걸렸다. 결정 어휘 추출 커버리지는 ${pct(E2.decision_coverage)}로, 결정을 하나도 찾지 못한 라벨은 ${Math.round((1 - E2.decision_coverage) * N2)}개뿐이다. 표 ${tabRef()}는 이슈 코드별 빈도이다.`);
FIGURE("fig_e2_filter.png", `그림 ${figN}. (a) 724개 Gemini 라벨에 대한 필터 판정. (b) 이슈 코드 빈도. 진한 색은 오류(기각), 회색은 경고(검수), 연한 색은 정보.`, 15.5);
const codeDesc = {
  K01: "정지 주장 vs 운동(감속 중이면 정보, 감속 없으면 오류)", K05: "감속 서술 vs 속도 상승", S06: "범주당 결정 2개 이상(정보)", T02: "keyframe 범주의 결정 부재",
  K04: "직진 주장 vs 45° 이상 회전", S07: "인과 연결어 부재", S08: "상호 배타 결정 쌍", K03: "회전 주장 vs 15° 미만 회전각", T01: "keyframe 유형과 모순되는 결정",
  T03: "keyframe 유형에 드문 결정", S05: "Table-1 결정 부재", S01: "너무 짧음", S12: "원인 요소 부재", S03: "INSUFFICIENT_EVIDENCE", T05: "가속 keyframe에 정지 서술", K02: "차로 변경 주장 vs 횡변위 부족",
};
TABLE(`표 ${tabN}. 이슈 코드별 검출 건수(724개 GT 라벨)`, ["코드", "내용", "건수", "비율"],
  Object.entries(E2.issue_codes).map(([c, n]) => [c, codeDesc[c] || "", String(n), pct(n / N2)]), [1200, 5560, 1300, 1300]);
H3("4.3.2 필터가 잡아낸 실제 오류");
P(`가장 의미 있는 검출은 K04(직진 주장 vs 큰 회전각) ${E2.issue_codes.K04}건이다. 그림 ${figRef()}은 그중 한 사례로, 라벨은 “교차로에서 교차 차량이 지나가길 기다린 뒤 가속하여 직진한다(proceed straight)”고 서술하지만 ego 경로는 6초 동안 ${Math.abs(Math.round(EX.k04_case.kinematics.heading_change_deg))}° 회전한다. 정지 상태에서 출발하는 저속 구간이라 사람이 프레임만 보아도 놓치기 쉬운 오류이며, 텍스트만 검사하는 필터로는 원리적으로 잡을 수 없다. 12건 모두 같은 양상(정지 후 출발하며 회전)이었다.`);
FIGURE("fig_k04_example.png", `그림 ${figN}. K04로 기각된 라벨의 ego 경로(a)와 속도(b). 라벨은 직진을 서술하지만 차량은 좌회전한다.`, 15);
P(`두 번째는 K05(감속 서술 vs 속도 상승) ${E2.issue_codes.K05}건이다. 예를 들어 “following a lead vehicle …, it applies gentle deceleration”이라는 라벨의 윈도우에서 속도는 5.2 m/s에서 11.5 m/s로 단조 증가한다. 원인은 4.2절의 keyframe 잡음이다. 프롬프트에 “TARGET META ACTION: [gentle_decel]”이 주어지면 VLM은 영상보다 이 메타 정보를 우선하여 존재하지 않는 감속을 서술한다. 즉 keyframe 검출 오류가 라벨 오류로 전이되며, 이것이 속도 확인 단계와 K 검사를 함께 두어야 하는 이유이다.`);
P(`T01·S08(각 ${E2.issue_codes.T01}, ${E2.issue_codes.S08}건)은 가속 keyframe에 “stop for static constraints”가 함께 추출되어 모순으로 판정된 경우로, 문장이 “정지선에서 정지한 뒤 출발하며 속도를 회복한다”처럼 과거 정지와 현재 가속을 함께 서술한 것이다. 이는 결정 어휘의 정의상 한 윈도우에 결정이 둘 담긴 셈이라 검수 대상으로 남겨 두는 것이 타당하다. S05(${E2.issue_codes.S05}건)는 “applies gentle deceleration to prepare for a planned right turn”처럼 통제 어휘 없이 서술된 라벨이고, S03 1건은 Gemini가 INSUFFICIENT_EVIDENCE를 출력했음에도 기존 파이프라인이 저장한 것이다.`);
H3("4.3.3 필터의 반복 개선");
P(`필터는 GT 라벨에 대한 오탐 분석을 세 차례 반복하며 다듬었다(표 ${tabRef()}). 첫 버전은 108개를 기각했는데, 그 대부분이 오탐이었다. “has stopped at the stop sign”의 과거 정지를 현재 결정으로 추출하거나, “stop sign”의 stop을 정지 동사로 세거나, “prepare for a planned right turn”의 의도를 회전 실행으로 보거나, 정지 판정을 “6초 안에 완전 정지”로만 두어 정지선을 향해 감속 중인 정상 라벨을 기각한 것이다. history/intent 예외, 정지 동사의 형태 제한, 감속 추세가 있으면 정지 주장을 허용하는 완화, 저속 구간에서 노이즈에 강한 회전각 계산(첫·끝 1 m 이동 방향 기준)을 적용하면서 기각은 ${V.reject}개, 검수는 ${V.review}개로 줄었고 남은 기각의 대부분은 4.3.2절과 같은 실제 문제이다.`);
TABLE(`표 ${tabN}. 필터 개선 반복에 따른 판정 변화(724개 GT 라벨)`, ["버전", "주요 변경", "통과", "검수", "기각"], [
  ["v1", "초기 규칙", "457", "159", "108"],
  ["v2", "history/intent 예외, 정지 동사 제한, K01 완화, 회전각 계산 개선", "584", "112", "28"],
  ["v3", "단수형·완전정지 표현 추가, 정지 후 출발 케이스 K05 예외, S06 정보 강등", String(V.pass), String(V.review), String(V.reject)],
], [900, 5260, 1000, 1000, 1200]);
P(`E5에서 규칙 기반 라벨 ${E5.n}개에 같은 필터를 적용하면 ${E5.verdicts.pass}개가 통과하고 ${E5.verdicts.review || 0}개 검수, ${E5.verdicts.reject || 0}개 기각으로, 운동학에서 만든 문장은 운동학 검사와 일관됨을 확인하였다. 남은 몇 건은 keyframe 유형과 실제 운동이 어긋난 데이터(T 검사)이다.`);

H2("4.4 E3: 규칙 기반 기준선");
P(`표 ${tabRef()}와 그림 ${figRef()}은 픽셀을 전혀 보지 않는 규칙 기반 백엔드의 라벨을 Gemini 라벨과 비교한 결과이다. 결정 F1 ${f2(O.decision_f1, 3)}(정밀도 ${f2(O.decision_precision, 3)}, 재현율 ${f2(O.decision_recall, 3)}), 종방향 첫 결정 일치 ${pct(O.lon_match)}, 횡방향 ${pct(O.lat_match)}, 결정이 하나라도 겹치는 비율 ${pct(O.any_decision_overlap)}이다. 문장 지표는 ROUGE-L ${f2(O.rouge_l, 3)}, BLEU-4 ${f2(O.bleu4, 3)}로 낮은데, 원인 요소를 서술할 수 없는 기준선이 GT와 표현을 공유할 수 없으므로 당연한 결과이며 이 지표의 의미가 결정 F1보다 약함을 보여 준다.`);
TABLE(`표 ${tabN}. keyframe 유형별 규칙 기반 기준선 성능`, ["유형", "n", "결정 F1", "종방향 일치", "횡방향 일치", "ROUGE-L"],
  typeOrder.filter((t) => E3.by_group[t]).map((t) => { const g = E3.by_group[t]; return [t, String(g.n), f2(g.decision_f1), f2(g.lon_match), f2(g.lat_match), f2(g.rouge_l)]; }).concat([["전체", String(O.n), f2(O.decision_f1), f2(O.lon_match), f2(O.lat_match), f2(O.rouge_l)]]),
  [1900, 900, 1400, 1700, 1700, 1760]);
FIGURE("fig_e3_baseline.png", `그림 ${figN}. keyframe 유형별 규칙 기반 기준선과 Gemini 라벨의 일치도.`, 15.5);
P(`유형별로 보면 횡방향 이벤트(steer_l/r, sharp_steer)에서 횡방향 일치가 0.88–1.0으로 높다. 회전은 경로에서 직접 관측되기 때문이다. 반대로 gentle_decel의 결정 F1이 ${f2(E3.by_group.gentle_decel.decision_f1)}로 가장 낮은데, 이 유형의 GT 종방향 결정이 stop(${E2.gt_primary_lon_by_type.gentle_decel["stop for static constraints"]}), lead following(${E2.gt_primary_lon_by_type.gentle_decel["lead obstacle following"]}), speed adaptation(${E2.gt_primary_lon_by_type.gentle_decel["speed adaptation"]}), yield(${E2.gt_primary_lon_by_type.gentle_decel["yield"]})로 넓게 퍼져 있고, 이 넷을 가르는 것은 “왜 감속했는가”라는 시각 정보이기 때문이다. 즉 운동학이 결정의 절반가량을 설명하고 나머지는 시각적 원인 이해가 필요하며, 이 간극이 VLM(과 LoRA 파인튜닝)이 기여해야 할 부분이다. 기준선의 처리 시간은 라벨당 ${(E3.seconds_per_label * 1000).toFixed(2)} ms이다.`);

H2("4.5 E4: 사용자 영상 입력");
P(`센서가 없는 영상에 대한 검증은 두 가지로 하였다. 첫째, 실제 스마트폰 영상(1080×1920, 30 fps, 18.7초, 야간 주차장에서 이동하며 촬영)을 등록·실행하여 파이프라인 전체(모션 추정 → keyframe → 프레임 추출 → 규칙 기반 라벨 → 필터 → 내보내기)가 끝까지 동작함을 확인하였다. 모션 추정은 ${E4r.motion_seconds_measured}초(실시간 대비 약 4배)가 걸렸고 ${E4r.samples}개 샘플에서 ${E4r.keyframes.length}개 keyframe(${E4r.keyframes.map((k) => k.type).join(", ")})이 검출되었다(그림 ${figRef()}(a)). 다만 이 영상은 차량 주행이 아니라 사람이 들고 걸은 것이어서 속도 대리 신호가 걸음마다 진동하며, 주행 의미의 결정 시점으로 해석할 수는 없다. 실제 주행 영상 확보가 최종 보고서까지의 과제이다.`);
FIGURE("fig_e4_video_motion.png", `그림 ${figN}. 스마트폰 영상에서 추정한 속도·요 대리 신호와 검출된 keyframe(점선).`, 15);
P(`둘째, 정답을 아는 합성 영상으로 검출 정확성을 확인하였다. 원근 투영된 텍스처 평면 위를 카메라가 6 m/s로 전진(0–5초), 정지(5–8초), 전진하며 0.35 rad/s 우회전(8–24초)하도록 렌더링한 24초 영상(240×160, 20 fps)에서, 검출기는 ${E4s.keyframes.map((k) => `${(k.timestamp_us / 1e6).toFixed(1)} s ${k.type}`).join(", ")}을 찾았다(그림 ${figRef()}(b)). 정지 시작(5.0 s)과 회전 시작(8.0 s)이 각각 0.5초 선행 규칙에 맞게 검출되었고, 회전 방향(우회전)도 요 부호로 올바르게 구분되었다. 처리 시간은 모션 추정 ${E4s.motion_seconds}초(실시간 대비 ${E4s.realtime_factor}배), keyframe·프레임 추출 ${E4s.keyframe_and_frames_seconds}초였으며 ${E4s.keyframes.length}개 윈도우 중 ${E4s.verdicts.pass || 0}개가 필터를 통과해 내보내졌다. 한편 회전이 시작되면 속도 대리 신호가 크게 감소하는 현상이 보이는데, 회전 중에는 수평 흐름이 지배적이 되어 팽창률 추정이 불안정해지기 때문이다. 이로 인해 회전 구간에서 거짓 가속·감속 이벤트가 생길 수 있어, 특징점 기반 시각 주행거리계로 대체하거나 회전 중 종방향 임계값을 높이는 개선이 필요하다.`);
FIGURE("fig_e4_synthetic.png", `그림 ${figN}. 합성 주행 영상의 속도·요 대리 신호. 회색 구간은 전진, 붉은 구간은 정지이며 점선이 검출된 keyframe이다.`, 15);

H2("4.6 E6: 파인튜닝 파이프라인 dry-run");
P(`이 맥(16 GB, GPU 없음)에서는 실제 Qwen3-VL을 학습할 수 없으므로, 기반 모델의 구조(설정 파일과 프로세서만 내려받음)를 hidden 64, 2층으로 축소한 무작위 초기화 모델(약 1천만 파라미터)로 학습 파이프라인 전체를 검증하였다. 합성 주행 영상 2클립을 프로젝트에 넣어 규칙 기반 라벨을 만든 뒤 클립 단위로 분할하고, LoRA와 결정 헤드를 붙여 2 스텝 학습, 어댑터·헤드 저장, 검증 클립에서 생성·평가, 최고 체크포인트 저장까지 수행하였다. 이어서 저장된 어댑터를 qwen 백엔드로 다시 읽어 한 윈도우를 라벨링하고, 헤드 예측이 라벨의 meta에 실려 필터의 H01 검사에 전달되는 것을 확인하였다. 전 과정은 CPU에서 약 50초가 걸렸고 pytest로 자동화되어 있다. 실제 모델로의 실행은 HF 토큰(프레임 다운로드)과 GPU가 있는 환경에서 scripts/run_finetune.sh로 수행하며, 결과 비교 표와 그림은 compare_backends.py가 생성한다.`);

H2("4.7 종합 분석과 한계");
UL([
  `달성: 신청서 Ⅲ의 자동 추론 엔진(4종 백엔드 인터페이스)과 품질 필터링(26개 규칙), Ⅳ의 정량 평가 도구를 동작하는 애플리케이션으로 구현하였고, 24개 테스트와 재현 스크립트로 검증하였다.`,
  `발견 1: 기존 CoC 데이터의 감속 keyframe ${pct(E1u.decel_no_speed_drop / E1u.decel_total, 0)}가 속도 변화 없는 잡음이며, 이 메타 정보가 프롬프트를 통해 VLM 라벨의 오류로 전이된다. 속도 확인과 K 검사가 이를 차단한다.`,
  `발견 2: 텍스트만 보는 검사로는 잡을 수 없는 “회전 중 직진” 유형의 라벨 오류가 GT의 1.7%에 존재하며, 운동학 대조가 이를 검출한다.`,
  `발견 3: 운동학만으로 결정의 약 55%를 맞힐 수 있으나 감속의 원인(정지선·선행차·곡선·양보)은 구분할 수 없어, 시각 이해가 필요한 나머지가 VLM 파인튜닝의 목표 영역이다.`,
  `한계 1: 파인튜닝 파이프라인은 구현·검증되었으나 실제 Qwen3-VL 학습과 zero-shot 대비 비교는 GPU 환경에서 수행할 예정이다. 기준 라벨 자체가 Gemini 출력이므로 “모델 대 사람” 정확도는 아직 측정되지 않았다.`,
  `한계 2: 실제 차량 주행 영상으로의 검증이 없다. 광학 흐름 대리 신호는 정지·출발·회전 시작 검출에는 충분하지만 회전 중 속도 추정이 불안정하다.`,
  `한계 3: 필터는 영어 통제 어휘에 특화되어 있어 다른 언어나 자유 서술에는 정규식 사전을 확장해야 한다.`,
]);

// =========================================================================== 결론
H1("■ 결론 및 소감");
P(`본 중간보고서는 사용자가 직접 촬영한 주행 영상에 CoC 인과 설명을 자동으로 붙이는 오토라벨링 애플리케이션 Autolabel의 설계·구현·검증 결과를 정리하였다. 영상만으로 결정 시점을 찾는 모션 추정기, 통제 어휘에 기반한 결정·원인 추출기, 텍스트와 실제 운동을 대조하는 품질 필터, 검수 UI, 학습 데이터 내보내기, 정량 평가 도구를 하나의 파이프라인으로 완성하였고, 기존 데이터를 재분석하여 keyframe 잡음과 VLM 라벨 오류라는 두 가지 실질적인 데이터 품질 문제를 찾아 대응하였다. 남은 기간에는 (1) GPU 환경에서 run_finetune.sh로 Qwen3-VL LoRA 학습을 실행하고, (2) zero-shot Qwen·Gemini·규칙 기반·파인튜닝 모델을 같은 평가 틀에서 비교하며, (3) 실제 주행 영상을 촬영해 센서 없는 입력에 대한 검증을 완료하고, (4) 회전 중 속도 추정을 보강할 계획이다.`);
P(`소감으로, 이번 단계에서 가장 크게 배운 것은 “라벨을 만드는 것”보다 “라벨을 의심하는 것”이 어렵고 중요하다는 점이다. 처음에는 VLM이 만든 문장을 정답으로 두고 필터를 만들었지만, 필터의 첫 버전이 108개를 기각했을 때 그 대부분은 필터의 오탐이었고 소수는 진짜 라벨 오류였다. 둘을 가르기 위해 문장을 하나씩 읽고 차량의 경로를 그려 보는 과정에서, 규칙을 세 번 갈아엎으며 과거 맥락과 의도 표현이라는 언어적 미묘함을 코드로 옮기는 경험을 했다. 또한 가속도 신호가 속도와 모순되는 데이터셋의 특성을 발견했을 때, 상류(keyframe) 오류가 하류(라벨)로 조용히 전파된다는 사실이 데이터 파이프라인 전체를 하나의 시스템으로 보아야 하는 이유를 실감하게 했다. 외부 API와 GPU 없이도 재현 가능한 실험을 먼저 갖추어 두니 이후 모델을 바꾸어도 같은 잣대로 비교할 수 있게 되었다는 점도 값진 성과라 생각한다.`);

// =========================================================================== 참고문헌
H1("■ 참고문헌");
const refs = [
  `Y. Wang, W. Luo, J. Bai, et al., "Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail," arXiv:2511.00088, 2025.`,
  `S. Bai, Y. Cai, R. Chen, et al., "Qwen3-VL Technical Report," arXiv:2511.21631, 2025.`,
  `E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in Proc. ICLR, 2022.`,
  `C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li, "DriveLM: Driving with Graph Visual Question Answering," in Proc. ECCV, 2024.`,
  `G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion," in Proc. Scandinavian Conf. on Image Analysis (SCIA), LNCS 2749, pp. 363-370, 2003.`,
  `C.-Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries," in Proc. ACL Workshop on Text Summarization Branches Out, pp. 74-81, 2004.`,
  `A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré, "Snorkel: Rapid Training Data Creation with Weak Supervision," Proc. VLDB Endowment, vol. 11, no. 3, pp. 269-282, 2017.`,
  `K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "BLEU: a Method for Automatic Evaluation of Machine Translation," in Proc. ACL, pp. 311-318, 2002.`,
  `NVIDIA, "Physical AI Autonomous Vehicles Dataset," Hugging Face, https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles, 2025.`,
  `Gemini Team, Google, "Gemini: A Family of Highly Capable Multimodal Models," arXiv:2312.11805, 2023.`,
  `J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," in Proc. NeurIPS, 2022.`,
  `윤승제, 정지원, 홍준, 임경일, 김재환, 김형주, "자율주행 차량의 학습 데이터 자동 생성 시스템 개발," 한국ITS학회 논문지, 제19권, 제5호, pp. 162-177, 2020.`,
  `이용, 장래영, 박민우, 이건우, 최명석, "자동-레이블링 기반 영상 학습데이터 제작 시스템," 한국콘텐츠학회논문지, 제21권, 제6호, pp. 701-715, 2021.`,
  `사의환, 최경수, 조성현, 김성진, "자율주행 차량의 인공지능 학습용 데이터 수집 환경 및 전처리 데이터 구축에 관한 연구," 제어로봇시스템학회 국내학술대회 논문집, pp. 536-537, 2022.`,
  `이태오, 김택국, "LoRA 기반 파인튜닝을 활용한 초등 경제 교육용 대규모 언어모델 개발 및 성능 평가," 사물인터넷융복합논문지, 제12권, 제1호, pp. 17-23, 2026.`,
  `김대건, 김남규, "LoRA 미세조정 성능 향상을 위한 커리큘럼 러닝 기반 방법론," 한국컴퓨터정보학회논문지, 제29권, 제3호, pp. 43-54, 2024.`,
  `NVlabs, "alpamayo: NVIDIA Alpamayo 1 Nano open 10B reasoning VLA model," GitHub, https://github.com/NVlabs/alpamayo, 2025.`,
];
refs.forEach((r, i) => P(`[${i + 1}] ${r}`, { hanging: true }));

// =========================================================================== 부록
BR();
H1("■ 부록 A. 사용법");
P("설치와 실행은 다음과 같다. 규칙 기반 백엔드는 추가 의존성이 없고, Gemini는 API 키, Qwen은 GPU와 torch/transformers/peft가 필요하다.");
blocks.push({ k: "codetext", text: fs.readFileSync(path.join(ROOT, "README.md"), "utf8").split("## 빠른 시작")[1].split("## 기존 데이터셋")[0].replace(/```bash|```/g, "").trim() });
H1("■ 부록 B. 원천코드");
P("핵심 모듈의 원천코드이다(테스트·UI·실험 스크립트는 저장소 참조).");
["autolabel/schemas.py", "autolabel/motion.py", "autolabel/keyframes.py", "autolabel/windows.py", "autolabel/vocab.py", "autolabel/prompt.py",
  "autolabel/backends/base.py", "autolabel/backends/rule_based.py", "autolabel/backends/gemini.py", "autolabel/backends/qwen.py", "autolabel/backends/openai_compat.py",
  "autolabel/filters.py", "autolabel/evaluate.py", "autolabel/sources.py", "autolabel/pipeline.py", "autolabel/cli.py",
  "autolabel/training/data.py", "autolabel/training/collate.py", "autolabel/training/model.py", "autolabel/training/train.py", "autolabel/training/fetch_frames.py"].forEach((f) => CODE(f, path.join(ROOT, f)));

// =========================================================================== DOCX renderer
function imgSize(file) {
  const b = fs.readFileSync(file);
  if (b[0] === 0x89 && b[1] === 0x50) return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), type: "png" };
  // jpeg: scan for SOF0/2
  let i = 2;
  while (i < b.length) {
    if (b[i] !== 0xff) { i++; continue; }
    const marker = b[i + 1];
    if (marker >= 0xc0 && marker <= 0xc3) return { h: b.readUInt16BE(i + 5), w: b.readUInt16BE(i + 7), type: "jpg" };
    i += 2 + b.readUInt16BE(i + 2);
  }
  return { w: 1600, h: 900, type: "jpg" };
}
const run = (t, o = {}) => new TextRun({ text: t, font: { ascii: EN, hAnsi: EN, eastAsia: KO, cs: KO }, size: o.size || BODY, bold: o.bold, color: o.color });
const para = (t, o = {}) => new Paragraph({ children: [run(t, o)], alignment: o.align || AlignmentType.JUSTIFIED, spacing: { after: o.after ?? 120, line: o.line ?? 300 }, indent: o.hanging ? { left: 600, hanging: 600 } : undefined, keepNext: o.keepNext });
const border = { style: BorderStyle.SINGLE, size: 4, color: "9ca3af" };
function table(cap, header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, head) => new TableCell({
    width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    shading: head ? { type: ShadingType.CLEAR, fill: "e5e7eb", color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ children: [run(t, { size: 18, bold: head })], alignment: AlignmentType.LEFT, spacing: { after: 0, line: 240 } })],
  });
  return [
    para(cap, { align: AlignmentType.CENTER, size: 19, keepNext: true, after: 60 }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED,
      borders: { top: border, bottom: border, left: border, right: border, insideHorizontal: border, insideVertical: border },
      rows: [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, widths[i], true)) })].concat(rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, widths[i], false)) }))),
    }),
    para("", { after: 120 }),
  ];
}
function figure(file, cap, cm) {
  const p = path.join(FIG, file);
  const s = imgSize(p);
  const wpx = Math.round(cm * 37.8), hpx = Math.round(wpx * s.h / s.w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true, children: [new ImageRun({ type: s.type, data: fs.readFileSync(p), transformation: { width: wpx, height: hpx } })] }),
    para(cap, { align: AlignmentType.CENTER, size: 19, after: 200 }),
  ];
}
function codeParas(text, size = 14) {
  return text.split("\n").map((l) => new Paragraph({ children: [new TextRun({ text: l.replace(/\t/g, "    ") || " ", font: { ascii: MONO, hAnsi: MONO, eastAsia: KO }, size })], spacing: { after: 0, line: 220 } }));
}
function cover() {
  const c = (t, o = {}) => para(t, { align: AlignmentType.CENTER, ...o });
  const r = (t, o = {}) => para(t, { align: AlignmentType.RIGHT, ...o });
  return [
    para("붙임2", { size: 20, align: AlignmentType.LEFT }),
    para("", { after: 400 }),
    c("2026 학년도 제 2 학기", { size: 28 }),
    para("", { after: 200 }),
    c("연구논문/작품 중간보고서", { size: 40, bold: true }),
    para("", { after: 600 }),
    ...table("", ["항목", "내용"], [
      ["제목", "VLM과 LoRA를 활용한 자율주행 데이터셋 Autolabelling"],
      ["논문/작품", "○ 논문(   )  작품( ✓ )"],
      ["GitHub URL", "https://github.com/churrosboy/Autolabel"],
      ["평가등급 (A, B, F 중 택1, 지도교수가 부여)", ""],
      ["지도교수 수정보완 사항", "○\n○\n○"],
    ], [3200, 6160]),
    para("", { after: 400 }),
    c("팀원 명단", { size: 24, bold: true }),
    c("김동환 (인)  (학번: 2021313973)", { size: 24 }),
    para("", { after: 800 }),
    c("2026 년  9 월  27 일", { size: 24 }),
    para("", { after: 400 }),
    r("지도교수 :  허재필   서명            ", { size: 24 }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}
function renderDocx() {
  const children = [];
  for (const b of blocks) {
    if (b.k === "cover") children.push(...cover());
    else if (b.k === "h1") children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 }, keepNext: true, children: [run(b.t, { size: 30, bold: true })] }));
    else if (b.k === "h2") children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 }, keepNext: true, children: [run(b.t, { size: 25, bold: true })] }));
    else if (b.k === "h3") children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 160, after: 80 }, keepNext: true, children: [run(b.t, { size: 22, bold: true })] }));
    else if (b.k === "p") children.push(para(b.t, b));
    else if (b.k === "ul") b.items.forEach((t) => children.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [run(t)], alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 300 } })));
    else if (b.k === "table") children.push(...table(b.cap, b.header, b.rows, b.widths));
    else if (b.k === "fig") children.push(...figure(b.file, b.cap, b.cm));
    else if (b.k === "br") children.push(new Paragraph({ children: [new PageBreak()] }));
    else if (b.k === "codetext") children.push(...codeParas(b.text, 16));
    else if (b.k === "code") {
      children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 60 }, keepNext: true, children: [run(b.title, { size: 20, bold: true })] }));
      children.push(...codeParas(fs.readFileSync(b.file, "utf8"), 13));
    }
  }
  const doc = new Document({
    creator: "김동환", title: "연구논문/작품 중간보고서 - VLM과 LoRA를 활용한 자율주행 데이터셋 Autolabelling",
    styles: { default: { document: { run: { font: { ascii: EN, hAnsi: EN, eastAsia: KO }, size: BODY } } } },
    numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 480, hanging: 240 } } } }] }] },
    sections: [{
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1247, right: 1247 } } },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: { ascii: EN, hAnsi: EN, eastAsia: KO }, size: 18 })] })] }) },
      children,
    }],
  });
  return Packer.toBuffer(doc);
}

// =========================================================================== Markdown renderer
function renderMd() {
  const out = [];
  for (const b of blocks) {
    if (b.k === "cover") out.push("# 연구논문/작품 중간보고서 (2026학년도 제2학기)\n\n- 제목: VLM과 LoRA를 활용한 자율주행 데이터셋 Autolabelling\n- 논문/작품: 작품\n- GitHub URL: https://github.com/churrosboy/Autolabel\n- 팀원: 김동환 (2021313973)\n- 지도교수: 허재필\n- 2026년 9월 27일\n");
    else if (b.k === "h1") out.push(`\n## ${b.t.replace(/^■\s*/, "")}\n`);
    else if (b.k === "h2") out.push(`\n### ${b.t}\n`);
    else if (b.k === "h3") out.push(`\n#### ${b.t}\n`);
    else if (b.k === "p") out.push(b.t + "\n");
    else if (b.k === "ul") out.push(b.items.map((t) => `- ${t}`).join("\n") + "\n");
    else if (b.k === "table") out.push(`**${b.cap}**\n\n| ${b.header.join(" | ")} |\n| ${b.header.map(() => "---").join(" | ")} |\n${b.rows.map((r) => `| ${r.map((c) => c.replace(/\n/g, " ")).join(" | ")} |`).join("\n")}\n`);
    else if (b.k === "fig") out.push(`![${b.cap}](figures/${b.file})\n\n*${b.cap}*\n`);
    else if (b.k === "codetext") out.push("```bash\n" + b.text + "\n```\n");
    else if (b.k === "code") out.push(`\n#### ${b.title}\n\n\`\`\`python\n${fs.readFileSync(b.file, "utf8")}\n\`\`\`\n`);
  }
  return out.join("\n");
}

(async () => {
  const outDir = path.join(ROOT, "report");
  fs.writeFileSync(path.join(outDir, "중간보고서_Autolabel.docx"), await renderDocx());
  fs.writeFileSync(path.join(outDir, "중간보고서_Autolabel.md"), renderMd());
  const bodyChars = blocks.filter((b) => ["p", "ul"].includes(b.k)).reduce((a, b) => a + (b.t ? b.t.length : b.items.join("").length), 0);
  console.log(`written: report/중간보고서_Autolabel.docx, .md  (body ≈ ${bodyChars} chars, ${figN} figures, ${tabN} tables)`);
})();
