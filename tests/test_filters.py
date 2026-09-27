from autolabel.filters import FilterConfig, QualityFilter
from autolabel.keyframes import detect_keyframes
from autolabel.windows import make_windows


def _window(ego, kf_type_prefix):
    kfs = [k for k in detect_keyframes(ego) if k.type.startswith(kf_type_prefix)]
    assert kfs, f"no {kf_type_prefix} keyframe"
    return make_windows("clip", ego, kfs[:1])[0]


def test_clean_label_passes(ego_brake):
    w = _window(ego_brake, "gentle_decel") if any(k.type == "gentle_decel" for k in detect_keyframes(ego_brake)) else _window(ego_brake, "strong_decel")
    r = QualityFilter().check(
        "The vehicle performs stop for static constraints because the red traffic light ahead required this action to stop safely at the line.",
        w.keyframe_type, w,
    )
    assert r.verdict == "pass", r.to_dict()
    assert r.decisions["longitudinal"] == ["stop for static constraints"]


def test_structural_errors():
    qf = QualityFilter()
    assert qf.check("INSUFFICIENT_EVIDENCE").verdict == "reject"
    assert "S05" in qf.check("The ego vehicle drives along the road because it is sunny.").codes
    assert "S08" in qf.check("The vehicle performs acceleration for passing and stop for static constraints because the light is red.").codes
    assert "S04" in qf.check("차량은 신호등이 빨간색이라 정지선에서 stop for static constraints 를 수행한다 because.").codes
    assert "S09" in qf.check("The vehicle maybe performs lane keeping & centering because the lane is possibly clear.").codes
    assert "S10" in qf.check("The vehicle performs a turn because frame 12 shows the intersection.").codes


def test_keyframe_type_contradiction():
    r = QualityFilter().check("The vehicle performs acceleration for passing because the lead truck is slow.", "gentle_decel")
    assert "T01" in r.codes and r.verdict == "reject"


def test_kinematic_stop_claim_without_deceleration(ego_cruise):
    from autolabel.keyframes import uniform_keyframes

    w = make_windows("clip", ego_cruise, uniform_keyframes(ego_cruise, 8.0)[:1])[0]
    r = QualityFilter().check("The vehicle performs stop for static constraints because the traffic light is red.", "go_straight", w)
    assert "K01" in r.codes and r.verdict == "reject"


def test_kinematic_turn_claim_matches_turn(ego_turn):
    w = _window(ego_turn, "steer")
    ok = QualityFilter().check("The vehicle performs a left turn because the routing intent required turning at the intersection.", w.keyframe_type, w)
    assert "K03" not in ok.codes
    bad = QualityFilter().check("The vehicle performs lane keeping & centering because the road ahead is clear.", w.keyframe_type, w)
    assert "K04" in bad.codes


def test_duplicate_detection():
    r = QualityFilter().check("The vehicle performs lane keeping & centering because the lane is clear.", siblings=["the vehicle performs lane keeping & centering because the lane is clear"])
    assert "S13" in r.codes


def test_disabled_codes_and_score():
    qf = QualityFilter(FilterConfig(disabled={"S12", "S07"}))
    r = qf.check("The vehicle performs lane keeping & centering.")
    assert "S12" not in r.codes and "S07" not in r.codes
    assert 0.0 <= r.score <= 1.0
