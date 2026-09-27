from autolabel.keyframes import KeyframeConfig, detect_keyframes, uniform_keyframes
from autolabel.windows import make_windows


def types(kfs):
    return [k.type for k in kfs]


def test_brake_profile_yields_decel_stop_accel(ego_brake):
    kfs = detect_keyframes(ego_brake, KeyframeConfig(per_window_cap=False))
    t = types(kfs)
    assert "gentle_decel" in t or "strong_decel" in t
    assert "stop" in t
    assert "gentle_accel" in t or "strong_accel" in t
    # keyframe precedes the actual braking start (8 s) by the lead time
    decel = next(k for k in kfs if k.type.endswith("decel"))
    assert 6.5e6 <= decel.timestamp_us <= 8.5e6


def test_turn_profile_yields_steer_then_straight(ego_turn):
    kfs = detect_keyframes(ego_turn, KeyframeConfig(per_window_cap=False))
    t = types(kfs)
    assert any(x.startswith("steer_l") or x.startswith("sharp_steer_l") for x in t)
    assert "go_straight" in t


def test_cruise_has_no_keyframes_and_uniform_fallback(ego_cruise):
    assert detect_keyframes(ego_cruise) == []
    u = uniform_keyframes(ego_cruise, every_s=8.0)
    assert len(u) >= 2 and all(k.type == "go_straight" for k in u)


def test_per_window_cap_limits_events(ego_brake):
    many = detect_keyframes(ego_brake, KeyframeConfig(per_window_cap=False, cooldown_s=0))
    capped = detect_keyframes(ego_brake, KeyframeConfig(per_window_cap=True))
    assert len(capped) <= len(many)
    # at most one longitudinal per 8 s bucket
    buckets = {}
    for k in capped:
        b = buckets.setdefault(k.timestamp_us // 8_000_000, [])
        b.append(k.category)
    assert all(v.count("longitudinal") <= 1 for v in buckets.values())


def test_speed_confirmation_rejects_fake_decel(ego_cruise):
    # inject a negative acceleration blip without any speed change
    ego = ego_cruise
    ego.accelerations[80:95, 0] = -2.0
    with_confirm = detect_keyframes(ego, KeyframeConfig(confirm_with_speed=True))
    without = detect_keyframes(ego, KeyframeConfig(confirm_with_speed=False))
    assert "gentle_decel" in types(without)
    assert "gentle_decel" not in types(with_confirm)


def test_windows_are_clamped_and_have_16_frames(ego_brake):
    kfs = detect_keyframes(ego_brake)
    wins = make_windows("clip", ego_brake, kfs)
    assert wins
    for w in wins:
        assert len(w.frame_timestamps_us) == 16
        assert w.start_us >= ego_brake.timestamps_us[0]
        assert w.end_us <= ego_brake.timestamps_us[-1]
        assert w.ego.n > 50
