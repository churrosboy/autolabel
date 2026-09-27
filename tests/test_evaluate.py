from autolabel.evaluate import bleu, compare, evaluate_pairs, rouge_l


def test_identical_text_scores_one():
    t = "The vehicle performs stop for static constraints because the light is red."
    assert abs(rouge_l(t, t) - 1.0) < 1e-9
    assert bleu(t, t) > 0.99
    m = compare(t, t)
    assert m["decision_f1"] == 1.0 and m["lon_match"] == 1.0


def test_decision_overlap_partial():
    pred = "The vehicle performs stop for static constraints and lane keeping & centering because of the red light."
    ref = "The vehicle yields and performs lane keeping & centering because a pedestrian is crossing."
    m = compare(pred, ref)
    assert 0 < m["decision_f1"] < 1
    assert m["lat_match"] == 1.0 and m["lon_match"] == 0.0


def test_evaluate_pairs_groups():
    pairs = [("The vehicle performs a turn because of routing intent.", "The vehicle turns left due to routing intent.")] * 3
    res = evaluate_pairs(pairs, by=["steer_l", "steer_l", "steer_r"])
    assert res["overall"]["n"] == 3
    assert set(res["by_group"]) == {"steer_l", "steer_r"}
    assert res["overall"]["decision_f1"] == 1.0
