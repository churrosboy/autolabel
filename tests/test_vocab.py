from autolabel.vocab import (
    CAUSAL_CONNECTIVES,
    STOP_WORDS,
    extract_components,
    extract_decisions,
)


def flat(text):
    d = extract_decisions(text)
    return d["longitudinal"], d["lateral"]


def test_canonical_terms_are_found():
    lon, lat = flat("The vehicle performs stop for static constraints and lane keeping & centering because the light is red.")
    assert lon == ["stop for static constraints"]
    assert lat == ["lane keeping & centering"]


def test_paraphrases_map_to_canonical():
    assert flat("The ego vehicle decelerates to a complete stop behind a stationary lead vehicle.")[0][:1] == ["stop for static constraints"]
    assert "lead obstacle following" in flat("It maintains a safe following distance to the lead vehicle.")[0]
    assert flat("It changes lanes to the left because of a stopped truck.")[1] == ["lane change"]
    assert flat("The vehicle executes a planned right turn onto the new road.")[1] == ["turn"]
    assert flat("It accelerates to track its set speed.")[0] == ["set speed tracking"]


def test_history_context_is_ignored():
    lon, _ = flat("Because the ego vehicle has stopped at the stop sign and the intersection is clear, it engages in set speed tracking.")
    assert lon == ["set speed tracking"]
    lon, _ = flat("After yielding to the pedestrian, the vehicle resumes its target speed.")
    assert "yield" not in lon and "set speed tracking" in lon


def test_turn_intent_is_not_a_turn():
    _, lat = flat("The ego vehicle performs speed adaptation to prepare for a planned right turn at the approaching intersection.")
    assert lat == []
    _, lat = flat("The ego vehicle turns right onto the side street.")
    assert lat == ["turn"]


def test_stop_sign_is_not_a_stop_verb():
    assert not STOP_WORDS.search("It approaches an intersection with a stop sign and keeps its set speed.")
    assert STOP_WORDS.search("It comes to a complete stop at the line.")
    assert STOP_WORDS.search("The vehicle stops for the red light.")


def test_components_and_connectives():
    text = "Because the traffic light is red and a pedestrian is crossing, the vehicle yields."
    comps = extract_components(text)
    assert "traffic lights" in comps and "critical objects" in comps
    assert CAUSAL_CONNECTIVES.search(text)
    assert not CAUSAL_CONNECTIVES.search("The vehicle performs lane keeping & centering.")
