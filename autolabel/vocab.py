"""Controlled vocabulary for CoC labels (Alpamayo-R1, Table 1 & Table 2).

Everything that reads or checks a CoC sentence goes through this module so the
prompt, the quality filter and the evaluator agree on the same canonical terms.
"""
from __future__ import annotations

import re
from typing import Iterable

# --------------------------------------------------------------------------- #
# Table 1: driving decisions
# --------------------------------------------------------------------------- #
LONGITUDINAL = [
    "set speed tracking",
    "lead obstacle following",
    "speed adaptation",
    "gap-searching",
    "acceleration for passing",
    "yield",
    "stop for static constraints",
]
LATERAL = [
    "lane keeping & centering",
    "merge / split",
    "out-of-lane nudge",
    "in-lane nudge",
    "lane change",
    "pull-over / curb approach",
    "turn",
    "lateral maneuver abort",
]
DECISIONS = LONGITUDINAL + LATERAL

DECISION_DEFINITIONS = {
    "set speed tracking": "Maintain/reach target speed when unconstrained.",
    "lead obstacle following": "Maintain safe gap to lead entity moving in same traffic flow.",
    "speed adaptation": "Adjust speed for road features (curves, bumps, etc.), independent of a lead.",
    "gap-searching": "Adjust speed to match target stream for planned lateral maneuver.",
    "acceleration for passing": "Increase speed to pass slower lead with lateral plan.",
    "yield": "Slow/stop to concede priority to agents (pedestrians, cross-traffic, cut-ins).",
    "stop for static constraints": "Decelerate/hold at control points (stop lines, red lights).",
    "lane keeping & centering": "Maintain position within lane boundaries.",
    "merge / split": "Transition between facilities (on-ramp, weave segments).",
    "out-of-lane nudge": "Brief intentional lane-line crossing to clear hazard, then return.",
    "in-lane nudge": "Temporary offset within lane to clear hazard.",
    "lane change": "Full adjacent-lane transition with gap negotiation.",
    "pull-over / curb approach": "Move toward edge/shoulder or stop area.",
    "turn": "Planned path onto different road segment with heading change.",
    "lateral maneuver abort": "Cancel ongoing lateral maneuver and re-center.",
}

# Regex patterns (case-insensitive) that map free text -> canonical decision.
# Order matters only within a category: the first pattern that matches wins,
# but several decisions may match a single sentence.
_DECISION_PATTERNS: dict[str, list[str]] = {
    "set speed tracking": [
        r"set[- ]speed[- ]tracking",
        r"maintain(?:s|ing|ed)? (?:its |the |a )?(?:target |set |current |constant |cruising )?speed",
        r"resum(?:e|es|ing|ed) (?:its |the )?(?:target |set |cruising )?speed",
        r"accelerat\w* (?:back )?(?:up )?to (?:its |the )?(?:target |set |desired |cruising )?speed",
        r"reach(?:es|ing)? (?:its |the )?(?:target |set |desired )?speed",
        r"proceed\w* at (?:its |the )?(?:target |set |current )?speed",
        r"track\w* (?:its |the )?(?:set |target |desired )?speed",
        r"\bset speed\b",
        r"accelerat\w* (?:to )?(?:proceed|continue) (?:straight|forward|through)",
        r"\baccelerat(?:es|ing|e)\b(?! (?:for passing|to pass|to overtake|in order to pass))",
    ],
    "lead obstacle following": [
        r"lead[- ]obstacle[- ]following",
        r"lead[- ]vehicle[- ]following",
        r"follow(?:s|ing|ed)? (?:the |a |its )?(?:lead|leading|preceding|slower|front)\b",
        r"follow(?:s|ing|ed)? (?:the |a )?vehicle (?:ahead|in front)",
        r"maintain(?:s|ing|ed)? (?:a |an )?(?:safe |adequate |appropriate )?(?:following )?(?:gap|distance|headway)",
        r"keep(?:s|ing)? (?:a |an )?(?:safe )?(?:distance|gap) (?:from|to|behind)",
        r"(?:stop|slow|brak|decelerat|wait)\w* (?:down )?(?:behind|for) (?:the |a |an )?(?:stationary |stopped |slowing |slow |slower |queued |lead |leading )*(?:vehicle|car|truck|bus|van|traffic)",
        r"\bqueue\w* behind",
    ],
    "speed adaptation": [
        r"speed[- ]adaptation",
        r"adapt(?:s|ing|ed)? (?:its |the )?speed",
        r"adjust(?:s|ing|ed)? (?:its |the )?speed (?:for|to|in)",
        r"slow(?:s|ing|ed)? (?:down )?(?:for|to (?:safely )?(?:navigate|negotiate|handle|take)) (?:the |a |an )?(?:curve|bend|turn|bump|intersection|roundabout|corner|wet)",
        r"(?:decelerat|slow|brak)\w* (?:down )?(?:for|to prepare for|in preparation for|ahead of|before|to (?:safely )?(?:navigate|negotiate|execute|make|take|perform)) (?:the |a |an |its )?(?:planned |upcoming |sharp |gradual |left |right |right-hand |left-hand )*(?:turn|curve|bend|corner|roundabout|bump|intersection|road curvature|curvature)",
        r"(?:decelerat|slow|reduc)\w* (?:its |the )?(?:speed )?(?:for|due to|because of) (?:the |a |an )?(?:road |upcoming )?(?:curvature|curve|bend|bump|narrowing|wet road|geometry)",
    ],
    "gap-searching": [
        r"gap[- ]search\w*",
        r"search(?:es|ing)? for (?:a |an )?(?:suitable |adequate )?gap",
    ],
    "acceleration for passing": [
        r"accelerat\w* for passing",
        r"accelerat\w* (?:in order )?to (?:pass|overtake)",
        r"\bovertak\w*",
        r"\bpass(?:es|ing)? (?:the |a )?(?:slower |slow |stopped |parked )?(?:lead |leading )?(?:vehicle|car|truck|bus)",
    ],
    "yield": [
        r"\byield\w*",
        r"giv(?:e|es|ing) way",
        r"conced\w* (?:priority|right[- ]of[- ]way)",
        r"(?:stop|slow|wait)\w* (?:for|to let) (?:the |a |an )?(?:pedestrian|cyclist|crossing|cross[- ]traffic|oncoming|cut[- ]in)",
    ],
    "stop for static constraints": [
        r"stop\w* for (?:the |a )?static constraints?",
        r"\bstop(?:s|ping)? (?:at|for|before|due to) (?:the |a |an )?(?:red (?:traffic )?light|red signal|stop line|stop sign|traffic (?:light|signal)|crosswalk|stop[- ]line|intersection)",
        r"compl(?:y|ies|ying) with (?:the |a )?(?:stop sign|stop line|red (?:traffic )?light|traffic (?:light|signal))",
        r"(?:decelerat|slow|brak)\w* (?:down )?(?:for|at|due to|because of) (?:the |a |an )?(?:red (?:traffic )?light|red signal|stop sign|stop line)",
        r"(?:come|comes|coming|came|decelerat\w*|slow\w*|brak\w*) to a (?:complete |full |controlled )?stop\b",
        r"to a (?:complete |full |controlled )?stop\b",
        r"hold\w* (?:its )?position at (?:the |a )?(?:stop line|light|intersection)",
        r"remain\w* (?:stopped|stationary|at a standstill)",
        r"(?:full|complete) stop\b",
        r"\bstop(?:s|ping)?\b(?! sign| line|-line| ?controlled| signal| light)(?=.*\b(?:red (?:traffic )?light|stop sign|stop line|red signal|traffic (?:light|signal))\b)",
    ],
    "lane keeping & centering": [
        r"lane[- ]keeping",
        r"lane[- ]cent(?:e|re)ring",
        r"keep(?:s|ing)? (?:to |within )?(?:its |the |the current )?lane",
        r"stay(?:s|ing)? (?:in|within|centered in) (?:its |the |the current )?lane",
        r"maintain(?:s|ing|ed)? (?:its |the )?(?:current )?(?:lane|position within (?:its |the )?lane|trajectory|heading|course)",
        r"continu(?:e|es|ing) (?:straight|in its lane|along its lane|forward)",
        r"proceed(?:s|ing)? straight",
        r"remain\w* (?:in|within|centered in) (?:its |the )?lane",
        r"centered in (?:its |the )?lane",
    ],
    "merge / split": [
        r"\bmerg(?:e|es|ing|ed)\b",
        r"merge ?/ ?split",
        r"\bsplit\b",
        r"on[- ]ramp|off[- ]ramp|weav(?:e|es|ing)",
    ],
    "out-of-lane nudge": [
        r"out[- ]of[- ]lane[- ]nudge",
        r"nudg\w* (?:out of|outside|beyond|across) (?:its |the )?lane",
        r"cross(?:es|ing)? (?:the )?lane (?:line|marking)",
    ],
    "in-lane nudge": [
        r"in[- ]lane[- ]nudge",
        r"nudg\w* (?:within|inside|in) (?:its |the )?lane",
        r"(?:slight|small|brief|minor|lateral)\w* (?:offset|shift|nudge)",
        r"\bnudg\w*",
    ],
    "lane change": [
        r"lane[- ]chang\w*",
        r"chang(?:e|es|ing|ed) (?:into |to )?(?:the |a )?(?:left |right |adjacent )?lanes?",
        r"mov(?:e|es|ing|ed) (?:in)?to (?:the |an? )?(?:adjacent|left|right|next|neighbo\w+) lane",
        r"(?:enter|shift)(?:s|ing|ed)? (?:in)?to (?:the |an? )?(?:adjacent|left|right|next|neighbo\w+) lane",
        r"transition\w* (?:in)?to (?:the |an? )?(?:adjacent|left|right|next) lane",
    ],
    "pull-over / curb approach": [
        r"pull(?:s|ing|ed)?[- ]?over",
        r"curb[- ]approach",
        r"approach\w* (?:the )?(?:curb|shoulder|kerb)",
        r"mov(?:e|es|ing)? (?:toward|towards|to) (?:the )?(?:shoulder|curb|kerb|road edge|side of the road)",
    ],
    "turn": [
        r"(?<!in )\b(?:left|right|u)[- ]turn(?:s|ing)?\b",
        r"\bturn(?:s|ing|ed)? (?:left|right|onto|into|at|toward|towards|to the (?:left|right))\b",
        r"execut\w* (?:a |the )?(?:planned |sharp |gradual |left |right )?turn",
        r"perform\w* (?:a |the )?(?:planned |sharp |gradual |left |right )?turn",
        r"(?:complet|initiat|mak|negotiat|navigat)\w* (?:a |the |its )?(?:planned |sharp |gradual |left |right )?turn\b",
        r"\[turn\]",
        r"\bturn(?:s|ing|ed)?\b(?! signal| lane| indicator| ?-?taking)",
    ],
    "lateral maneuver abort": [
        r"maneuver[- ]abort",
        r"abort\w* (?:the |its |an? )?(?:lateral |lane[- ]change |ongoing )?(?:maneuver|lane change)",
        r"re[- ]?cent(?:e|re)r\w* after abort",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in _DECISION_PATTERNS.items()
}


def category_of(decision: str) -> str:
    if decision in LONGITUDINAL:
        return "longitudinal"
    if decision in LATERAL:
        return "lateral"
    return "unknown"


# A decision mention preceded by one of these is *history/context*, not the decision
# taken after the keyframe ("after stopping at the stop sign, it accelerates...").
_HISTORY_PREFIX = re.compile(
    r"(?:\b(?:after|having|has|have|had|was|were|been|already|once|since|following|previously|initially|while)\s+(?:been\s+|just\s+|previously\s+|initially\s+|fully\s+|completely\s+)?"
    r"|\bfrom (?:a |the )?(?:complete |full )?)\s*$",
    re.I,
)
# "prepare for a planned right turn" is routing intent, not a turn being executed
_TURN_INTENT_PREFIX = re.compile(
    r"(?:prepar\w* (?:for|to)|in anticipation of|anticipat\w*|approach\w*|intend\w* (?:to )?(?:make |execute |perform )?|upcoming|before (?:making |the |executing |initiating )?|ahead of|designated|planned upcoming|prior to|for (?:an? |the )?(?:upcoming |planned |imminent )?)\s*(?:an? |the |its )?(?:planned |upcoming |imminent |sharp |gradual |gentle |left |right |slight )*$",
    re.I,
)


def _is_history(text: str, start: int) -> bool:
    prefix = text[max(0, start - 40):start]
    return bool(_HISTORY_PREFIX.search(prefix))


def _is_turn_intent(text: str, start: int) -> bool:
    prefix = text[max(0, start - 60):start]
    return bool(_TURN_INTENT_PREFIX.search(prefix)) or bool(re.search(r"^(?:s|ing|ed)?\s+(?:lane|signal|indicator)\b", text[start + 4:start + 14], re.I))


def _first_valid_match(decision: str, patterns: list[re.Pattern], text: str):
    best = None
    for pat in patterns:
        for m in pat.finditer(text):
            if _is_history(text, m.start()):
                continue
            if decision == "turn" and _is_turn_intent(text, m.start()):
                continue
            if best is None or m.start() < best:
                best = m.start()
            break
    return best


def extract_decisions(text: str) -> dict[str, list[str]]:
    """Return canonical decisions found in ``text`` grouped by category.

    Longitudinal / lateral lists keep the order of first appearance in the
    sentence so the "primary" decision is the first element.  Mentions that
    are clearly *history* ("after stopping at the stop sign, ...") or *intent*
    ("prepares for a planned right turn") are ignored.
    """
    found: list[tuple[int, str]] = []
    for decision, patterns in _COMPILED.items():
        best = _first_valid_match(decision, patterns, text)
        if best is not None:
            found.append((best, decision))
    found.sort()
    out = {"longitudinal": [], "lateral": []}
    for _, d in found:
        out[category_of(d)].append(d)
    # "in-lane nudge" generic pattern also fires on "out-of-lane nudge"
    if "out-of-lane nudge" in out["lateral"] and "in-lane nudge" in out["lateral"]:
        if not re.search(r"in[- ]lane[- ]nudge|nudg\w* (?:within|inside|in) ", text, re.I):
            out["lateral"].remove("in-lane nudge")
    # "turn" generic pattern fires inside "lane change ... turn signal" etc.; keep as-is
    return out


# --------------------------------------------------------------------------- #
# Table 2: critical components
# --------------------------------------------------------------------------- #
COMPONENTS = [
    "critical objects",
    "traffic lights",
    "yield/stop control",
    "road events",
    "lane/lanelines",
    "routing intent",
    "odd constraints",
]

_COMPONENT_PATTERNS: dict[str, str] = {
    "critical objects": r"\b(?:vehicle|car|truck|bus|van|suv|pedestrian|cyclist|bicycle|bike|motorcycl\w*|scooter|agent|obstacle|lead|cut[- ]in|oncoming|cross[- ]traffic|traffic ahead|animal|debris|cone)s?\b",
    "traffic lights": r"\b(?:traffic (?:light|signal)|red (?:light|signal)|green (?:light|signal|arrow)|yellow (?:light|signal)|amber|signal turn\w*|light turn\w*)\b",
    "yield/stop control": r"\b(?:stop sign|yield sign|stop line|yield line|give[- ]way|all[- ]way stop|crosswalk|zebra)\b",
    "road events": r"\b(?:curve|curvature|bend|corner|speed bump|bump|hump|narrow\w*|intersection|junction|roundabout|ramp|slope|hill|uphill|downhill|road geometry|road feature|wet road|pothole)\b",
    "lane/lanelines": r"\b(?:lane|lanes|laneline|lane line|lane marking|lane boundary|dashed line|solid line|road marking|median|divider)s?\b",
    "routing intent": r"\b(?:routing|route|intend\w*|intent|planned|plan|navigation|destination|target lane|to turn (?:left|right)|heading (?:to|toward|left|right)|through)\b",
    "odd constraints": r"\b(?:weather|rain\w*|wet|snow\w*|fog\w*|ice|icy|night|dark\w*|glare|low visibility|construction|work zone|school bus|school zone|emergency vehicle|parking lot|gravel)\b",
}
_COMPONENT_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in _COMPONENT_PATTERNS.items()}


def extract_components(text: str) -> list[str]:
    return [k for k, pat in _COMPONENT_COMPILED.items() if pat.search(text)]


# --------------------------------------------------------------------------- #
# Keyframe types (from ego-motion) and what decisions they are compatible with
# --------------------------------------------------------------------------- #
KEYFRAME_TYPES = [
    "strong_accel", "gentle_accel", "gentle_decel", "strong_decel", "stop",
    "sharp_steer_l", "sharp_steer_r", "steer_l", "steer_r", "go_straight",
]

KEYFRAME_CATEGORY = {
    **{t: "longitudinal" for t in ["strong_accel", "gentle_accel", "gentle_decel", "strong_decel", "stop"]},
    **{t: "lateral" for t in ["sharp_steer_l", "sharp_steer_r", "steer_l", "steer_r", "go_straight"]},
}

# decisions we EXPECT for a keyframe type (soft) and decisions that CONTRADICT it (hard)
KEYFRAME_EXPECTED: dict[str, set[str]] = {
    "gentle_decel": {"yield", "stop for static constraints", "lead obstacle following", "speed adaptation", "gap-searching"},
    "strong_decel": {"yield", "stop for static constraints", "lead obstacle following", "speed adaptation"},
    "stop": {"stop for static constraints", "yield", "lead obstacle following"},
    "gentle_accel": {"set speed tracking", "acceleration for passing", "gap-searching", "lead obstacle following", "speed adaptation"},
    "strong_accel": {"set speed tracking", "acceleration for passing", "gap-searching"},
    "steer_l": {"turn", "lane change", "merge / split", "out-of-lane nudge", "in-lane nudge", "pull-over / curb approach", "speed adaptation"},
    "steer_r": {"turn", "lane change", "merge / split", "out-of-lane nudge", "in-lane nudge", "pull-over / curb approach", "speed adaptation"},
    "sharp_steer_l": {"turn", "lane change", "merge / split", "out-of-lane nudge"},
    "sharp_steer_r": {"turn", "lane change", "merge / split", "out-of-lane nudge"},
    "go_straight": {"lane keeping & centering", "set speed tracking", "lateral maneuver abort", "turn", "lane change"},
}
KEYFRAME_CONTRADICTS: dict[str, set[str]] = {
    "gentle_decel": {"acceleration for passing"},
    "strong_decel": {"acceleration for passing", "set speed tracking"},
    "stop": {"acceleration for passing", "set speed tracking"},
    "gentle_accel": {"stop for static constraints"},
    "strong_accel": {"stop for static constraints", "yield"},
    "steer_l": set(),
    "steer_r": set(),
    "sharp_steer_l": {"lane keeping & centering"},
    "sharp_steer_r": {"lane keeping & centering"},
    "go_straight": set(),
}

# Mutually exclusive decision pairs inside one label
CONTRADICTORY_PAIRS: list[tuple[str, str]] = [
    ("acceleration for passing", "stop for static constraints"),
    ("acceleration for passing", "yield"),
    ("set speed tracking", "stop for static constraints"),
    ("lane keeping & centering", "lane change"),
    ("lane keeping & centering", "turn"),
    ("lane keeping & centering", "merge / split"),
    ("lane keeping & centering", "out-of-lane nudge"),
    ("lane keeping & centering", "pull-over / curb approach"),
    ("lane change", "lateral maneuver abort"),
]

# Generic motion verbs (used for kinematic consistency checks)
DECEL_WORDS = re.compile(r"\b(?:decelerat\w*|slow(?:s|ing)?(?: down)?(?! lead| vehicle| traffic|-moving)|brak(?:e|es|ing)|reduc\w* (?:its |the )?speed|comes? to a (?:complete |full )?stop|(?<!has )(?<!having )(?<!after )(?<!its )(?<!their )stop(?:s|ping)?\b(?! sign| line|-line| ?controlled| signal| light)|halts?)\b", re.I)
ACCEL_WORDS = re.compile(r"\b(?:accelerat\w*|speed(?:s|ed|ing)? up|increas\w* (?:its |the )?speed|picks? up speed)\b", re.I)
STOP_WORDS = re.compile(r"\b(?:comes? to a (?:complete |full |controlled )?stop|(?<!has )(?<!having )(?<!after )(?<!its )(?<!their )stop(?:s|ping)?\b(?! sign| line|-line| ?controlled| signal| light)|halts?|remains? (?:stopped|stationary|at a standstill)|holds? (?:its )?position)\b", re.I)

HEDGE_WORDS = re.compile(r"\b(?:maybe|perhaps|possibly|probably|likely|might|may be|appears? to|seems? to|unclear|cannot (?:be )?determine\w*|not (?:clearly )?visible|hard to (?:tell|see)|uncertain|ambiguous|i think|i believe)\b", re.I)
FUTURE_EVIDENCE = re.compile(r"\b(?:frames? (?:5|6|7|8|9|1[0-6])\b|later frames?|future frames?|subsequent frames?|outcome frames?|stage (?:ii|2))", re.I)
META_LEAK = re.compile(r"(?:THOUGHT_PROCESS|FINAL_COC|Stage [12I]+|\bframes? \d|\[[^\]]*<[^\]]*\]|<[^>]{2,40}>)", re.I)
CAUSAL_CONNECTIVES = re.compile(
    r"\b(?:because|since|due to|as a result|owing to|in response to|given(?: that)?|so (?:that|it|the)|in order to|therefore|thus|hence|"
    r"in anticipation of|to prepare for|guided by|observing|noting|detecting|seeing|after (?:observing|noting|detecting|seeing|waiting|yielding|stopping|confirming|checking)|"
    r"dictat\w*|necessitat\w*|prompt\w*|trigger\w*|caus\w*|lead\w* (?:it |the vehicle )?to|result\w* in|forc\w*|requir\w*|as (?:the|a|an|it|there)\b|for (?:lead|speed|set|gap|lane|yield|stop|static|following|passing|merging|turning)|decision to|"
    r"to (?:safely |gently |gradually )?(?:avoid|maintain|ensure|prepare|comply|respect|allow|let|clear|follow|keep|reach|make|execute|proceed|navigate|negotiate|stop|yield|complete|pass|merge|continue|slow|adapt|adjust|hold|remain|stay|come|perform|accommodate|conform|resume|reduce|meet|match|transition|handle|track|wait|respond|initiate|approach|enter|exit|cross|travel|drive|go|turn|decelerate|accelerate|brake|give|concede|manage|preserve|secure|create|obtain|gain|prevent|halt|position|react|respond|clear)\b)",
    re.I,
)


def compact_decision_list(decisions: dict[str, list[str]]) -> list[str]:
    return list(decisions.get("longitudinal", [])) + list(decisions.get("lateral", []))


def all_terms() -> Iterable[str]:
    return DECISIONS
