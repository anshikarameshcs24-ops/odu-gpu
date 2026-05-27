"""Label metadata for CMAI behaviour detection."""

CMAI_BEHAVIOURS = {
    0: "pacing_aimless_locomotion",
    1: "inappropriate_robing_disrobing",
    2: "trying_to_get_to_a_different_place",
    3: "intentional_falling",
    4: "eating_inappropriate_substances",
    5: "handling_things_inappropriately",
    6: "hiding_hoarding_things",
    7: "performing_repetitive_mannerisms",
    8: "general_restlessness",
    9: "repetitive_sentences_or_questions",
    10: "strange_noises",
    11: "constant_unwarranted_requests_for_attention",
    12: "negativism",
    13: "complaining",
    14: "repetitive_vocalisations",
    15: "hitting_including_self",
    16: "kicking",
    17: "grabbing",
    18: "pushing",
    19: "throwing_things",
    20: "biting",
    21: "scratching",
    22: "spitting",
    23: "screaming",
    24: "making_verbal_sexual_advances",
    25: "cursing_or_verbal_aggression",
    26: "tearing_things_or_destroying_property",
    27: "physical_sexual_advances",
    28: "general_agitation_unclassified",
}

NUM_CMAI_CLASSES = 29

CMAI_CATEGORIES = {
    "physically_non_aggressive": list(range(0, 9)),
    "verbally_non_aggressive": list(range(9, 15)),
    "physically_aggressive": list(range(15, 23)),
    "verbally_aggressive": list(range(23, 29)),
}

AGITATION_RISK_LEVELS = {
    0: "low",
    1: "moderate",
    2: "high",
    3: "imminent",
}

EARLY_WARNING_BEHAVIOURS = {0, 7, 8, 9, 12, 13, 14}

