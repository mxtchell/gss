"""
GSS Explorer Configuration
Metric groups, dimensions, and labels from dataset misc_info
"""
import os

DATABASE_ID = os.getenv('DATABASE_ID', 'B855F1B7-35EA-46E1-B1D7-1630EEA5CA82')
TABLE_NAME = "read_csv('gss_max_ready_2026_v6.csv')"

# Metric groups from dataset misc_info
METRIC_GROUPS = {
    "sexual_activity": [
        "sexually_active_hq4_flag",
        "explicit_inactive_flag",
        "chosen_not_to_have_sex"
    ],
    "relationship": [
        "is_committed_relationship"
    ],
    "satisfaction": [
        "current_sex_life_satisfied_t2b",
        "current_sex_life_not_satisfied",
        "first_time_satisfaction_t2b"
    ],
    "emotions_first_time": [
        "emotion_confident_first_time",
        "emotion_proud_first_time",
        "emotion_ashamed_first_time"
    ],
    "benefits": [
        "benefit_better_mood",
        "benefit_feel_happier",
        "benefit_feel_healthier",
        "benefit_more_confident",
        "benefit_sleep_better"
    ],
    "lube_reasons": [
        "lube_reason_not_needed"
    ],
    "fe9_agreement_t3b": [
        "fe9_confident_that_you_know_about_what_to_expect_when_having",
        "fe9_having_sex_for_the_first_time_might_be_painful_for_me_ag",
        "fe9_i_am_concerned_about_becoming_pregnant_making_my_partner",
        "fe9_i_am_concerned_about_catching_an_sti_std_agreement_how_y",
        "fe9_i_am_looking_forward_to_having_sex_when_the_time_is_righ",
        "fe9_i_am_not_entirely_sure_of_all_the_things_that_one_does_w",
        "fe9_i_feel_nervous_scared_about_it_agreement_how_you_may_vie",
        "fe9_i_feel_well_informed_and_confident_agreement_how_you_may",
        "fe9_i_know_what_to_do_to_protect_myself_and_my_partner_from_",
        "fe9_i_worry_it_will_be_painful_for_me_or_my_partner_agreemen"
    ],
    "numeric": [
        "age_first_sex_numeric"
    ]
}

METRIC_GROUP_LABELS = {
    "sexual_activity": "Sexual Activity",
    "relationship": "Relationship",
    "satisfaction": "Satisfaction",
    "emotions_first_time": "First-time Emotions",
    "benefits": "Perceived Benefits",
    "lube_reasons": "Lube Reasons",
    "fe9_agreement_t3b": "FE9 Agreement (T3B)",
    "numeric": "Numeric"
}

NUMERIC_METRICS = ["age_first_sex_numeric"]

# Build flat list of all metrics
ALL_METRICS = []
for group_metrics in METRIC_GROUPS.values():
    for m in group_metrics:
        if m not in ALL_METRICS:
            ALL_METRICS.append(m)

DIMENSIONS = [
    "unique_identifier",
    "survey_id",
    "country",
    "region",
    "language",
    "gender",
    "relationship_status",
    "virgin_recode",
    "sexual_experience",
    "age_first_sex_bucket",
    "inactive_reason_group"
]

METRIC_LABELS = {
    "sexually_active_hq4_flag": "Sexually Active",
    "explicit_inactive_flag": "Sexually Inactive (Explicit)",
    "chosen_not_to_have_sex": "Chosen Not to Have Sex",
    "is_committed_relationship": "In Committed Relationship",
    "current_sex_life_satisfied_t2b": "Satisfied with Sex Life (T2B)",
    "current_sex_life_not_satisfied": "Not Satisfied with Sex Life",
    "first_time_satisfaction_t2b": "Satisfied with First Time (T2B)",
    "emotion_confident_first_time": "Felt Confident (First Time)",
    "emotion_proud_first_time": "Felt Proud (First Time)",
    "emotion_ashamed_first_time": "Felt Ashamed (First Time)",
    "benefit_better_mood": "Better Mood",
    "benefit_feel_happier": "Feel Happier",
    "benefit_feel_healthier": "Feel Healthier",
    "benefit_more_confident": "More Confident",
    "benefit_sleep_better": "Sleep Better",
    "lube_reason_not_needed": "Lube Not Needed",
    "age_first_sex_numeric": "Age at First Sex",
    "fe9_confident_that_you_know_about_what_to_expect_when_having": "Know What to Expect",
    "fe9_having_sex_for_the_first_time_might_be_painful_for_me_ag": "Worried About Pain",
    "fe9_i_am_concerned_about_becoming_pregnant_making_my_partner": "Concerned About Pregnancy",
    "fe9_i_am_concerned_about_catching_an_sti_std_agreement_how_y": "Concerned About STIs",
    "fe9_i_am_looking_forward_to_having_sex_when_the_time_is_righ": "Looking Forward to Sex",
    "fe9_i_am_not_entirely_sure_of_all_the_things_that_one_does_w": "Unsure What to Do",
    "fe9_i_feel_nervous_scared_about_it_agreement_how_you_may_vie": "Feel Nervous/Scared",
    "fe9_i_feel_well_informed_and_confident_agreement_how_you_may": "Feel Well Informed",
    "fe9_i_know_what_to_do_to_protect_myself_and_my_partner_from_": "Know How to Protect",
    "fe9_i_worry_it_will_be_painful_for_me_or_my_partner_agreemen": "Worried About Pain for Partner"
}

DIMENSION_LABELS = {
    "unique_identifier": "Respondent ID",
    "survey_id": "Survey",
    "country": "Country",
    "region": "Region",
    "language": "Language",
    "gender": "Gender",
    "relationship_status": "Relationship Status",
    "virgin_recode": "Virgin Status",
    "sexual_experience": "Sexual Experience",
    "age_first_sex_bucket": "Age at First Sex",
    "inactive_reason_group": "Reason for Inactivity"
}

# Prompt templates
INSIGHT_PROMPT_TEMPLATE = """Analyze this GSS survey data and provide insights:

{{facts}}

Write a brief analysis (100 words max) covering:
1. **Key Finding** - The most notable result
2. **Comparison** - Notable differences across segments (if breakout provided)
3. **Implication** - What this suggests

Use markdown formatting. Be specific with numbers."""

MAX_PROMPT_TEMPLATE = "Answer user question in 30 words or less using following facts:\n{{facts}}"
