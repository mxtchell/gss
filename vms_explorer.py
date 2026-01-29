from __future__ import annotations
from skill_framework import skill, SkillParameter, SkillInput, SkillOutput

from vms_explorer_helper.vms_functionality import run_vms_analysis
from vms_explorer_helper.vms_config import INSIGHT_PROMPT_TEMPLATE, MAX_PROMPT_TEMPLATE

@skill(
    name="VMS Survey Explorer",
    llm_name="vms_explorer",
    description=(
        "Analyzes the Reckitt VMS (Vitamins, Minerals, Supplements) US consumer survey. "
        "3,536 respondents across 72 brands in 9 categories: Brain Health Supplements (Neuriva, Prevagen, Focus Factor), "
        "Multivitamins (Centrum, One A Day, Nature Made), Sleep Aids (ZzzQuil, Melatonin, Unisom), "
        "Focus/Energy Supplements, Supplements (Omega/Minerals), Energy/Caffeine, Cognitive Rx, Stimulant Rx, Sleep Rx. "
        "Data covers brain health benefit interests, brand awareness & usage, consumer psychographics, and primary health needs."
    ),
    capabilities=(
        "1) Brain health interest analysis: long-term memory (59%), short-term memory (61%), focus & concentration (63%), "
        "sleep (60%), mood (52%), stress relief (57%), working memory, reasoning, eye health, mental energy, motor skills. "
        "2) Brand awareness and usage rates across 72 brands - top used: Centrum, Nature Made, Melatonin, One A Day, ZzzQuil. "
        "3) Psychographic profiling: self-care values, preference for naturals, science-driven, HCP reliance, aging concerns. "
        "4) Primary need states: improving brain health, boosting performance, maintaining/preventing decline, restoring after decline. "
        "5) Category awareness (51.6% aware of brain health supplements). "
        "6) Cross-tabulation by demographics: gender (Male/Female), age band (18-24 to 65+), region (South/West/Northeast/Midwest), "
        "education, employment, marital status, shopping for (self vs others), user type (VMS vs Prescription). "
        "7) Brand-level filtering: compare specific brands, filter by brand category."
    ),
    limitations=(
        "Single survey wave (Nov 2024), no time-series trending. Data is unweighted by default (weight column available). "
        "Cannot perform statistical significance testing. Usage defined as last 12 months (usage_level_code 1-4). "
        "254,592 rows in long format (respondent x brand); respondent-level metrics auto-deduplicate."
    ),
    example_questions=(
        "What percentage of consumers are interested in improving long-term memory? | "
        "What are all the brain health interest areas and how do they rank? | "
        "How does interest in sleep compare to interest in focus by age group? | "
        "What percentage of consumers are aware of brain health supplements? | "
        "Who uses ZzzQuil? Break down by gender. | "
        "What percentage of ZzzQuil users are female? | "
        "Compare brand awareness for Neuriva vs Prevagen vs Focus Factor. | "
        "What are the top 10 most-used brands? | "
        "How do psychographic profiles differ between Males and Females? | "
        "What primary needs drive Brain Health Supplement users vs Sleep Aid users? | "
        "Show brain health interests broken out by region. | "
        "What percentage of consumers are interested in BOTH long-term AND short-term memory? | "
        "How does brand usage vary across age groups for Centrum vs One A Day?"
    ),
    parameter_guidance=(
        "METRIC GROUPS (pass group name to get all metrics in that group): "
        "brain_health_interests = 11 interest areas (memory, focus, sleep, mood, stress, etc). "
        "psychographics = 10 attitudinal segments (self-care, naturals, science, aging, etc). "
        "needs = 4 primary need states (improve, boost, maintain, restore). "
        "brand_awareness = brand_aware flag per brand. "
        "brand_usage = is_user flag (used in last 12 months). "
        "category_awareness = aware_brain_health_category. "
        "INDIVIDUAL METRICS: interest_sleep, interest_mood, interest_focus_concentration, psycho_self_care, is_user, brand_aware, etc. "
        "BREAKOUT DIMENSIONS: gender, age_band, region, brand_name, brand_category, education, employment_status, marital_status, shopping_for, user_type. "
        "FILTERS: Use other_filters to narrow to specific brands (brand_name='ZzzQuil'), categories (brand_category='Sleep Aids'), "
        "demographics (gender='Female'), or user types (user_type='VMS'). "
        "TIPS: For brand comparisons, use brand_name as breakout and filter by brand_category. "
        "For respondent-level metrics (interests, psychographics), data auto-deduplicates by respondent. "
        "For brand-level metrics (awareness, usage), all rows are used. "
        "COMPOSITION QUESTIONS (e.g. 'what % of ZzzQuil users are female'): use metric=respondent_share, "
        "breakout=gender, filter is_user=1 AND brand_name='ZzzQuil'. This shows the % distribution across segments. "
        "For 'who uses X by Y' questions, use respondent_share with is_user=1 filter."
    ),
    parameters=[
        SkillParameter(
            name="metrics",
            description=(
                "Metrics or metric group to analyze. "
                "Groups: brain_health_interests (11 interest areas), psychographics (10 attitudes), "
                "needs (4 need states), brand_awareness, brand_usage, category_awareness. "
                "Or specific metrics: interest_long_term_memory, interest_short_term_memory, interest_working_memory, "
                "interest_focus_concentration, interest_reasoning, interest_eye_health, interest_stress_relief, "
                "interest_sleep, interest_mood, interest_mental_energy, interest_motor_skills, "
                "psycho_self_care, psycho_prefers_naturals, psycho_wants_science, psycho_hcp_reliant, "
                "psycho_health_daily_concern, psycho_too_busy, psycho_younger_for_longer, psycho_age_concerned, "
                "psycho_afraid_health_decline, psycho_family_support, "
                "need_improve_brain_health, need_boosting, need_maintain_prevent, need_restore_after_decline, "
                "brand_aware, is_user, aware_brain_health_category. "
                "Special: respondent_share (% of group in each breakout segment - use with filters like is_user=1 to show composition, "
                "e.g. 'what % of ZzzQuil users are female')."
            ),
            is_multi=True
        ),
        SkillParameter(
            name="breakout_dimension",
            constrained_to="dimension",
            description=(
                "Primary dimension to break out results by. Options: "
                "gender (Male/Female/Other), age_band (18-24, 25-34, 35-44, 45-54, 55-64, 65+), "
                "region (South/West/Northeast/Midwest), brand_name (72 brands), "
                "brand_category (Brain Health Supplements/Multivitamins/Sleep Aids/Focus Energy/etc), "
                "education, employment_status, marital_status, shopping_for (Myself/Someone else), "
                "user_type (VMS/Prescription), is_hispanic, state."
            )
        ),
        SkillParameter(
            name="breakout_dimension_2",
            constrained_to="dimension",
            description="Optional secondary dimension for cross-tabulation (creates grouped chart). Same options as breakout_dimension."
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            description=(
                "Filters to narrow the analysis. Examples: "
                "brand_name = 'ZzzQuil' | brand_name IN ('Neuriva','Prevagen','Focus Factor') | "
                "brand_category = 'Sleep Aids' | gender = 'Female' | age_band = '55-64' | "
                "region = 'South' | user_type = 'VMS' | is_user = 1 (users only) | brand_aware = 1 (aware only)."
            )
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Custom prompt for generating insights",
            default_value=INSIGHT_PROMPT_TEMPLATE
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for max mode responses",
            default_value=MAX_PROMPT_TEMPLATE
        )
    ]
)
def vms_explorer(parameters: SkillInput) -> SkillOutput:
    """VMS Survey Explorer - analyze Reckitt VMS survey metrics with flexible breakouts"""
    return run_vms_analysis(parameters)


if __name__ == '__main__':
    from skill_framework import preview_skill

    skill_input: SkillInput = vms_explorer.create_input(arguments={
        'metrics': ["brain_health_interests"],
        'breakout_dimension': "gender",
        'breakout_dimension_2': None,
        'other_filters': []
    })
    out = vms_explorer(skill_input)
    preview_skill(vms_explorer, out)
