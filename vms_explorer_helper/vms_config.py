"""
VMS Explorer Configuration
Metric groups, dimensions, and labels for Reckitt VMS US Survey
"""
import os

DATABASE_ID = os.getenv('DATABASE_ID', 'B855F1B7-35EA-46E1-B1D7-1630EEA5CA82')
TABLE_NAME = "read_csv('reckitt_vms_survey_long.csv')"

# Metric groups from dataset
METRIC_GROUPS = {
    "brain_health_interests": [
        "interest_long_term_memory",
        "interest_short_term_memory",
        "interest_working_memory",
        "interest_focus_concentration",
        "interest_reasoning",
        "interest_eye_health",
        "interest_stress_relief",
        "interest_sleep",
        "interest_mood",
        "interest_mental_energy",
        "interest_motor_skills"
    ],
    "psychographics": [
        "psycho_self_care",
        "psycho_prefers_naturals",
        "psycho_wants_science",
        "psycho_hcp_reliant",
        "psycho_health_daily_concern",
        "psycho_too_busy",
        "psycho_younger_for_longer",
        "psycho_age_concerned",
        "psycho_afraid_health_decline",
        "psycho_family_support"
    ],
    "needs": [
        "need_improve_brain_health",
        "need_boosting",
        "need_maintain_prevent",
        "need_restore_after_decline"
    ],
    "brand_awareness": [
        "brand_aware"
    ],
    "brand_usage": [
        "is_user",
        "freq_daily_usage"
    ],
    "category_awareness": [
        "aware_brain_health_category"
    ]
}

METRIC_GROUP_LABELS = {
    "brain_health_interests": "Brain Health Interests",
    "psychographics": "Psychographics",
    "needs": "Primary Needs",
    "brand_awareness": "Brand Awareness",
    "brand_usage": "Brand Usage",
    "category_awareness": "Category Awareness"
}

# Metrics that are numeric (not percentages)
NUMERIC_METRICS = ["age"]

# Brand-level metrics - these vary per brand row and need SUM/COUNT(*) to handle NULLs correctly
BRAND_METRICS = {"is_user", "brand_aware"}

# Respondent-level metrics - identical across all brand rows for same respondent
# Must deduplicate by respondent when breaking out by brand dimensions
RESPONDENT_METRICS = {
    "interest_long_term_memory", "interest_short_term_memory", "interest_working_memory",
    "interest_focus_concentration", "interest_reasoning", "interest_eye_health",
    "interest_stress_relief", "interest_sleep", "interest_mood", "interest_mental_energy",
    "interest_motor_skills",
    "psycho_self_care", "psycho_prefers_naturals", "psycho_wants_science",
    "psycho_hcp_reliant", "psycho_health_daily_concern", "psycho_too_busy",
    "psycho_younger_for_longer", "psycho_age_concerned", "psycho_afraid_health_decline",
    "psycho_family_support",
    "need_improve_brain_health", "need_boosting", "need_maintain_prevent",
    "need_restore_after_decline",
    "freq_daily_usage", "aware_brain_health_category"
}

# Calculated metrics - these get expanded to SQL expressions
CALCULATED_METRICS = {
    # Example: % of users among those aware
    "pct_users_among_aware": {
        "sql": "SUM(CASE WHEN brand_aware = 1 THEN CAST(is_user AS DOUBLE) * weight ELSE NULL END) / NULLIF(SUM(CASE WHEN brand_aware = 1 THEN weight ELSE NULL END), 0)",
        "is_pct": True
    }
}

# Special metrics handled by custom SQL logic (not via CALCULATED_METRICS)
SPECIAL_METRICS = {"respondent_share"}

# Build flat list of all metrics
ALL_METRICS = []
for group_metrics in METRIC_GROUPS.values():
    for m in group_metrics:
        if m not in ALL_METRICS:
            ALL_METRICS.append(m)
# Add special metrics
for m in SPECIAL_METRICS:
    if m not in ALL_METRICS:
        ALL_METRICS.append(m)

# Dimensions available for breakouts
DIMENSIONS = [
    "gender",
    "age_band",
    "region",
    "state",
    "education",
    "employment_status",
    "marital_status",
    "shopping_for",
    "user_type",
    "is_hispanic",
    "brand_name",
    "brand_category",
    "usage_level",
    "is_user"
]

# Human-readable metric labels
METRIC_LABELS = {
    # Brain health interests
    "interest_long_term_memory": "Interest: Long-term Memory",
    "interest_short_term_memory": "Interest: Short-term Memory",
    "interest_working_memory": "Interest: Working Memory",
    "interest_focus_concentration": "Interest: Focus & Concentration",
    "interest_reasoning": "Interest: Reasoning",
    "interest_eye_health": "Interest: Eye Health",
    "interest_stress_relief": "Interest: Stress Relief",
    "interest_sleep": "Interest: Sleep",
    "interest_mood": "Interest: Mood",
    "interest_mental_energy": "Interest: Mental Energy",
    "interest_motor_skills": "Interest: Motor Skills",
    # Psychographics
    "psycho_self_care": "Values Self-Care",
    "psycho_prefers_naturals": "Prefers Natural Products",
    "psycho_wants_science": "Wants Scientific Evidence",
    "psycho_hcp_reliant": "HCP Reliant",
    "psycho_health_daily_concern": "Health is Daily Concern",
    "psycho_too_busy": "Too Busy for Health",
    "psycho_younger_for_longer": "Wants to Stay Young",
    "psycho_age_concerned": "Concerned About Aging",
    "psycho_afraid_health_decline": "Afraid of Health Decline",
    "psycho_family_support": "Has Family Support",
    # Needs
    "need_improve_brain_health": "Need: Improve Brain Health",
    "need_boosting": "Need: Boosting Performance",
    "need_maintain_prevent": "Need: Maintain/Prevent Decline",
    "need_restore_after_decline": "Need: Restore After Decline",
    # Brand metrics
    "brand_aware": "Brand Awareness",
    "is_user": "Brand Usage (L12M)",
    "freq_daily_usage": "Daily Usage Frequency",
    "aware_brain_health_category": "Aware of Brain Health Supplements",
    # Calculated
    "pct_users_among_aware": "% Users Among Aware",
    # Special
    "respondent_share": "% of Group"
}

# Human-readable dimension labels
DIMENSION_LABELS = {
    "gender": "Gender",
    "age_band": "Age Group",
    "region": "Region",
    "state": "State",
    "education": "Education",
    "employment_status": "Employment",
    "marital_status": "Marital Status",
    "shopping_for": "Shopping For",
    "user_type": "User Type",
    "is_hispanic": "Hispanic",
    "brand_name": "Brand",
    "brand_category": "Brand Category",
    "usage_level": "Usage Frequency",
    "is_user": "Is User (L12M)"
}

# Brand categories for reference
BRAND_CATEGORIES = [
    "Brain Health Supplements",
    "Multivitamins",
    "Sleep Aids",
    "Focus/Energy Supplements",
    "Supplements (Omega/Minerals)",
    "Energy/Caffeine",
    "Cognitive Rx",
    "Stimulant Rx",
    "Sleep Rx"
]

# Key brands for quick filtering
KEY_BRANDS = [
    "Centrum",
    "Nature Made (Multi)",
    "Melatonin",
    "One A Day",
    "ZzzQuil",
    "Nature Made",
    "Neuriva",
    "Prevagen",
    "Focus Factor",
    "Olly (Multi)",
    "Spring Valley",
    "Ashwagandha"
]

# Prompt templates
INSIGHT_PROMPT_TEMPLATE = """Analyze this VMS (Vitamins, Minerals, Supplements) survey data and provide insights:

{{facts}}

Write a brief analysis (100 words max) covering:
1. **Key Finding** - The most notable result
2. **Comparison** - Notable differences across segments (if breakout provided)
3. **Implication** - What this suggests for brand strategy

Use markdown formatting. Be specific with numbers. Focus on actionable insights for Reckitt."""

MAX_PROMPT_TEMPLATE = "Answer user question in 30 words or less using following facts:\n{{facts}}"
