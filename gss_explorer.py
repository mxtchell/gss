"""
GSS Survey Explorer Skill
Flexible survey analysis with support for multiple metrics and breakouts
"""
from skill_framework import skill, SkillParameter, SkillInput, SkillOutput

from gss_explorer_helper.gss_functionality import run_gss_analysis
from gss_explorer_helper.gss_config import (
    METRIC_GROUPS, METRIC_GROUP_LABELS, ALL_METRICS, DIMENSIONS,
    METRIC_LABELS, DIMENSION_LABELS, INSIGHT_PROMPT_TEMPLATE, MAX_PROMPT_TEMPLATE
)


@skill(
    name="GSS Survey Explorer",
    llm_name="gss_explorer",
    description="""Analyzes GSS (Global Sex Survey) respondent-level data across multiple metrics and demographic dimensions.

This skill can analyze:
- Sexual activity status and relationship patterns
- Satisfaction with current sex life and first-time experiences
- Perceived benefits of sex (mood, happiness, health, confidence, sleep)
- First-time emotions (confident, proud, ashamed)
- Attitudes and concerns about sex (FE9 agreement statements)
- Age at first sexual experience

Supports flexible breakout analysis by country, gender, relationship status, sexual experience level, and other demographics.""",

    capabilities="""- Analyze single metrics or groups of related metrics together
- Compare metrics across one or two demographic dimensions simultaneously
- Calculate percentages for binary flags and averages for numeric metrics
- Generate bar charts for single breakouts, grouped column charts for dual breakouts
- Provide narrative insights highlighting highest/lowest segments and gaps
- Display detailed data tables with all segments and metrics""",

    limitations="""- Data is respondent-level survey data, not time-series
- Some metrics may have limited respondent counts in certain segments
- Cannot perform statistical significance testing
- Cannot combine filters with breakout dimensions on the same field""",

    example_questions="""- What are the top perceived benefits of sex?
- How does satisfaction with sex life vary by gender?
- Compare first-time emotions across countries
- Show sexual activity rates by relationship status
- What is the average age at first sex by country and gender?
- How do FE9 agreement scores differ between virgins and non-virgins?
- Break down satisfaction metrics by sexual experience level""",

    parameter_guidance="""**metrics**: Can be a metric group name (benefits, satisfaction, emotions_first_time, sexual_activity, relationship, fe9_agreement_t3b, numeric) or specific metric names. Defaults to benefits group if not specified.

**breakout_dimension**: Primary dimension to segment results by. Common choices: country, gender, relationship_status, sexual_experience, virgin_recode.

**breakout_dimension_2**: Secondary dimension for cross-tabulation analysis. Creates grouped comparisons.

**other_filters**: Additional filters to narrow the analysis population.""",

    parameters=[
        SkillParameter(
            name="metrics",
            description="Metrics or metric group to analyze. Can be group name (benefits, satisfaction, emotions_first_time, sexual_activity, fe9_agreement_t3b) or specific metric names.",
            is_required=False,
            is_multi=True
        ),
        SkillParameter(
            name="breakout_dimension",
            description="Primary dimension to break out results by (e.g., country, gender, relationship_status)",
            constrained_to="dimension",
            is_required=False
        ),
        SkillParameter(
            name="breakout_dimension_2",
            description="Secondary dimension for cross-tabulation (creates grouped chart)",
            constrained_to="dimension",
            is_required=False
        ),
        SkillParameter(
            name="other_filters",
            description="Additional filters to apply to the data",
            constrained_to="filter",
            is_required=False,
            is_multi=True
        ),
        SkillParameter(
            name="insight_prompt",
            description="Custom prompt for generating insights",
            parameter_type="prompt",
            default_value=INSIGHT_PROMPT_TEMPLATE
        ),
        SkillParameter(
            name="max_prompt",
            description="Prompt for max mode responses",
            parameter_type="max_prompt",
            default_value=MAX_PROMPT_TEMPLATE
        )
    ]
)
def gss_explorer(parameters: SkillInput) -> SkillOutput:
    """Main skill function for GSS survey exploration"""
    return run_gss_analysis(parameters)


if __name__ == '__main__':
    from skill_framework import preview_skill

    skill_input: SkillInput = gss_explorer.create_input(arguments={
        'metrics': ["benefits"],
        'breakout_dimension': "gender",
        'breakout_dimension_2': None,
        'other_filters': []
    })
    out = gss_explorer(skill_input)
    preview_skill(gss_explorer, out)
