from __future__ import annotations
from skill_framework import skill, SkillParameter, SkillInput, SkillOutput

from gss_explorer_helper.gss_functionality import run_gss_analysis
from gss_explorer_helper.gss_config import INSIGHT_PROMPT_TEMPLATE, MAX_PROMPT_TEMPLATE

@skill(
    name="GSS Survey Explorer",
    llm_name="gss_explorer",
    description="Analyzes GSS (Global Sex Survey) respondent-level data across multiple metrics and demographic dimensions. Covers sexual activity status, relationship status (is_committed_relationship), satisfaction, perceived benefits, first-time emotions, and FE9 agreement statements. Use is_committed_relationship metric to analyze % in committed relationships. Use other_filters to filter by sexually_active_hq4_flag=1 when analyzing only sexually active respondents.",
    capabilities="Analyze single metrics or groups of related metrics. Compare across one or two demographic dimensions. Generate column charts for comparisons. Provide narrative insights and detailed data tables. Filter to specific populations using other_filters.",
    limitations="Data is respondent-level survey data, not time-series. Some metrics may have limited respondent counts in certain segments. Cannot perform statistical significance testing.",
    example_questions="What are the top perceived benefits of sex? How does satisfaction with sex life vary by gender? What % of people are sexually active globally? (use sexually_active_hq4_flag) What % of sexually active people are in committed relationships? (use pct_committed_among_active) What is the average age at first sex by country?",
    parameter_guidance="metrics: Use sexually_active_hq4_flag for '% sexually active globally'. Use pct_committed_among_active for '% of sexually active in committed relationships'. Use group names (benefits, satisfaction, emotions_first_time) or specific metrics. breakout_dimension: Segment by country, gender, relationship_status. other_filters: Filter population when needed.",
    parameters=[
        SkillParameter(
            name="metrics",
            description="Metrics or metric group to analyze. Can be group name (benefits, satisfaction, emotions_first_time, sexual_activity, fe9_agreement_t3b) or specific metric names.",
            is_multi=True
        ),
        SkillParameter(
            name="breakout_dimension",
            constrained_to="dimension",
            description="Primary dimension to break out results by (e.g., country, gender, relationship_status)"
        ),
        SkillParameter(
            name="breakout_dimension_2",
            constrained_to="dimension",
            description="Secondary dimension for cross-tabulation (creates grouped chart)"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            description="Additional filters to apply to the data"
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
def gss_explorer(parameters: SkillInput) -> SkillOutput:
    """GSS Survey Explorer - analyze survey metrics with flexible breakouts"""
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
