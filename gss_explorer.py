from __future__ import annotations
from skill_framework import skill, SkillParameter, SkillInput, SkillOutput

from gss_explorer_helper.gss_functionality import run_gss_analysis
from gss_explorer_helper.gss_config import INSIGHT_PROMPT_TEMPLATE, MAX_PROMPT_TEMPLATE

@skill(
    name="GSS Survey Explorer",
    llm_name="gss_explorer",
    description="Analyzes GSS (Global Sex Survey) respondent-level data across multiple metrics and demographic dimensions. Covers sexual activity, satisfaction, perceived benefits, first-time emotions, and FE9 agreement statements.",
    capabilities="Analyze single metrics or groups of related metrics. Compare across one or two demographic dimensions. Generate bar charts for single breakouts, grouped column charts for dual breakouts. Provide narrative insights and detailed data tables.",
    limitations="Data is respondent-level survey data, not time-series. Some metrics may have limited respondent counts in certain segments. Cannot perform statistical significance testing.",
    example_questions="What are the top perceived benefits of sex? How does satisfaction with sex life vary by gender? Compare first-time emotions across countries. Show sexual activity rates by relationship status. What is the average age at first sex by country and gender?",
    parameter_guidance="metrics: Can be a metric group name (benefits, satisfaction, emotions_first_time, sexual_activity, fe9_agreement_t3b) or specific metric names. breakout_dimension: Primary dimension to segment by (country, gender, relationship_status). breakout_dimension_2: Secondary dimension for cross-tabulation.",
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
