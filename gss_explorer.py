"""
GSS Survey Explorer Skill
Flexible survey analysis with support for multiple metrics and breakouts
"""
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from answerrocket import skill
from answerrocket.skill import SkillInput, SkillOutput, SkillParameter, SkillVisualization, wire_layout
from answer_rocket import AnswerRocketClient

DATABASE_ID = os.getenv('DATABASE_ID', 'B855F1B7-35EA-46E1-B1D7-1630EEA5CA82')
TABLE_NAME = "read_csv('gss_max_ready_2026_v6.csv')"

# Metric groups from dataset config
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

# Build flat list of all metrics
METRICS = []
NUMERIC_METRICS = ["age_first_sex_numeric"]
for group_name, group_metrics in METRIC_GROUPS.items():
    for m in group_metrics:
        if m not in METRICS:
            METRICS.append(m)

# Available dimensions for breakouts
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

# Friendly labels for metrics
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


def resolve_metrics(metrics_input):
    """Resolve metric input - could be a group name, single metric, or list"""
    if not metrics_input:
        return METRIC_GROUPS["benefits"]

    if isinstance(metrics_input, str):
        # Check if it's a group name
        if metrics_input.lower() in METRIC_GROUPS:
            return METRIC_GROUPS[metrics_input.lower()]
        # Single metric
        return [metrics_input]

    # List of metrics
    resolved = []
    for m in metrics_input:
        if isinstance(m, str) and m.lower() in METRIC_GROUPS:
            resolved.extend(METRIC_GROUPS[m.lower()])
        else:
            resolved.append(m)
    return resolved


def clean_breakout(breakout):
    """Clean and validate breakout dimension"""
    if not breakout or str(breakout).lower() in ['none', '', 'null', 'na']:
        return None
    if breakout in DIMENSIONS:
        return breakout
    return None


@skill(
    skill_name='gss_explorer',
    skill_version='1.0.0',
    skill_description='Explore GSS survey data. Analyze benefits of sex, satisfaction levels, first-time experiences, and demographics. Supports single/multiple metrics, single/dual breakouts by country, gender, relationship status, and more.',
    inputs=[
        SkillParameter(
            name='metrics',
            display_name='Metrics',
            param_type='list',
            required=False
        ),
        SkillParameter(
            name='breakout_dimension',
            display_name='Primary Breakout',
            param_type='string',
            required=False
        ),
        SkillParameter(
            name='breakout_dimension_2',
            display_name='Secondary Breakout',
            param_type='string',
            required=False
        )
    ]
)
def gss_explorer(*args, **kwargs):
    """Main skill function for GSS survey exploration"""

    # Extract inputs
    skill_input = SkillInput.from_dict(kwargs)
    metrics_input = skill_input.get_user_input('metrics')
    breakout1 = skill_input.get_user_input('breakout_dimension')
    breakout2 = skill_input.get_user_input('breakout_dimension_2')
    filters = skill_input.get_filter()

    # Resolve metrics (handles groups, single, multiple)
    metrics = resolve_metrics(metrics_input)

    # Validate metrics exist
    all_metrics = METRICS + NUMERIC_METRICS
    metrics = [m for m in metrics if m in all_metrics]
    if not metrics:
        metrics = METRIC_GROUPS["benefits"]

    # Clean breakouts
    breakout1 = clean_breakout(breakout1)
    breakout2 = clean_breakout(breakout2)

    # Can't have breakout2 without breakout1
    if breakout2 and not breakout1:
        breakout1 = breakout2
        breakout2 = None

    # Can't have same breakout twice
    if breakout1 and breakout2 and breakout1 == breakout2:
        breakout2 = None

    print(f"DEBUG: Metrics: {metrics}")
    print(f"DEBUG: Breakout1: {breakout1}, Breakout2: {breakout2}")
    print(f"DEBUG: Filters: {filters}")

    # Build SQL query
    metric_selects = []
    for metric in metrics:
        if metric in NUMERIC_METRICS:
            metric_selects.append(f"AVG({metric}) AS {metric}")
        else:
            metric_selects.append(f"AVG({metric}) * 100 AS {metric}")

    # Determine grouping
    group_cols = []
    if breakout1:
        group_cols.append(breakout1)
    if breakout2:
        group_cols.append(breakout2)

    if group_cols:
        sql_query = f"""
        SELECT
            {', '.join(group_cols)},
            COUNT(*) AS respondent_count,
            {', '.join(metric_selects)}
        FROM {TABLE_NAME}
        WHERE 1=1
        """
    else:
        sql_query = f"""
        SELECT
            COUNT(*) AS respondent_count,
            {', '.join(metric_selects)}
        FROM {TABLE_NAME}
        WHERE 1=1
        """

    # Add filters
    if filters:
        for filter_item in filters:
            if isinstance(filter_item, dict) and 'dim' in filter_item:
                dim = filter_item['dim']
                values = filter_item.get('val')
                if isinstance(values, list) and values:
                    values_str = "', '".join(str(v) for v in values)
                    sql_query += f" AND {dim} IN ('{values_str}')"

    if group_cols:
        sql_query += f" GROUP BY {', '.join(group_cols)}"
        sql_query += f" ORDER BY {metrics[0]} DESC"

    print(f"DEBUG: SQL: {sql_query}")

    # Execute query
    try:
        client = AnswerRocketClient()
        result = client.data.execute_sql_query(DATABASE_ID, sql_query, row_limit=500)

        if not result.success or not hasattr(result, 'df'):
            error_msg = result.error if hasattr(result, 'error') else 'Unknown error'
            raise Exception(f"Query failed: {error_msg}")

        df = result.df.copy()
        print(f"DEBUG: Retrieved {len(df)} rows")

    except Exception as e:
        print(f"DEBUG: Query failed: {e}")
        return SkillOutput(
            final_prompt=f"Failed to retrieve survey data: {str(e)}",
            narrative="Error loading survey data.",
            visualizations=[]
        )

    if len(df) == 0:
        return SkillOutput(
            final_prompt="No data found matching the criteria.",
            narrative="No survey data available.",
            visualizations=[]
        )

    # Determine display settings
    is_percentage = all(m not in NUMERIC_METRICS for m in metrics)
    value_suffix = "%" if is_percentage else ""

    # Build title
    metric_names = [METRIC_LABELS.get(m, m.replace('_', ' ').title()) for m in metrics]
    if len(metric_names) > 3:
        title = f"Analysis of {len(metric_names)} Metrics"
    else:
        title = ", ".join(metric_names)

    if breakout1:
        title += f" by {DIMENSION_LABELS.get(breakout1, breakout1)}"
    if breakout2:
        title += f" and {DIMENSION_LABELS.get(breakout2, breakout2)}"

    # === BUILD CHART ===
    chart_config = build_chart(df, metrics, breakout1, breakout2, is_percentage, value_suffix)

    # === BUILD TABLE ===
    columns, table_data = build_table(df, metrics, breakout1, breakout2, is_percentage, value_suffix)

    # === BUILD NARRATIVE ===
    narrative = build_narrative(df, metrics, breakout1, breakout2, is_percentage, value_suffix, title)

    # === CREATE LAYOUT ===
    layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px", "fontFamily": "system-ui, -apple-system, sans-serif"},
            "children": [
                {
                    "name": "Header",
                    "type": "Paragraph",
                    "children": "",
                    "text": title,
                    "style": {"fontSize": "24px", "fontWeight": "bold", "marginBottom": "20px", "color": "#1e293b"}
                },
                {
                    "name": "Chart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "400px",
                    "options": chart_config
                },
                {
                    "name": "InsightsHeader",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Key Insights",
                    "style": {"fontSize": "18px", "fontWeight": "bold", "marginTop": "25px", "marginBottom": "10px", "color": "#1e293b"}
                },
                {
                    "name": "Insights",
                    "type": "Markdown",
                    "children": "",
                    "text": narrative,
                    "style": {"fontSize": "14px", "lineHeight": "1.6", "color": "#374151"}
                },
                {
                    "name": "TableHeader",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Detailed Data",
                    "style": {"fontSize": "18px", "fontWeight": "bold", "marginTop": "25px", "marginBottom": "15px", "color": "#1e293b"}
                },
                {
                    "name": "ResultsTable",
                    "type": "DataTable",
                    "children": "",
                    "columns": columns,
                    "data": table_data
                }
            ]
        },
        "inputVariables": []
    }

    # Render layout
    try:
        html = wire_layout(layout, {})
    except Exception as e:
        print(f"DEBUG: wire_layout failed: {e}")
        html = f"<div>Error rendering layout: {e}</div>"

    # Summary
    if breakout1 and breakout2:
        summary = f"Analyzed {len(metrics)} metric(s) by {DIMENSION_LABELS.get(breakout1, breakout1)} and {DIMENSION_LABELS.get(breakout2, breakout2)}."
    elif breakout1:
        summary = f"Analyzed {len(metrics)} metric(s) across {len(df)} {DIMENSION_LABELS.get(breakout1, breakout1)} segments."
    else:
        summary = f"Analyzed {len(metrics)} metric(s) across {int(df['respondent_count'].iloc[0]):,} respondents."

    return SkillOutput(
        final_prompt=summary,
        narrative=narrative,
        visualizations=[
            SkillVisualization(title="GSS Survey Explorer", layout=html)
        ]
    )


def build_chart(df, metrics, breakout1, breakout2, is_percentage, value_suffix):
    """Build appropriate chart based on breakout configuration"""

    if not breakout1:
        # NO BREAKOUT: Simple horizontal bar chart of metrics
        categories = [METRIC_LABELS.get(m, m.replace('_', ' ').title()) for m in metrics]
        values = [round(float(df[m].iloc[0]), 1) if pd.notna(df[m].iloc[0]) else 0 for m in metrics]

        return {
            "chart": {"type": "bar", "backgroundColor": "#ffffff", "height": max(300, len(metrics) * 50)},
            "title": {"text": ""},
            "xAxis": {"categories": categories},
            "yAxis": {
                "title": {"text": "Percentage (%)" if is_percentage else "Value"},
                "max": 100 if is_percentage else None,
                "min": 0
            },
            "series": [{"name": "Value", "data": values, "colorByPoint": True}],
            "legend": {"enabled": False},
            "credits": {"enabled": False},
            "tooltip": {"valueSuffix": value_suffix, "valueDecimals": 1, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"bar": {"dataLabels": {"enabled": True, "format": "{y:.1f}" + value_suffix}}}
        }

    elif breakout1 and not breakout2:
        # SINGLE BREAKOUT: Bar chart with categories
        categories = df[breakout1].astype(str).tolist()
        series = []

        for metric in metrics:
            label = METRIC_LABELS.get(metric, metric.replace('_', ' ').title())
            values = df[metric].fillna(0).round(1).tolist()
            series.append({"name": label, "data": values})

        return {
            "chart": {"type": "bar", "backgroundColor": "#ffffff", "height": max(400, len(categories) * 30)},
            "title": {"text": ""},
            "xAxis": {"categories": categories, "title": {"text": DIMENSION_LABELS.get(breakout1, breakout1)}},
            "yAxis": {
                "title": {"text": "Percentage (%)" if is_percentage else "Value"},
                "max": 100 if is_percentage else None,
                "min": 0
            },
            "series": series,
            "legend": {"enabled": len(metrics) > 1, "align": "center", "verticalAlign": "bottom"},
            "credits": {"enabled": False},
            "tooltip": {"shared": True, "valueSuffix": value_suffix, "valueDecimals": 1, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"bar": {"dataLabels": {"enabled": len(df) <= 10, "format": "{y:.1f}" + value_suffix}}}
        }

    else:
        # DUAL BREAKOUT: Grouped bar chart
        # Primary breakout on X axis, secondary breakout as series
        primary_vals = df[breakout1].unique().tolist()
        secondary_vals = df[breakout2].unique().tolist()

        # For simplicity, show first metric only in dual breakout chart
        metric = metrics[0]
        metric_label = METRIC_LABELS.get(metric, metric.replace('_', ' ').title())

        series = []
        for sec_val in secondary_vals:
            data = []
            for pri_val in primary_vals:
                mask = (df[breakout1] == pri_val) & (df[breakout2] == sec_val)
                if mask.any():
                    val = df.loc[mask, metric].iloc[0]
                    data.append(round(float(val), 1) if pd.notna(val) else 0)
                else:
                    data.append(0)
            series.append({"name": str(sec_val), "data": data})

        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 450},
            "title": {"text": metric_label},
            "xAxis": {"categories": [str(v) for v in primary_vals], "title": {"text": DIMENSION_LABELS.get(breakout1, breakout1)}},
            "yAxis": {
                "title": {"text": "Percentage (%)" if is_percentage else "Value"},
                "max": 100 if is_percentage else None,
                "min": 0
            },
            "series": series,
            "legend": {"enabled": True, "title": {"text": DIMENSION_LABELS.get(breakout2, breakout2)}},
            "credits": {"enabled": False},
            "tooltip": {"shared": False, "valueSuffix": value_suffix, "valueDecimals": 1, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"column": {"dataLabels": {"enabled": len(primary_vals) <= 6, "format": "{y:.1f}" + value_suffix}}}
        }


def build_table(df, metrics, breakout1, breakout2, is_percentage, value_suffix):
    """Build table columns and data"""

    columns = []

    # Add breakout columns
    if breakout1:
        columns.append({"name": DIMENSION_LABELS.get(breakout1, breakout1)})
    if breakout2:
        columns.append({"name": DIMENSION_LABELS.get(breakout2, breakout2)})

    # Add N column
    columns.append({"name": "N"})

    # Add metric columns
    for metric in metrics:
        columns.append({"name": METRIC_LABELS.get(metric, metric.replace('_', ' ').title())})

    # Build rows
    table_data = []

    if not breakout1:
        # Single row, no breakout
        row = [f"{int(df['respondent_count'].iloc[0]):,}"]
        for metric in metrics:
            val = df[metric].iloc[0]
            row.append(f"{val:.1f}{value_suffix}" if pd.notna(val) else "N/A")
        table_data.append(row)
    else:
        # Multiple rows with breakouts
        for _, df_row in df.iterrows():
            row = []
            if breakout1:
                row.append(str(df_row[breakout1]))
            if breakout2:
                row.append(str(df_row[breakout2]))
            row.append(f"{int(df_row['respondent_count']):,}")
            for metric in metrics:
                val = df_row[metric]
                row.append(f"{val:.1f}{value_suffix}" if pd.notna(val) else "N/A")
            table_data.append(row)

    return columns, table_data


def build_narrative(df, metrics, breakout1, breakout2, is_percentage, value_suffix, title):
    """Build insights narrative"""

    parts = []

    if not breakout1:
        # Overall stats
        total = int(df['respondent_count'].iloc[0])
        parts.append(f"Based on **{total:,}** respondents:\n")

        # Rank and describe metrics
        metric_vals = []
        for metric in metrics:
            val = df[metric].iloc[0]
            if pd.notna(val):
                metric_vals.append((METRIC_LABELS.get(metric, metric), float(val)))

        metric_vals.sort(key=lambda x: x[1], reverse=True)

        if metric_vals:
            top = metric_vals[0]
            parts.append(f"- **Top result:** {top[0]} at {top[1]:.1f}{value_suffix}")
            if len(metric_vals) > 1:
                bottom = metric_vals[-1]
                parts.append(f"- **Lowest:** {bottom[0]} at {bottom[1]:.1f}{value_suffix}")

    elif breakout1 and not breakout2:
        # Single breakout insights
        parts.append(f"Analysis across **{len(df)}** {DIMENSION_LABELS.get(breakout1, breakout1)} segments:\n")

        for metric in metrics[:3]:  # Limit to first 3 metrics for readability
            label = METRIC_LABELS.get(metric, metric)
            if metric in df.columns:
                max_val = df[metric].max()
                min_val = df[metric].min()
                if pd.notna(max_val) and pd.notna(min_val):
                    max_seg = df.loc[df[metric].idxmax(), breakout1]
                    min_seg = df.loc[df[metric].idxmin(), breakout1]
                    gap = max_val - min_val
                    parts.append(f"- **{label}:** Highest in {max_seg} ({max_val:.1f}{value_suffix}), lowest in {min_seg} ({min_val:.1f}{value_suffix}). Range: {gap:.1f} pts.")

    else:
        # Dual breakout insights
        parts.append(f"Cross-analysis by **{DIMENSION_LABELS.get(breakout1, breakout1)}** and **{DIMENSION_LABELS.get(breakout2, breakout2)}**:\n")

        metric = metrics[0]
        label = METRIC_LABELS.get(metric, metric)

        if metric in df.columns:
            max_val = df[metric].max()
            min_val = df[metric].min()
            if pd.notna(max_val) and pd.notna(min_val):
                max_row = df.loc[df[metric].idxmax()]
                min_row = df.loc[df[metric].idxmin()]
                parts.append(f"- **{label}** highest ({max_val:.1f}{value_suffix}): {max_row[breakout1]} / {max_row[breakout2]}")
                parts.append(f"- **{label}** lowest ({min_val:.1f}{value_suffix}): {min_row[breakout1]} / {min_row[breakout2]}")
                parts.append(f"- Overall range: {max_val - min_val:.1f} percentage points")

    return "\n".join(parts)
