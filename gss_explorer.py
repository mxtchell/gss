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

    # === CONFIG ===
    DATABASE_ID = os.getenv('DATABASE_ID', 'B855F1B7-35EA-46E1-B1D7-1630EEA5CA82')
    TABLE_NAME = "read_csv('gss_max_ready_2026_v6.csv')"

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

    NUMERIC_METRICS = ["age_first_sex_numeric"]

    # Build flat list of all metrics
    ALL_METRICS = []
    for group_metrics in METRIC_GROUPS.values():
        for m in group_metrics:
            if m not in ALL_METRICS:
                ALL_METRICS.append(m)

    DIMENSIONS = [
        "unique_identifier", "survey_id", "country", "region", "language",
        "gender", "relationship_status", "virgin_recode", "sexual_experience",
        "age_first_sex_bucket", "inactive_reason_group"
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

    # === HELPER FUNCTIONS ===
    def resolve_metrics(metrics_input):
        if not metrics_input:
            return METRIC_GROUPS["benefits"]
        if isinstance(metrics_input, str):
            if metrics_input.lower() in METRIC_GROUPS:
                return METRIC_GROUPS[metrics_input.lower()]
            return [metrics_input]
        resolved = []
        for m in metrics_input:
            if isinstance(m, str) and m.lower() in METRIC_GROUPS:
                resolved.extend(METRIC_GROUPS[m.lower()])
            else:
                resolved.append(m)
        return resolved

    def clean_breakout(breakout):
        if not breakout or str(breakout).lower() in ['none', '', 'null', 'na']:
            return None
        if breakout in DIMENSIONS:
            return breakout
        return None

    def get_label(metric):
        return METRIC_LABELS.get(metric, metric.replace('_', ' ').title())

    def get_dim_label(dim):
        return DIMENSION_LABELS.get(dim, dim.replace('_', ' ').title())

    # === EXTRACT INPUTS ===
    skill_input = SkillInput.from_dict(kwargs)
    metrics_input = skill_input.get_user_input('metrics')
    breakout1 = skill_input.get_user_input('breakout_dimension')
    breakout2 = skill_input.get_user_input('breakout_dimension_2')
    filters = skill_input.get_filter()

    # Resolve and validate
    metrics = resolve_metrics(metrics_input)
    metrics = [m for m in metrics if m in ALL_METRICS]
    if not metrics:
        metrics = METRIC_GROUPS["benefits"]

    breakout1 = clean_breakout(breakout1)
    breakout2 = clean_breakout(breakout2)

    if breakout2 and not breakout1:
        breakout1, breakout2 = breakout2, None
    if breakout1 and breakout2 and breakout1 == breakout2:
        breakout2 = None

    print(f"DEBUG: Metrics: {metrics}")
    print(f"DEBUG: Breakout1: {breakout1}, Breakout2: {breakout2}")

    # === BUILD SQL ===
    metric_selects = []
    for metric in metrics:
        if metric in NUMERIC_METRICS:
            metric_selects.append(f"AVG({metric}) AS {metric}")
        else:
            metric_selects.append(f"AVG({metric}) * 100 AS {metric}")

    group_cols = [b for b in [breakout1, breakout2] if b]

    if group_cols:
        sql_query = f"""
        SELECT {', '.join(group_cols)}, COUNT(*) AS respondent_count, {', '.join(metric_selects)}
        FROM {TABLE_NAME} WHERE 1=1
        """
    else:
        sql_query = f"""
        SELECT COUNT(*) AS respondent_count, {', '.join(metric_selects)}
        FROM {TABLE_NAME} WHERE 1=1
        """

    if filters:
        for f in filters:
            if isinstance(f, dict) and 'dim' in f:
                dim, values = f['dim'], f.get('val')
                if isinstance(values, list) and values:
                    values_str = "', '".join(str(v) for v in values)
                    sql_query += f" AND {dim} IN ('{values_str}')"

    if group_cols:
        sql_query += f" GROUP BY {', '.join(group_cols)} ORDER BY {metrics[0]} DESC"

    print(f"DEBUG: SQL: {sql_query}")

    # === EXECUTE QUERY ===
    try:
        client = AnswerRocketClient()
        result = client.data.execute_sql_query(DATABASE_ID, sql_query, row_limit=500)
        if not result.success or not hasattr(result, 'df'):
            raise Exception(f"Query failed: {getattr(result, 'error', 'Unknown')}")
        df = result.df.copy()
        print(f"DEBUG: Retrieved {len(df)} rows")
    except Exception as e:
        print(f"DEBUG: Query failed: {e}")
        return SkillOutput(final_prompt=f"Error: {e}", narrative="Error loading data.", visualizations=[])

    if len(df) == 0:
        return SkillOutput(final_prompt="No data found.", narrative="No data available.", visualizations=[])

    # === BUILD OUTPUT ===
    is_pct = all(m not in NUMERIC_METRICS for m in metrics)
    suffix = "%" if is_pct else ""

    # Title
    metric_names = [get_label(m) for m in metrics]
    title = ", ".join(metric_names[:3]) if len(metric_names) <= 3 else f"{len(metric_names)} Metrics"
    if breakout1:
        title += f" by {get_dim_label(breakout1)}"
    if breakout2:
        title += f" and {get_dim_label(breakout2)}"

    # === CHART ===
    if not breakout1:
        categories = [get_label(m) for m in metrics]
        values = [round(float(df[m].iloc[0]), 1) if pd.notna(df[m].iloc[0]) else 0 for m in metrics]
        chart = {
            "chart": {"type": "bar", "backgroundColor": "#ffffff", "height": max(300, len(metrics) * 50)},
            "title": {"text": ""}, "xAxis": {"categories": categories},
            "yAxis": {"title": {"text": "%" if is_pct else "Value"}, "max": 100 if is_pct else None, "min": 0},
            "series": [{"name": "Value", "data": values, "colorByPoint": True}],
            "legend": {"enabled": False}, "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"bar": {"dataLabels": {"enabled": True, "format": "{y:.1f}" + suffix}}}
        }
    elif breakout1 and not breakout2:
        categories = df[breakout1].astype(str).tolist()
        series = [{"name": get_label(m), "data": df[m].fillna(0).round(1).tolist()} for m in metrics]
        chart = {
            "chart": {"type": "bar", "backgroundColor": "#ffffff", "height": max(400, len(categories) * 30)},
            "title": {"text": ""}, "xAxis": {"categories": categories, "title": {"text": get_dim_label(breakout1)}},
            "yAxis": {"title": {"text": "%" if is_pct else "Value"}, "max": 100 if is_pct else None, "min": 0},
            "series": series, "legend": {"enabled": len(metrics) > 1}, "credits": {"enabled": False},
            "tooltip": {"shared": True, "valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"bar": {"dataLabels": {"enabled": len(df) <= 10, "format": "{y:.1f}" + suffix}}}
        }
    else:
        pri_vals, sec_vals = df[breakout1].unique().tolist(), df[breakout2].unique().tolist()
        metric = metrics[0]
        series = []
        for sv in sec_vals:
            data = []
            for pv in pri_vals:
                mask = (df[breakout1] == pv) & (df[breakout2] == sv)
                val = df.loc[mask, metric].iloc[0] if mask.any() else 0
                data.append(round(float(val), 1) if pd.notna(val) else 0)
            series.append({"name": str(sv), "data": data})
        chart = {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 450},
            "title": {"text": get_label(metric)},
            "xAxis": {"categories": [str(v) for v in pri_vals], "title": {"text": get_dim_label(breakout1)}},
            "yAxis": {"title": {"text": "%" if is_pct else "Value"}, "max": 100 if is_pct else None, "min": 0},
            "series": series, "legend": {"enabled": True, "title": {"text": get_dim_label(breakout2)}},
            "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"column": {"dataLabels": {"enabled": len(pri_vals) <= 6, "format": "{y:.1f}" + suffix}}}
        }

    # === TABLE ===
    columns = []
    if breakout1:
        columns.append({"name": get_dim_label(breakout1)})
    if breakout2:
        columns.append({"name": get_dim_label(breakout2)})
    columns.append({"name": "N"})
    columns.extend([{"name": get_label(m)} for m in metrics])

    table_data = []
    if not breakout1:
        row = [f"{int(df['respondent_count'].iloc[0]):,}"]
        row.extend([f"{df[m].iloc[0]:.1f}{suffix}" if pd.notna(df[m].iloc[0]) else "N/A" for m in metrics])
        table_data.append(row)
    else:
        for _, r in df.iterrows():
            row = []
            if breakout1:
                row.append(str(r[breakout1]))
            if breakout2:
                row.append(str(r[breakout2]))
            row.append(f"{int(r['respondent_count']):,}")
            row.extend([f"{r[m]:.1f}{suffix}" if pd.notna(r[m]) else "N/A" for m in metrics])
            table_data.append(row)

    # === NARRATIVE ===
    narrative = []
    if not breakout1:
        total = int(df['respondent_count'].iloc[0])
        narrative.append(f"Based on **{total:,}** respondents:\n")
        vals = [(get_label(m), float(df[m].iloc[0])) for m in metrics if pd.notna(df[m].iloc[0])]
        vals.sort(key=lambda x: x[1], reverse=True)
        if vals:
            narrative.append(f"- **Top:** {vals[0][0]} at {vals[0][1]:.1f}{suffix}")
            if len(vals) > 1:
                narrative.append(f"- **Lowest:** {vals[-1][0]} at {vals[-1][1]:.1f}{suffix}")
    elif breakout1 and not breakout2:
        narrative.append(f"Analysis across **{len(df)}** {get_dim_label(breakout1)} segments:\n")
        for m in metrics[:3]:
            if m in df.columns:
                mx, mn = df[m].max(), df[m].min()
                if pd.notna(mx) and pd.notna(mn):
                    mx_seg = df.loc[df[m].idxmax(), breakout1]
                    mn_seg = df.loc[df[m].idxmin(), breakout1]
                    narrative.append(f"- **{get_label(m)}:** Highest in {mx_seg} ({mx:.1f}{suffix}), lowest in {mn_seg} ({mn:.1f}{suffix}). Gap: {mx-mn:.1f} pts.")
    else:
        narrative.append(f"Cross-analysis by **{get_dim_label(breakout1)}** and **{get_dim_label(breakout2)}**:\n")
        m = metrics[0]
        if m in df.columns:
            mx, mn = df[m].max(), df[m].min()
            if pd.notna(mx) and pd.notna(mn):
                mx_r, mn_r = df.loc[df[m].idxmax()], df.loc[df[m].idxmin()]
                narrative.append(f"- **{get_label(m)}** highest ({mx:.1f}{suffix}): {mx_r[breakout1]} / {mx_r[breakout2]}")
                narrative.append(f"- **{get_label(m)}** lowest ({mn:.1f}{suffix}): {mn_r[breakout1]} / {mn_r[breakout2]}")

    narrative_text = "\n".join(narrative)

    # === LAYOUT ===
    layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px", "fontFamily": "system-ui, -apple-system, sans-serif"},
            "children": [
                {"name": "Header", "type": "Paragraph", "children": "", "text": title,
                 "style": {"fontSize": "24px", "fontWeight": "bold", "marginBottom": "20px", "color": "#1e293b"}},
                {"name": "Chart", "type": "HighchartsChart", "children": "", "minHeight": "400px", "options": chart},
                {"name": "InsightsHeader", "type": "Paragraph", "children": "", "text": "Key Insights",
                 "style": {"fontSize": "18px", "fontWeight": "bold", "marginTop": "25px", "marginBottom": "10px", "color": "#1e293b"}},
                {"name": "Insights", "type": "Markdown", "children": "", "text": narrative_text,
                 "style": {"fontSize": "14px", "lineHeight": "1.6", "color": "#374151"}},
                {"name": "TableHeader", "type": "Paragraph", "children": "", "text": "Detailed Data",
                 "style": {"fontSize": "18px", "fontWeight": "bold", "marginTop": "25px", "marginBottom": "15px", "color": "#1e293b"}},
                {"name": "ResultsTable", "type": "DataTable", "children": "", "columns": columns, "data": table_data}
            ]
        },
        "inputVariables": []
    }

    try:
        html = wire_layout(layout, {})
    except Exception as e:
        html = f"<div>Error: {e}</div>"

    # Summary
    if breakout1 and breakout2:
        summary = f"Analyzed {len(metrics)} metric(s) by {get_dim_label(breakout1)} and {get_dim_label(breakout2)}."
    elif breakout1:
        summary = f"Analyzed {len(metrics)} metric(s) across {len(df)} {get_dim_label(breakout1)} segments."
    else:
        summary = f"Analyzed {len(metrics)} metric(s) across {int(df['respondent_count'].iloc[0]):,} respondents."

    return SkillOutput(
        final_prompt=summary,
        narrative=narrative_text,
        visualizations=[SkillVisualization(title="GSS Survey Explorer", layout=html)]
    )
