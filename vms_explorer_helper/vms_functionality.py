"""
VMS Explorer Functionality
Core analysis logic for the VMS Survey Explorer skill
"""
import pandas as pd
import jinja2
from skill_framework import SkillInput, SkillOutput, SkillVisualization, ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient
from ar_analytics import ArUtils

from .vms_config import (
    DATABASE_ID, TABLE_NAME, METRIC_GROUPS, NUMERIC_METRICS, ALL_METRICS,
    DIMENSIONS, METRIC_LABELS, DIMENSION_LABELS, METRIC_GROUP_LABELS,
    CALCULATED_METRICS, BRAND_METRICS, RESPONDENT_METRICS
)

# Reckitt brand colors
BRAND_PINK = "#E40046"  # Reckitt pink
BRAND_SLATE = "#415A6C"


def get_label(metric):
    return METRIC_LABELS.get(metric, metric.replace('_', ' ').title())


def get_dim_label(dim):
    return DIMENSION_LABELS.get(dim, dim.replace('_', ' ').title())


def resolve_metrics(metrics_input):
    """Resolve metric input - could be a group name, single metric, or list"""
    if not metrics_input:
        return METRIC_GROUPS["brain_health_interests"]
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
    """Clean and validate breakout dimension"""
    if not breakout or str(breakout).lower() in ['none', '', 'null', 'na']:
        return None
    if breakout in DIMENSIONS:
        return breakout
    return None


def build_filter_sql(filters):
    """Build SQL filter clause from other_filters parameter"""
    if not filters:
        return "", []

    filter_conditions = []
    filter_display = []

    for f in filters:
        if isinstance(f, dict) and 'dim' in f:
            dim = f['dim']
            op = f.get('op', '=')
            values = f.get('val')

            if values is None:
                continue

            # Build SQL condition
            if isinstance(values, list) and values:
                if len(values) == 1:
                    filter_conditions.append(f"{dim} = '{values[0]}'")
                    filter_display.append(f"{get_dim_label(dim)}: {values[0]}")
                else:
                    values_str = "', '".join(str(v) for v in values)
                    filter_conditions.append(f"{dim} IN ('{values_str}')")
                    filter_display.append(f"{get_dim_label(dim)}: {', '.join(str(v) for v in values)}")
            elif isinstance(values, str):
                filter_conditions.append(f"{dim} = '{values}'")
                filter_display.append(f"{get_dim_label(dim)}: {values}")
            elif isinstance(values, (int, float)):
                filter_conditions.append(f"{dim} {op} {values}")
                filter_display.append(f"{get_dim_label(dim)} {op} {values}")

    if filter_conditions:
        return " AND " + " AND ".join(filter_conditions), filter_display

    return "", []


def build_param_info(metrics, breakout1, breakout2, filter_display):
    """Build parameter display descriptions for pills"""
    param_info = []

    # Metrics pill
    metric_labels = [get_label(m) for m in metrics[:3]]
    if len(metrics) > 3:
        metric_labels.append(f"+{len(metrics) - 3} more")
    param_info.append(ParameterDisplayDescription(
        key="metrics",
        value=f"Metrics: {', '.join(metric_labels)}"
    ))

    # Breakout pills
    breakouts = []
    if breakout1:
        breakouts.append(get_dim_label(breakout1))
    if breakout2:
        breakouts.append(get_dim_label(breakout2))
    if breakouts:
        param_info.append(ParameterDisplayDescription(
            key="breakouts",
            value=f"Breakouts: {', '.join(breakouts)}"
        ))

    # Filter pills
    if filter_display:
        param_info.append(ParameterDisplayDescription(
            key="filters",
            value=f"Filters: {'; '.join(filter_display)}"
        ))

    return param_info


def run_vms_analysis(parameters: SkillInput) -> SkillOutput:
    """Main analysis function for VMS Explorer"""

    # Extract parameters
    metrics_input = parameters.arguments.metrics
    breakout1 = parameters.arguments.breakout_dimension
    breakout2 = getattr(parameters.arguments, 'breakout_dimension_2', None)
    filters = parameters.arguments.other_filters or []

    # Resolve and validate metrics
    metrics = resolve_metrics(metrics_input)
    metrics = [m for m in metrics if m in ALL_METRICS]
    if not metrics:
        metrics = METRIC_GROUPS["brain_health_interests"]

    # Clean breakouts
    breakout1 = clean_breakout(breakout1)
    breakout2 = clean_breakout(breakout2)

    if breakout2 and not breakout1:
        breakout1, breakout2 = breakout2, None
    if breakout1 and breakout2 and breakout1 == breakout2:
        breakout2 = None

    print(f"DEBUG: Metrics: {metrics}")
    print(f"DEBUG: Breakout1: {breakout1}, Breakout2: {breakout2}")

    # Classify requested metrics
    brand_metrics_requested = [m for m in metrics if m in BRAND_METRICS]
    respondent_metrics_requested = [m for m in metrics if m in RESPONDENT_METRICS]
    calculated_metrics_requested = [m for m in metrics if m in CALCULATED_METRICS]
    numeric_metrics_requested = [m for m in metrics if m in NUMERIC_METRICS]

    group_cols = [b for b in [breakout1, breakout2] if b]
    has_brand_breakout = any(b in ['brand_name', 'brand_category'] for b in group_cols)

    # Build metric select expressions
    def brand_metric_select(metric):
        """Brand metrics: use SUM/COUNT(*) to include NULLs in denominator"""
        return f"SUM(CAST({metric} AS DOUBLE)) * 100.0 / COUNT(*) AS {metric}"

    def respondent_metric_select(metric):
        """Respondent metrics: always AVG (dedup handled by CTE when needed)"""
        if metric in NUMERIC_METRICS:
            return f"AVG({metric}) AS {metric}"
        return f"AVG({metric}) * 100 AS {metric}"

    def calc_metric_select(metric):
        calc_def = CALCULATED_METRICS[metric]
        if calc_def.get("is_pct"):
            return f"{calc_def['sql']} * 100 AS {metric}"
        return f"{calc_def['sql']} AS {metric}"

    # Determine query strategy
    # is_mixed_query tracks whether filters/GROUP BY are handled inside the query branches
    is_mixed_query = False

    if brand_metrics_requested and respondent_metrics_requested and has_brand_breakout:
        # Mixed query with brand breakout: need two CTEs
        # CTE1: respondent-level metrics deduped, then joined back
        # CTE2: brand-level metrics from all rows
        resp_metric_selects = [respondent_metric_select(m) for m in respondent_metrics_requested]
        brand_metric_selects = [brand_metric_select(m) for m in brand_metrics_requested]
        calc_selects = [calc_metric_select(m) for m in calculated_metrics_requested]

        resp_cols = ', '.join(respondent_metrics_requested)
        brand_sel = ', '.join(brand_metric_selects)
        resp_sel = ', '.join(resp_metric_selects)
        all_selects = ', '.join(brand_metric_selects + resp_metric_selects + calc_selects)

        sql_query = f"""
            WITH resp_dedup AS (
                SELECT DISTINCT respondent_id, {', '.join(group_cols)}, {resp_cols}
                FROM {TABLE_NAME} WHERE 1=1{{filter_placeholder}}
            ),
            resp_agg AS (
                SELECT {', '.join(group_cols)}, {resp_sel}
                FROM resp_dedup
                GROUP BY {', '.join(group_cols)}
            ),
            brand_agg AS (
                SELECT {', '.join(group_cols)}, COUNT(*) AS respondent_count, {brand_sel}{', ' + ', '.join(calc_selects) if calc_selects else ''}
                FROM {TABLE_NAME} WHERE 1=1{{filter_placeholder}}
                GROUP BY {', '.join(group_cols)}
            )
            SELECT b.{', b.'.join(group_cols)}, b.respondent_count,
                   {', '.join(f'b.{m}' for m in brand_metrics_requested)},
                   {', '.join(f'r.{m}' for m in respondent_metrics_requested)}
                   {', ' + ', '.join(f'b.{m}' for m in calculated_metrics_requested) if calculated_metrics_requested else ''}
            FROM brand_agg b
            JOIN resp_agg r ON {' AND '.join(f'b.{g} = r.{g}' for g in group_cols)}
            ORDER BY {metrics[0]} DESC
        """
        # For mixed queries, filters are embedded in CTEs, not appended
        filter_sql, filter_display = build_filter_sql(filters)
        sql_query = sql_query.replace("{filter_placeholder}", filter_sql)
        param_info = build_param_info(metrics, breakout1, breakout2, filter_display)

        is_mixed_query = True
        print(f"DEBUG: SQL (mixed): {sql_query}")

    elif has_brand_breakout and not brand_metrics_requested and respondent_metrics_requested:
        # Respondent-level metrics only, but with brand breakout (e.g. needs by brand_category)
        # Must deduplicate by respondent per breakout group
        # Filters go inside the CTE so we filter before dedup
        resp_metric_selects = [respondent_metric_select(m) for m in respondent_metrics_requested]
        calc_selects = [calc_metric_select(m) for m in calculated_metrics_requested]
        all_selects = resp_metric_selects + calc_selects

        filter_sql, filter_display = build_filter_sql(filters)
        param_info = build_param_info(metrics, breakout1, breakout2, filter_display)

        sql_query = f"""
            WITH unique_respondents AS (
                SELECT DISTINCT respondent_id, {', '.join(group_cols)}, {', '.join(respondent_metrics_requested)}
                FROM {TABLE_NAME} WHERE 1=1{filter_sql}
            )
            SELECT {', '.join(group_cols)}, COUNT(*) AS respondent_count, {', '.join(all_selects)}
            FROM unique_respondents
            GROUP BY {', '.join(group_cols)} ORDER BY {metrics[0]} DESC
        """
        # Mark as handled so the generic filter/group-by block below skips
        is_mixed_query = True

    elif brand_metrics_requested and not has_brand_breakout:
        # Brand metrics without brand breakout (e.g. overall brand_aware)
        metric_selects = []
        for m in metrics:
            if m in BRAND_METRICS:
                metric_selects.append(brand_metric_select(m))
            elif m in CALCULATED_METRICS:
                metric_selects.append(calc_metric_select(m))
            elif m in NUMERIC_METRICS:
                metric_selects.append(f"AVG({m}) AS {m}")
            else:
                metric_selects.append(f"AVG({m}) * 100 AS {m}")

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

    else:
        # Pure respondent-level metrics without brand breakout - deduplicate
        metric_selects = []
        for m in metrics:
            if m in CALCULATED_METRICS:
                metric_selects.append(calc_metric_select(m))
            elif m in NUMERIC_METRICS:
                metric_selects.append(f"AVG({m}) AS {m}")
            else:
                metric_selects.append(f"AVG({m}) * 100 AS {m}")

        dedup_cols = [m for m in metrics if m not in CALCULATED_METRICS]
        if group_cols:
            sql_query = f"""
            WITH unique_respondents AS (
                SELECT DISTINCT respondent_id, {', '.join(group_cols)}, {', '.join(dedup_cols)}
                FROM {TABLE_NAME} WHERE 1=1
            )
            SELECT {', '.join(group_cols)}, COUNT(*) AS respondent_count, {', '.join(metric_selects)}
            FROM unique_respondents WHERE 1=1
            """
        else:
            sql_query = f"""
            WITH unique_respondents AS (
                SELECT DISTINCT respondent_id, {', '.join(dedup_cols)}
                FROM {TABLE_NAME} WHERE 1=1
            )
            SELECT COUNT(*) AS respondent_count, {', '.join(metric_selects)}
            FROM unique_respondents WHERE 1=1
            """

    # For simple queries (no special CTE handling), apply filters and GROUP BY here
    # The mixed and respondent-with-brand-breakout cases handle these internally
    if not is_mixed_query:
        filter_sql, filter_display = build_filter_sql(filters)
        sql_query += filter_sql
        param_info = build_param_info(metrics, breakout1, breakout2, filter_display)

        if group_cols:
            sql_query += f" GROUP BY {', '.join(group_cols)} ORDER BY {metrics[0]} DESC"

    print(f"DEBUG: SQL: {sql_query}")

    # Execute query
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

    # Build output
    is_pct = all(m not in NUMERIC_METRICS for m in metrics)
    suffix = "%" if is_pct else ""

    # Title
    metric_names = [get_label(m) for m in metrics]

    # Determine metric group name if applicable
    metric_group_title = None
    for group_name, group_metrics in METRIC_GROUPS.items():
        if set(metrics) == set(group_metrics):
            metric_group_title = METRIC_GROUP_LABELS.get(group_name, group_name.replace('_', ' ').title())
            break

    if metric_group_title:
        title = metric_group_title
    elif len(metric_names) == 1:
        title = metric_names[0]
    elif len(metric_names) == 2:
        title = f"{metric_names[0]} & {metric_names[1]}"
    elif len(metric_names) <= 4:
        title = ", ".join(metric_names[:-1]) + f" & {metric_names[-1]}"
    else:
        first_metric = metrics[0]
        if 'interest' in first_metric:
            title = "Brain Health Interests"
        elif 'psycho' in first_metric:
            title = "Consumer Psychographics"
        elif 'need' in first_metric:
            title = "Primary Needs"
        elif 'brand' in first_metric or 'aware' in first_metric:
            title = "Brand Metrics"
        else:
            title = "VMS Survey Metrics"

    subtitle = ""
    if breakout1:
        subtitle = f"by {get_dim_label(breakout1)}"
    if breakout2:
        subtitle += f" and {get_dim_label(breakout2)}"

    # Build chart
    chart = build_chart(df, metrics, breakout1, breakout2, is_pct, suffix)

    # Build table
    columns, table_data = build_table(df, metrics, breakout1, breakout2, suffix)

    # Build narrative
    narrative_text = build_narrative(df, metrics, breakout1, breakout2, suffix)

    # Build layout with Reckitt branding
    layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px", "fontFamily": "system-ui, -apple-system, sans-serif", "backgroundColor": "#ffffff"},
            "children": [
                {
                    "name": "HeaderContainer",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "backgroundColor": BRAND_PINK,
                        "padding": "20px 24px",
                        "borderRadius": "8px",
                        "marginBottom": "24px"
                    }
                },
                {
                    "name": "MainTitle",
                    "type": "Header",
                    "children": "",
                    "text": title,
                    "parentId": "HeaderContainer",
                    "style": {
                        "fontSize": "22px",
                        "fontWeight": "600",
                        "color": "#ffffff",
                        "margin": "0"
                    }
                },
                {
                    "name": "Subtitle",
                    "type": "Paragraph",
                    "children": "",
                    "text": subtitle if subtitle else "Overall Analysis",
                    "parentId": "HeaderContainer",
                    "style": {
                        "fontSize": "14px",
                        "color": "#fecdd3",
                        "marginTop": "4px"
                    }
                },
                {
                    "name": "Chart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "400px",
                    "options": chart
                },
                {
                    "name": "TableHeader",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Detailed Results",
                    "style": {
                        "fontSize": "16px",
                        "fontWeight": "600",
                        "marginTop": "28px",
                        "marginBottom": "12px",
                        "color": BRAND_SLATE
                    }
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

    # Build facts dataframe for insights
    facts_df = build_facts_df(df, metrics, breakout1, breakout2, suffix)
    insights_dfs = [facts_df]

    # Render prompts with facts
    facts_list = [facts_df.to_dict(orient='records')]

    insight_template = jinja2.Template(parameters.arguments.insight_prompt).render(facts=facts_list)
    max_response_prompt = jinja2.Template(parameters.arguments.max_prompt).render(facts=facts_list)

    # Generate insights using LLM
    ar_utils = ArUtils()
    generated_insights = ar_utils.get_llm_response(insight_template)

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=generated_insights,
        visualizations=[SkillVisualization(title="VMS Survey Explorer", layout=html)],
        insights_dfs=insights_dfs,
        parameter_display_descriptions=param_info
    )


def build_chart(df, metrics, breakout1, breakout2, is_pct, suffix):
    """Build Highcharts configuration based on breakout configuration"""

    # Reckitt-inspired colors
    colors = [BRAND_PINK, BRAND_SLATE, "#60a5fa", "#34d399", "#fbbf24", "#a78bfa", "#f87171", "#2dd4bf"]

    if not breakout1:
        # No breakout - column chart
        categories = [get_label(m) for m in metrics]
        values = [round(float(df[m].iloc[0]), 1) if pd.notna(df[m].iloc[0]) else 0 for m in metrics]
        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 380},
            "title": {"text": ""},
            "xAxis": {"categories": categories, "labels": {"style": {"fontSize": "11px", "color": BRAND_SLATE}, "rotation": -45 if len(categories) > 5 else 0}},
            "yAxis": {"title": {"text": "%" if is_pct else "Value", "style": {"color": BRAND_SLATE}}, "max": 100 if is_pct else None, "min": 0},
            "series": [{"name": "Value", "data": values, "colorByPoint": True, "colors": colors}],
            "legend": {"enabled": False},
            "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": BRAND_PINK},
            "plotOptions": {"column": {"dataLabels": {"enabled": True, "format": "{y:.1f}" + suffix, "style": {"fontWeight": "500", "color": BRAND_SLATE}}}}
        }

    elif breakout1 and not breakout2:
        # Single breakout - column chart
        categories = df[breakout1].astype(str).tolist()
        series = []
        for i, m in enumerate(metrics):
            series.append({
                "name": get_label(m),
                "data": df[m].fillna(0).round(1).tolist(),
                "color": colors[i % len(colors)]
            })
        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 400},
            "title": {"text": ""},
            "xAxis": {"categories": categories, "title": {"text": get_dim_label(breakout1), "style": {"color": BRAND_SLATE}}, "labels": {"style": {"fontSize": "11px", "color": BRAND_SLATE}, "rotation": -45 if len(categories) > 8 else 0}},
            "yAxis": {"title": {"text": "%" if is_pct else "Value", "style": {"color": BRAND_SLATE}}, "max": 100 if is_pct else None, "min": 0},
            "series": series,
            "legend": {"enabled": len(metrics) > 1},
            "credits": {"enabled": False},
            "tooltip": {"shared": True, "valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": BRAND_PINK},
            "plotOptions": {"column": {"dataLabels": {"enabled": len(df) <= 8, "format": "{y:.1f}" + suffix, "style": {"fontSize": "11px"}}}}
        }

    else:
        # Dual breakout - grouped column chart
        pri_vals = df[breakout1].unique().tolist()
        sec_vals = df[breakout2].unique().tolist()
        metric = metrics[0]
        series = []
        for i, sv in enumerate(sec_vals):
            data = []
            for pv in pri_vals:
                mask = (df[breakout1] == pv) & (df[breakout2] == sv)
                val = df.loc[mask, metric].iloc[0] if mask.any() else 0
                data.append(round(float(val), 1) if pd.notna(val) else 0)
            series.append({"name": str(sv), "data": data, "color": colors[i % len(colors)]})
        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 450},
            "title": {"text": get_label(metric), "style": {"fontSize": "16px", "color": BRAND_SLATE}},
            "xAxis": {"categories": [str(v) for v in pri_vals], "title": {"text": get_dim_label(breakout1), "style": {"color": BRAND_SLATE}}},
            "yAxis": {"title": {"text": "%" if is_pct else "Value", "style": {"color": BRAND_SLATE}}, "max": 100 if is_pct else None, "min": 0},
            "series": series,
            "legend": {"enabled": True, "title": {"text": get_dim_label(breakout2), "style": {"color": BRAND_SLATE}}},
            "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": BRAND_PINK},
            "plotOptions": {"column": {"dataLabels": {"enabled": len(pri_vals) <= 5, "format": "{y:.1f}" + suffix}}}
        }


def build_table(df, metrics, breakout1, breakout2, suffix):
    """Build table columns and data"""
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

    return columns, table_data


def build_facts_df(df, metrics, breakout1, breakout2, suffix):
    """Build facts dataframe for LLM prompts"""
    facts = []

    if not breakout1:
        total = int(df['respondent_count'].iloc[0])
        facts.append({'fact_type': 'overview', 'detail': f'Total respondents: {total:,}'})
        for m in metrics:
            val = df[m].iloc[0]
            if pd.notna(val):
                facts.append({
                    'fact_type': 'metric',
                    'metric': get_label(m),
                    'value': f'{val:.1f}{suffix}',
                    'respondents': total
                })
    else:
        for m in metrics[:3]:
            if m in df.columns:
                mx, mn = df[m].max(), df[m].min()
                if pd.notna(mx) and pd.notna(mn):
                    mx_seg = df.loc[df[m].idxmax(), breakout1]
                    mn_seg = df.loc[df[m].idxmin(), breakout1]
                    facts.append({
                        'fact_type': 'comparison',
                        'metric': get_label(m),
                        'highest_segment': str(mx_seg),
                        'highest_value': f'{mx:.1f}{suffix}',
                        'lowest_segment': str(mn_seg),
                        'lowest_value': f'{mn:.1f}{suffix}',
                        'gap': f'{mx - mn:.1f} points'
                    })

    return pd.DataFrame(facts)


def build_narrative(df, metrics, breakout1, breakout2, suffix):
    """Build insights narrative (50-100 words)"""
    parts = []

    if not breakout1:
        total = int(df['respondent_count'].iloc[0])
        vals = [(get_label(m), float(df[m].iloc[0])) for m in metrics if pd.notna(df[m].iloc[0])]
        vals.sort(key=lambda x: x[1], reverse=True)

        parts.append(f"Analysis of **{total:,}** VMS survey respondents shows **{vals[0][0]}** leads at **{vals[0][1]:.1f}{suffix}**")
        if len(vals) > 1:
            parts.append(f", while **{vals[-1][0]}** is lowest at **{vals[-1][1]:.1f}{suffix}**.")
            spread = vals[0][1] - vals[-1][1]
            parts.append(f" The **{spread:.1f} point spread** indicates meaningful variation in consumer priorities.")
        else:
            parts.append(".")

    elif breakout1 and not breakout2:
        num_segments = len(df)
        parts.append(f"Comparing **{num_segments}** {get_dim_label(breakout1)} segments:\n\n")

        for m in metrics[:2]:
            if m in df.columns:
                mx, mn = df[m].max(), df[m].min()
                if pd.notna(mx) and pd.notna(mn):
                    mx_seg = df.loc[df[m].idxmax(), breakout1]
                    mn_seg = df.loc[df[m].idxmin(), breakout1]
                    gap = mx - mn
                    parts.append(f"- **{get_label(m)}**: {mx_seg} leads at {mx:.1f}{suffix}, {mn_seg} lowest at {mn:.1f}{suffix} ({gap:.1f}pt gap)\n")

        if len(metrics) > 2:
            parts.append(f"\n*{len(metrics) - 2} additional metric(s) shown in table below.*")

    else:
        parts.append(f"Cross-analysis by **{get_dim_label(breakout1)}** and **{get_dim_label(breakout2)}**:\n\n")
        m = metrics[0]
        if m in df.columns:
            mx, mn = df[m].max(), df[m].min()
            if pd.notna(mx) and pd.notna(mn):
                mx_r, mn_r = df.loc[df[m].idxmax()], df.loc[df[m].idxmin()]
                gap = mx - mn
                parts.append(f"**{get_label(m)}** ranges from **{mn:.1f}{suffix}** ({mn_r[breakout1]} / {mn_r[breakout2]}) to **{mx:.1f}{suffix}** ({mx_r[breakout1]} / {mx_r[breakout2]}). ")
                parts.append(f"This **{gap:.1f} point gap** reveals significant variation across segments.")

    return "".join(parts)
