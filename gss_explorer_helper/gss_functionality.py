"""
GSS Explorer Functionality
Core analysis logic for the GSS Survey Explorer skill
"""
import pandas as pd
from skill_framework import SkillInput, SkillOutput, SkillVisualization
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient

from .gss_config import (
    DATABASE_ID, TABLE_NAME, METRIC_GROUPS, NUMERIC_METRICS, ALL_METRICS,
    DIMENSIONS, METRIC_LABELS, DIMENSION_LABELS
)


def get_label(metric):
    return METRIC_LABELS.get(metric, metric.replace('_', ' ').title())


def get_dim_label(dim):
    return DIMENSION_LABELS.get(dim, dim.replace('_', ' ').title())


def resolve_metrics(metrics_input):
    """Resolve metric input - could be a group name, single metric, or list"""
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
    """Clean and validate breakout dimension"""
    if not breakout or str(breakout).lower() in ['none', '', 'null', 'na']:
        return None
    if breakout in DIMENSIONS:
        return breakout
    return None


def run_gss_analysis(parameters: SkillInput) -> SkillOutput:
    """Main analysis function for GSS Explorer"""

    # Extract parameters
    metrics_input = parameters.arguments.metrics
    breakout1 = parameters.arguments.breakout_dimension
    breakout2 = getattr(parameters.arguments, 'breakout_dimension_2', None)
    filters = parameters.arguments.other_filters or []

    # Resolve and validate metrics
    metrics = resolve_metrics(metrics_input)
    metrics = [m for m in metrics if m in ALL_METRICS]
    if not metrics:
        metrics = METRIC_GROUPS["benefits"]

    # Clean breakouts
    breakout1 = clean_breakout(breakout1)
    breakout2 = clean_breakout(breakout2)

    if breakout2 and not breakout1:
        breakout1, breakout2 = breakout2, None
    if breakout1 and breakout2 and breakout1 == breakout2:
        breakout2 = None

    print(f"DEBUG: Metrics: {metrics}")
    print(f"DEBUG: Breakout1: {breakout1}, Breakout2: {breakout2}")

    # Build SQL query
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

    # Add filters
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
    title = ", ".join(metric_names[:3]) if len(metric_names) <= 3 else f"{len(metric_names)} Metrics"
    if breakout1:
        title += f" by {get_dim_label(breakout1)}"
    if breakout2:
        title += f" and {get_dim_label(breakout2)}"

    # Build chart
    chart = build_chart(df, metrics, breakout1, breakout2, is_pct, suffix)

    # Build table
    columns, table_data = build_table(df, metrics, breakout1, breakout2, suffix)

    # Build narrative
    narrative_text = build_narrative(df, metrics, breakout1, breakout2, suffix)

    # Build layout
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


def build_chart(df, metrics, breakout1, breakout2, is_pct, suffix):
    """Build Highcharts configuration based on breakout configuration"""

    if not breakout1:
        # No breakout - simple bar chart
        categories = [get_label(m) for m in metrics]
        values = [round(float(df[m].iloc[0]), 1) if pd.notna(df[m].iloc[0]) else 0 for m in metrics]
        return {
            "chart": {"type": "bar", "backgroundColor": "#ffffff", "height": max(300, len(metrics) * 50)},
            "title": {"text": ""}, "xAxis": {"categories": categories},
            "yAxis": {"title": {"text": "%" if is_pct else "Value"}, "max": 100 if is_pct else None, "min": 0},
            "series": [{"name": "Value", "data": values, "colorByPoint": True}],
            "legend": {"enabled": False}, "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"bar": {"dataLabels": {"enabled": True, "format": "{y:.1f}" + suffix}}}
        }

    elif breakout1 and not breakout2:
        # Single breakout
        categories = df[breakout1].astype(str).tolist()
        series = [{"name": get_label(m), "data": df[m].fillna(0).round(1).tolist()} for m in metrics]
        return {
            "chart": {"type": "bar", "backgroundColor": "#ffffff", "height": max(400, len(categories) * 30)},
            "title": {"text": ""}, "xAxis": {"categories": categories, "title": {"text": get_dim_label(breakout1)}},
            "yAxis": {"title": {"text": "%" if is_pct else "Value"}, "max": 100 if is_pct else None, "min": 0},
            "series": series, "legend": {"enabled": len(metrics) > 1}, "credits": {"enabled": False},
            "tooltip": {"shared": True, "valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"bar": {"dataLabels": {"enabled": len(df) <= 10, "format": "{y:.1f}" + suffix}}}
        }

    else:
        # Dual breakout - grouped column chart
        pri_vals = df[breakout1].unique().tolist()
        sec_vals = df[breakout2].unique().tolist()
        metric = metrics[0]
        series = []
        for sv in sec_vals:
            data = []
            for pv in pri_vals:
                mask = (df[breakout1] == pv) & (df[breakout2] == sv)
                val = df.loc[mask, metric].iloc[0] if mask.any() else 0
                data.append(round(float(val), 1) if pd.notna(val) else 0)
            series.append({"name": str(sv), "data": data})
        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 450},
            "title": {"text": get_label(metric)},
            "xAxis": {"categories": [str(v) for v in pri_vals], "title": {"text": get_dim_label(breakout1)}},
            "yAxis": {"title": {"text": "%" if is_pct else "Value"}, "max": 100 if is_pct else None, "min": 0},
            "series": series, "legend": {"enabled": True, "title": {"text": get_dim_label(breakout2)}},
            "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,1)", "useHTML": False},
            "plotOptions": {"column": {"dataLabels": {"enabled": len(pri_vals) <= 6, "format": "{y:.1f}" + suffix}}}
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


def build_narrative(df, metrics, breakout1, breakout2, suffix):
    """Build insights narrative"""
    parts = []

    if not breakout1:
        total = int(df['respondent_count'].iloc[0])
        parts.append(f"Based on **{total:,}** respondents:\n")
        vals = [(get_label(m), float(df[m].iloc[0])) for m in metrics if pd.notna(df[m].iloc[0])]
        vals.sort(key=lambda x: x[1], reverse=True)
        if vals:
            parts.append(f"- **Top:** {vals[0][0]} at {vals[0][1]:.1f}{suffix}")
            if len(vals) > 1:
                parts.append(f"- **Lowest:** {vals[-1][0]} at {vals[-1][1]:.1f}{suffix}")

    elif breakout1 and not breakout2:
        parts.append(f"Analysis across **{len(df)}** {get_dim_label(breakout1)} segments:\n")
        for m in metrics[:3]:
            if m in df.columns:
                mx, mn = df[m].max(), df[m].min()
                if pd.notna(mx) and pd.notna(mn):
                    mx_seg = df.loc[df[m].idxmax(), breakout1]
                    mn_seg = df.loc[df[m].idxmin(), breakout1]
                    parts.append(f"- **{get_label(m)}:** Highest in {mx_seg} ({mx:.1f}{suffix}), lowest in {mn_seg} ({mn:.1f}{suffix}). Gap: {mx-mn:.1f} pts.")

    else:
        parts.append(f"Cross-analysis by **{get_dim_label(breakout1)}** and **{get_dim_label(breakout2)}**:\n")
        m = metrics[0]
        if m in df.columns:
            mx, mn = df[m].max(), df[m].min()
            if pd.notna(mx) and pd.notna(mn):
                mx_r, mn_r = df.loc[df[m].idxmax()], df.loc[df[m].idxmin()]
                parts.append(f"- **{get_label(m)}** highest ({mx:.1f}{suffix}): {mx_r[breakout1]} / {mx_r[breakout2]}")
                parts.append(f"- **{get_label(m)}** lowest ({mn:.1f}{suffix}): {mn_r[breakout1]} / {mn_r[breakout2]}")

    return "\n".join(parts)
