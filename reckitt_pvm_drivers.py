"""
Reckitt Price-Volume-Mix Driver Analysis Skill

Uses net_revenue with units as volume to derive price for PVM decomposition.
Reckitt brands only (they have finance metrics).
"""

from __future__ import annotations
from skill_framework import skill, SkillParameter, SkillInput, SkillOutput, SkillVisualization, ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient
from ar_analytics import ArUtils
import pandas as pd
import numpy as np
import jinja2
import logging
import os

logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_ID = os.getenv('DATABASE_ID', '')
DATA_FILE = 'reckitt_surface_care_poc.csv'

# Colors
BRAND_BLUE = "#2563EB"
BRAND_SLATE = "#415A6C"
GREEN = "#4ade80"
RED = "#ef4444"

# Prompt templates
DEFAULT_MAX_PROMPT = """
Based on the following Price-Volume-Mix analysis:
{% for fact in facts %}
- {{ fact }}
{% endfor %}

Provide a concise executive summary (2-3 sentences) highlighting the key variance drivers.
"""

DEFAULT_INSIGHT_PROMPT = """
Analyze the following Price-Volume-Mix variance data:
{% for fact in facts %}
- {{ fact }}
{% endfor %}

Provide detailed insights covering:
1. Key variance drivers (Volume, Price, Mix)
2. Top contributing dimensions (brands, channels, regions)
3. Actionable recommendations for stakeholders

Format the insights in clear markdown with bullet points.
"""


def format_number(value, is_currency=True, decimals=1):
    """Format numbers with M/K/B abbreviations"""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return str(value)

    abs_value = abs(value)
    prefix = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000:
        formatted = f"{prefix}{abs_value / 1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        formatted = f"{prefix}{abs_value / 1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        formatted = f"{prefix}{abs_value / 1_000:.{decimals}f}K"
    else:
        formatted = f"{prefix}{abs_value:.{decimals}f}"

    if is_currency:
        formatted = f"${formatted}"

    return formatted


def format_display_name(name):
    """Format technical names to display names"""
    if not name:
        return name
    special_cases = {
        'net_revenue': 'Net Revenue',
        'gross_sales': 'Gross Sales',
        'gross_margin': 'Gross Margin',
        'sub_category': 'Sub-Category',
        'state_name': 'State',
    }
    if name.lower() in special_cases:
        return special_cases[name.lower()]
    return name.replace('_', ' ').title()


def build_filter_clause(filters):
    """Build SQL WHERE clause from filters"""
    clauses = []
    if filters:
        for filter_dict in filters:
            dim = filter_dict.get('dim')
            op = filter_dict.get('op', '=')
            val = filter_dict.get('val')
            if dim and val:
                if isinstance(val, list):
                    if len(val) == 1:
                        clauses.append(f"UPPER({dim}) {op} UPPER('{val[0]}')")
                    else:
                        val_str = ", ".join([f"UPPER('{v}')" for v in val])
                        clauses.append(f"UPPER({dim}) IN ({val_str})")
                elif isinstance(val, str):
                    clauses.append(f"UPPER({dim}) {op} UPPER('{val}')")
                else:
                    clauses.append(f"{dim} {op} {val}")
    return " AND " + " AND ".join(clauses) if clauses else ""


def run_pvm_analysis(client, metric, current_period, prior_period, breakout_dimensions, filters):
    """Run Price-Volume-Mix analysis"""

    filter_clause = build_filter_clause(filters)

    # Query current period data (Reckitt brands only)
    current_query = f"""
    SELECT brand, sub_category, segment, channel, state_name,
           SUM({metric}) as revenue,
           SUM(units) as units
    FROM read_csv('{DATA_FILE}')
    WHERE manufacturer = 'RECKITT BENCKISER'
    AND quarter = '{current_period}'
    {filter_clause}
    GROUP BY brand, sub_category, segment, channel, state_name
    """

    logger.info(f"Current period query: {current_query}")
    result = client.data.execute_sql_query(
        database_id=DATABASE_ID,
        sql_query=current_query,
        row_limit=50000
    )

    if not result.success or not hasattr(result, 'df') or result.df.empty:
        raise ValueError(f"No data found for current period: {current_period}")

    current_df = result.df.copy()

    # Query prior period data
    prior_query = f"""
    SELECT brand, sub_category, segment, channel, state_name,
           SUM({metric}) as revenue,
           SUM(units) as units
    FROM read_csv('{DATA_FILE}')
    WHERE manufacturer = 'RECKITT BENCKISER'
    AND quarter = '{prior_period}'
    {filter_clause}
    GROUP BY brand, sub_category, segment, channel, state_name
    """

    logger.info(f"Prior period query: {prior_query}")
    result = client.data.execute_sql_query(
        database_id=DATABASE_ID,
        sql_query=prior_query,
        row_limit=50000
    )

    if not result.success or not hasattr(result, 'df') or result.df.empty:
        raise ValueError(f"No data found for prior period: {prior_period}")

    prior_df = result.df.copy()

    # Calculate PVM
    actual_revenue = current_df['revenue'].sum()
    prior_revenue = prior_df['revenue'].sum()
    total_variance = actual_revenue - prior_revenue

    actual_units = current_df['units'].sum()
    prior_units = prior_df['units'].sum()

    mix_dimension = breakout_dimensions[0] if breakout_dimensions else 'brand'

    # Aggregate by mix dimension
    actual_by_dim = current_df.groupby(mix_dimension).agg({'revenue': 'sum', 'units': 'sum'}).reset_index()
    actual_by_dim['price'] = actual_by_dim['revenue'] / actual_by_dim['units'].replace(0, np.nan)
    actual_by_dim['price'] = actual_by_dim['price'].fillna(0)

    prior_by_dim = prior_df.groupby(mix_dimension).agg({'revenue': 'sum', 'units': 'sum'}).reset_index()
    prior_by_dim['price'] = prior_by_dim['revenue'] / prior_by_dim['units'].replace(0, np.nan)
    prior_by_dim['price'] = prior_by_dim['price'].fillna(0)

    merged = pd.merge(actual_by_dim, prior_by_dim, on=mix_dimension, how='outer', suffixes=('_actual', '_prior')).fillna(0)

    total_actual_units = actual_by_dim['units'].sum()
    total_prior_units = prior_by_dim['units'].sum()

    # Mix Impact
    mix_impact = 0
    for _, row in merged.iterrows():
        actual_share = row['units_actual'] / total_actual_units if total_actual_units > 0 else 0
        prior_share = row['units_prior'] / total_prior_units if total_prior_units > 0 else 0
        share_change = actual_share - prior_share
        mix_impact += share_change * row['price_prior'] * total_actual_units

    # Volume Impact
    prior_avg_price = prior_revenue / total_prior_units if total_prior_units > 0 else 0
    volume_impact = (total_actual_units - total_prior_units) * prior_avg_price

    # Price Impact (residual)
    price_impact = total_variance - volume_impact - mix_impact

    pvm_results = {
        'starting_value': prior_revenue,
        'volume_impact': volume_impact,
        'price_impact': price_impact,
        'mix_impact': mix_impact,
        'ending_value': actual_revenue,
        'total_variance': total_variance,
        'prior_units': prior_units,
        'actual_units': actual_units,
        'prior_price': prior_avg_price,
        'actual_price': actual_revenue / actual_units if actual_units > 0 else 0
    }

    # Generate facts
    facts = []
    pct_change = (total_variance / prior_revenue * 100) if prior_revenue != 0 else 0
    facts.append(f"Total {format_display_name(metric)} variance: {format_number(total_variance)} ({pct_change:+.1f}%)")

    if total_variance != 0:
        facts.append(f"Volume impact: {format_number(volume_impact)} ({volume_impact/abs(total_variance)*100:.1f}% of variance)")
        facts.append(f"Price impact: {format_number(price_impact)} ({price_impact/abs(total_variance)*100:.1f}% of variance)")
        facts.append(f"Mix impact: {format_number(mix_impact)} ({mix_impact/abs(total_variance)*100:.1f}% of variance)")

    # Dimensional breakouts
    breakout_data = {}
    for dim in breakout_dimensions[:3]:
        if dim in current_df.columns:
            actual_agg = current_df.groupby(dim)['revenue'].sum().reset_index()
            actual_agg.columns = [dim, 'actual']
            prior_agg = prior_df.groupby(dim)['revenue'].sum().reset_index()
            prior_agg.columns = [dim, 'prior']

            dim_merged = pd.merge(actual_agg, prior_agg, on=dim, how='outer').fillna(0)
            dim_merged['variance'] = dim_merged['actual'] - dim_merged['prior']
            dim_merged['variance_pct'] = (dim_merged['variance'] / dim_merged['prior'].replace(0, np.nan) * 100).fillna(0)
            dim_merged = dim_merged.sort_values('variance', key=abs, ascending=False).head(10)
            breakout_data[dim] = dim_merged

            for _, row in dim_merged.head(3).iterrows():
                direction = "increased" if row['variance'] > 0 else "decreased"
                facts.append(f"{format_display_name(dim)} '{row[dim]}': {format_number(row['variance'])} ({row['variance_pct']:+.1f}%) - {direction}")

    return pvm_results, facts, breakout_data


def build_waterfall_chart(pvm_results, current_period, prior_period, metric):
    """Build waterfall chart Highcharts config - embedded directly"""

    def get_color(value):
        return GREEN if value >= 0 else RED

    volume_val = pvm_results['volume_impact']
    price_val = pvm_results['price_impact']
    mix_val = pvm_results['mix_impact']

    scale = 1_000_000 if abs(pvm_results['starting_value']) >= 1_000_000 else 1_000
    scale_label = 'M' if scale == 1_000_000 else 'K'

    return {
        "chart": {"type": "waterfall", "backgroundColor": "#ffffff", "height": 450},
        "title": {"text": ""},
        "xAxis": {
            "categories": [prior_period, "Volume", "Price", "Mix", current_period],
            "labels": {"style": {"fontSize": "12px", "color": BRAND_SLATE}}
        },
        "yAxis": {
            "title": {"text": f"{format_display_name(metric)} (${scale_label})", "style": {"color": BRAND_SLATE}},
            "labels": {"format": "${value:,.0f}" + scale_label}
        },
        "series": [{
            "name": format_display_name(metric),
            "data": [
                {"name": prior_period, "y": round(pvm_results['starting_value'] / scale, 2), "color": BRAND_BLUE},
                {"name": "Volume", "y": round(volume_val / scale, 2), "color": get_color(volume_val)},
                {"name": "Price", "y": round(price_val / scale, 2), "color": get_color(price_val)},
                {"name": "Mix", "y": round(mix_val / scale, 2), "color": get_color(mix_val)},
                {"name": current_period, "isSum": True, "color": BRAND_BLUE}
            ],
            "dataLabels": {
                "enabled": True,
                "format": "${point.y:,.2f}" + scale_label,
                "style": {"fontWeight": "bold", "color": "#000000", "textOutline": "none"}
            }
        }],
        "legend": {"enabled": False},
        "credits": {"enabled": False},
        "tooltip": {"shared": True, "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": BRAND_BLUE}
    }


def build_bar_chart(breakout_df, dim, current_period, prior_period):
    """Build horizontal bar chart for dimensional breakout"""

    scale = 1_000_000 if breakout_df['actual'].abs().max() >= 1_000_000 else 1_000
    scale_label = 'M' if scale == 1_000_000 else 'K'

    categories = breakout_df[dim].astype(str).tolist()
    actual_data = [round(x / scale, 2) for x in breakout_df['actual'].tolist()]
    prior_data = [round(x / scale, 2) for x in breakout_df['prior'].tolist()]

    return {
        "chart": {"type": "bar", "backgroundColor": "#ffffff", "height": 400},
        "title": {"text": ""},
        "xAxis": {
            "categories": categories,
            "title": {"text": format_display_name(dim), "style": {"color": BRAND_SLATE}},
            "labels": {"style": {"fontSize": "11px", "color": BRAND_SLATE}}
        },
        "yAxis": {
            "title": {"text": f"Net Revenue (${scale_label})", "style": {"color": BRAND_SLATE}},
            "labels": {"format": "${value:,.0f}" + scale_label}
        },
        "series": [
            {"name": current_period, "data": actual_data, "color": BRAND_BLUE},
            {"name": prior_period, "data": prior_data, "color": "#94a3b8"}
        ],
        "legend": {"enabled": True, "align": "center", "verticalAlign": "bottom"},
        "credits": {"enabled": False},
        "tooltip": {"shared": True, "valueSuffix": scale_label, "backgroundColor": "rgba(255,255,255,0.95)"}
    }


def build_summary_table(pvm_results, current_period, prior_period):
    """Build summary data table"""

    columns = [
        {"name": "Metric"},
        {"name": current_period},
        {"name": prior_period},
        {"name": "Change"},
        {"name": "Change %"}
    ]

    data = [
        [
            "Net Revenue",
            format_number(pvm_results['ending_value']),
            format_number(pvm_results['starting_value']),
            format_number(pvm_results['total_variance']),
            f"{pvm_results['total_variance'] / pvm_results['starting_value'] * 100:+.1f}%" if pvm_results['starting_value'] != 0 else "N/A"
        ],
        [
            "Units",
            f"{pvm_results['actual_units']:,.0f}",
            f"{pvm_results['prior_units']:,.0f}",
            f"{pvm_results['actual_units'] - pvm_results['prior_units']:+,.0f}",
            f"{(pvm_results['actual_units'] - pvm_results['prior_units']) / pvm_results['prior_units'] * 100:+.1f}%" if pvm_results['prior_units'] != 0 else "N/A"
        ],
        [
            "Price ($/Unit)",
            f"${pvm_results['actual_price']:.2f}",
            f"${pvm_results['prior_price']:.2f}",
            f"${pvm_results['actual_price'] - pvm_results['prior_price']:+.2f}",
            f"{(pvm_results['actual_price'] - pvm_results['prior_price']) / pvm_results['prior_price'] * 100:+.1f}%" if pvm_results['prior_price'] != 0 else "N/A"
        ],
        ["", "", "", "", ""],
        [
            "Volume Impact", "", "",
            format_number(pvm_results['volume_impact']),
            f"{pvm_results['volume_impact'] / abs(pvm_results['total_variance']) * 100:.1f}% of var" if pvm_results['total_variance'] != 0 else "N/A"
        ],
        [
            "Price Impact", "", "",
            format_number(pvm_results['price_impact']),
            f"{pvm_results['price_impact'] / abs(pvm_results['total_variance']) * 100:.1f}% of var" if pvm_results['total_variance'] != 0 else "N/A"
        ],
        [
            "Mix Impact", "", "",
            format_number(pvm_results['mix_impact']),
            f"{pvm_results['mix_impact'] / abs(pvm_results['total_variance']) * 100:.1f}% of var" if pvm_results['total_variance'] != 0 else "N/A"
        ]
    ]

    return columns, data


@skill(
    name="Reckitt PVM Drivers",
    llm_name="reckitt_pvm_drivers",
    description=(
        "Price-Volume-Mix variance analysis for Reckitt Surface Care brands. "
        "Decomposes net revenue changes into Volume, Price, and Mix impacts using units as volume metric. "
        "Compares current period vs prior period (e.g., Q2'26 vs Q2'25 for YoY analysis)."
    ),
    capabilities=(
        "1) Price-Volume-Mix decomposition: Volume Impact, Price Impact, Mix Impact. "
        "2) Waterfall chart visualization showing variance bridge. "
        "3) Dimensional breakout by brand, sub_category, segment, channel, state. "
        "4) Top contributor identification."
    ),
    limitations=(
        "Only analyzes Reckitt Benckiser brands (LYSOL, LYSOL POWER, LYSOL LAUNDRY SANITIZER). "
        "Data covers Q2'25 through Q2'26 (5 quarters)."
    ),
    example_questions=(
        "What drove the net revenue change in Q2'26 vs Q2'25? | "
        "Show me price-volume-mix analysis for Q4'25 vs Q3'25. | "
        "Which brands contributed most to revenue growth?"
    ),
    parameter_guidance=(
        "METRIC: net_revenue (default), gross_sales, or gross_margin. "
        "CURRENT_PERIOD: Quarter to analyze (Q2'26, Q1'26, Q4'25, Q3'25, Q2'25). "
        "PRIOR_PERIOD: Comparison quarter (Q2'25 for YoY). "
        "BREAKOUT_DIMENSIONS: brand, sub_category, segment, channel, state_name."
    ),
    parameters=[
        SkillParameter(name="metric", description="Revenue metric: net_revenue, gross_sales, gross_margin", default_value="net_revenue"),
        SkillParameter(name="current_period", description="Current quarter (e.g., Q2'26)", default_value="Q2'26"),
        SkillParameter(name="prior_period", description="Prior quarter for comparison (e.g., Q2'25)", default_value="Q2'25"),
        SkillParameter(name="breakout_dimensions", constrained_to="dimensions", is_multi=True, description="Dimensions for breakout: brand, sub_category, segment, channel, state_name"),
        SkillParameter(name="other_filters", constrained_to="filters", is_multi=True, description="Filters to narrow analysis"),
        SkillParameter(name="max_prompt", parameter_type="prompt", description="Prompt for summary", default_value=DEFAULT_MAX_PROMPT),
        SkillParameter(name="insight_prompt", parameter_type="prompt", description="Prompt for insights", default_value=DEFAULT_INSIGHT_PROMPT)
    ]
)
def reckitt_pvm_drivers(parameters: SkillInput) -> SkillOutput:
    """Reckitt Price-Volume-Mix Driver Analysis"""

    logger.info(f"Skill received parameters: {parameters.arguments}")

    # Extract parameters
    metric = getattr(parameters.arguments, 'metric', 'net_revenue') or 'net_revenue'
    current_period = getattr(parameters.arguments, 'current_period', "Q2'26") or "Q2'26"
    prior_period = getattr(parameters.arguments, 'prior_period', "Q2'25") or "Q2'25"
    breakout_dimensions = getattr(parameters.arguments, 'breakout_dimensions', None) or ['brand', 'sub_category', 'channel']
    other_filters = getattr(parameters.arguments, 'other_filters', []) or []
    max_prompt = getattr(parameters.arguments, 'max_prompt', DEFAULT_MAX_PROMPT)
    insight_prompt = getattr(parameters.arguments, 'insight_prompt', DEFAULT_INSIGHT_PROMPT)

    # Get client
    try:
        client = AnswerRocketClient()
    except Exception as e:
        logger.error(f"Failed to initialize AnswerRocketClient: {e}")
        return SkillOutput(
            final_prompt=f"Failed to initialize client: {str(e)}",
            narrative=f"**Error**: Could not connect. {str(e)}",
            visualizations=[]
        )

    # Run analysis
    try:
        pvm_results, facts, breakout_data = run_pvm_analysis(
            client=client,
            metric=metric,
            current_period=current_period,
            prior_period=prior_period,
            breakout_dimensions=breakout_dimensions,
            filters=other_filters
        )
    except ValueError as e:
        logger.error(f"Analysis failed: {e}")
        return SkillOutput(
            final_prompt=f"Analysis could not be completed: {str(e)}",
            narrative=f"**Error**: {str(e)}",
            visualizations=[]
        )

    # Generate insights
    try:
        ar_utils = ArUtils()
        insight_template = jinja2.Template(insight_prompt).render(facts=facts)
        insights = ar_utils.get_llm_response(insight_template)
    except:
        insights = "**Price-Volume-Mix Analysis Complete**\n\n" + "\n".join([f"- {f}" for f in facts])

    # Build visualizations
    viz_list = []

    # Tab 1: PVM Waterfall
    waterfall_chart = build_waterfall_chart(pvm_results, current_period, prior_period, metric)
    table_cols, table_data = build_summary_table(pvm_results, current_period, prior_period)

    layout1 = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px", "fontFamily": "system-ui, -apple-system, sans-serif", "backgroundColor": "#ffffff"},
            "children": [
                {"name": "Header", "type": "FlexContainer", "children": "", "direction": "column",
                 "style": {"backgroundColor": BRAND_BLUE, "padding": "20px 24px", "borderRadius": "8px", "marginBottom": "24px"}},
                {"name": "Title", "type": "Header", "children": "", "text": f"{format_display_name(metric)} Variance Analysis",
                 "parentId": "Header", "style": {"fontSize": "22px", "fontWeight": "600", "color": "#ffffff", "margin": "0"}},
                {"name": "Subtitle", "type": "Paragraph", "children": "",
                 "text": f"{current_period} vs {prior_period} | Price-Volume-Mix Decomposition",
                 "parentId": "Header", "style": {"fontSize": "14px", "color": "#bfdbfe", "marginTop": "4px"}},
                {"name": "Chart", "type": "HighchartsChart", "children": "", "minHeight": "450px", "options": waterfall_chart},
                {"name": "TableHeader", "type": "Paragraph", "children": "", "text": "Summary",
                 "style": {"fontSize": "16px", "fontWeight": "600", "marginTop": "28px", "marginBottom": "12px", "color": BRAND_SLATE}},
                {"name": "Table", "type": "DataTable", "children": "", "columns": table_cols, "data": table_data}
            ]
        },
        "inputVariables": []
    }

    try:
        html1 = wire_layout(layout1, {})
        viz_list.append(SkillVisualization(title="PVM Analysis", layout=html1))
    except Exception as e:
        logger.error(f"Layout error: {e}")

    # Tab 2+: Dimensional breakouts
    for dim in breakout_dimensions[:3]:
        if dim in breakout_data:
            bar_chart = build_bar_chart(breakout_data[dim], dim, current_period, prior_period)

            # Build breakout table
            dim_df = breakout_data[dim]
            dim_cols = [
                {"name": format_display_name(dim)},
                {"name": current_period},
                {"name": prior_period},
                {"name": "Variance"},
                {"name": "Var %"}
            ]
            dim_data = []
            for _, row in dim_df.iterrows():
                dim_data.append([
                    str(row[dim]),
                    format_number(row['actual']),
                    format_number(row['prior']),
                    format_number(row['variance']),
                    f"{row['variance_pct']:+.1f}%"
                ])

            layout_dim = {
                "layoutJson": {
                    "type": "Document",
                    "style": {"padding": "20px", "fontFamily": "system-ui, -apple-system, sans-serif", "backgroundColor": "#ffffff"},
                    "children": [
                        {"name": "Header", "type": "FlexContainer", "children": "", "direction": "column",
                         "style": {"backgroundColor": BRAND_BLUE, "padding": "20px 24px", "borderRadius": "8px", "marginBottom": "24px"}},
                        {"name": "Title", "type": "Header", "children": "", "text": f"{format_display_name(dim)} Breakout",
                         "parentId": "Header", "style": {"fontSize": "22px", "fontWeight": "600", "color": "#ffffff", "margin": "0"}},
                        {"name": "Subtitle", "type": "Paragraph", "children": "",
                         "text": f"Top 10 Contributors to Variance",
                         "parentId": "Header", "style": {"fontSize": "14px", "color": "#bfdbfe", "marginTop": "4px"}},
                        {"name": "Chart", "type": "HighchartsChart", "children": "", "minHeight": "400px", "options": bar_chart},
                        {"name": "TableHeader", "type": "Paragraph", "children": "", "text": "Details",
                         "style": {"fontSize": "16px", "fontWeight": "600", "marginTop": "28px", "marginBottom": "12px", "color": BRAND_SLATE}},
                        {"name": "Table", "type": "DataTable", "children": "", "columns": dim_cols, "data": dim_data}
                    ]
                },
                "inputVariables": []
            }

            try:
                html_dim = wire_layout(layout_dim, {})
                viz_list.append(SkillVisualization(title=format_display_name(dim), layout=html_dim))
            except Exception as e:
                logger.error(f"Layout error for {dim}: {e}")

    # Parameter display
    param_info = [
        ParameterDisplayDescription(key="metric", value=f"Metric: {format_display_name(metric)}"),
        ParameterDisplayDescription(key="period", value=f"Period: {current_period} vs {prior_period}"),
        ParameterDisplayDescription(key="breakouts", value=f"Breakouts: {', '.join([format_display_name(d) for d in breakout_dimensions[:3]])}")
    ]

    max_response = jinja2.Template(max_prompt).render(facts=facts)

    return SkillOutput(
        final_prompt=max_response,
        narrative=insights,
        visualizations=viz_list,
        parameter_display_descriptions=param_info
    )


if __name__ == '__main__':
    from skill_framework import preview_skill

    skill_input: SkillInput = reckitt_pvm_drivers.create_input(arguments={
        'metric': 'net_revenue',
        'current_period': "Q2'26",
        'prior_period': "Q2'25",
        'breakout_dimensions': ['brand', 'sub_category', 'channel'],
        'other_filters': []
    })
    out = reckitt_pvm_drivers(skill_input)
    preview_skill(reckitt_pvm_drivers, out)
