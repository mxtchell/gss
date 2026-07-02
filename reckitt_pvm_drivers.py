"""
Reckitt Price-Volume-Mix Driver Analysis Skill

Uses net_revenue with units as volume to derive price for PVM decomposition.
Reckitt brands only (they have finance metrics).
"""

from __future__ import annotations
from skill_framework import skill, SkillParameter, SkillInput, SkillOutput, SkillVisualization
from skill_framework.layouts import wire_layout
import pandas as pd
import numpy as np
import json
import logging
import os

logger = logging.getLogger(__name__)

# Database Configuration - update these for your environment
DATABASE_ID = os.getenv('DATABASE_ID', '')
DATA_FILE = 'reckitt_surface_care_poc.csv'

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
4. Areas requiring immediate attention

Format the insights in clear markdown with bullet points.
"""

# Waterfall chart layout
WATERFALL_CHART_LAYOUT = """
{
    "layoutJson": {
        "type": "Document",
        "rows": 90,
        "columns": 160,
        "rowHeight": "1.11%",
        "colWidth": "0.625%",
        "gap": "0px",
        "style": {
            "backgroundColor": "#ffffff",
            "width": "100%",
            "height": "max-content",
            "padding": "15px",
            "gap": "20px"
        },
        "children": [
            {
                "name": "CardContainer0",
                "type": "CardContainer",
                "children": "",
                "minHeight": "80px",
                "rows": 2,
                "columns": 1,
                "style": {
                    "border-radius": "11.911px",
                    "background": "#2563EB",
                    "padding": "10px",
                    "fontFamily": "Arial"
                },
                "hidden": false
            },
            {
                "name": "Header0",
                "type": "Header",
                "children": "",
                "text": "Variance Analysis",
                "style": {
                    "fontSize": "20px",
                    "fontWeight": "700",
                    "color": "#ffffff",
                    "textAlign": "left",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "Paragraph0",
                "type": "Paragraph",
                "children": "",
                "text": "Price-Volume-Mix Decomposition",
                "style": {
                    "fontSize": "15px",
                    "fontWeight": "normal",
                    "textAlign": "center",
                    "verticalAlign": "start",
                    "color": "#fafafa",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "minHeight": "500px",
                "chartOptions": {},
                "options": {
                    "chart": {
                        "type": "waterfall",
                        "height": 500
                    },
                    "title": {
                        "text": ""
                    },
                    "xAxis": {
                        "categories": []
                    },
                    "yAxis": {
                        "title": {
                            "text": ""
                        }
                    },
                    "series": [],
                    "credits": {
                        "enabled": false
                    },
                    "legend": {
                        "enabled": false
                    },
                    "tooltip": {
                        "shared": true,
                        "backgroundColor": "rgba(255, 255, 255, 1)",
                        "useHTML": false
                    }
                },
                "hidden": false
            },
            {
                "name": "DataTable0",
                "type": "DataTable",
                "children": "",
                "columns": [],
                "data": [],
                "caption": "",
                "styles": {
                    "td": {
                        "vertical-align": "middle"
                    }
                }
            }
        ]
    },
    "inputVariables": [
        {
            "name": "sub_headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "Paragraph0", "fieldName": "text"}]
        },
        {
            "name": "headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "Header0", "fieldName": "text"}]
        },
        {
            "name": "chart_categories",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "HighchartsChart0", "fieldName": "options.xAxis.categories"}]
        },
        {
            "name": "chart_y_axis",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "HighchartsChart0", "fieldName": "options.yAxis"}]
        },
        {
            "name": "chart_data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "HighchartsChart0", "fieldName": "options.series"}]
        },
        {
            "name": "data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "DataTable0", "fieldName": "data"}]
        },
        {
            "name": "col_defs",
            "isRequired": false,
            "defaultValue": null,
            "targets": [{"elementName": "DataTable0", "fieldName": "columns"}]
        }
    ]
}
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

    # Use first breakout dimension for mix calculation
    mix_dimension = breakout_dimensions[0] if breakout_dimensions else 'brand'

    # Aggregate by mix dimension
    actual_by_dim = current_df.groupby(mix_dimension).agg({
        'revenue': 'sum',
        'units': 'sum'
    }).reset_index()
    actual_by_dim['price'] = actual_by_dim['revenue'] / actual_by_dim['units'].replace(0, np.nan)
    actual_by_dim['price'] = actual_by_dim['price'].fillna(0)

    prior_by_dim = prior_df.groupby(mix_dimension).agg({
        'revenue': 'sum',
        'units': 'sum'
    }).reset_index()
    prior_by_dim['price'] = prior_by_dim['revenue'] / prior_by_dim['units'].replace(0, np.nan)
    prior_by_dim['price'] = prior_by_dim['price'].fillna(0)

    # Merge
    merged = pd.merge(
        actual_by_dim,
        prior_by_dim,
        on=mix_dimension,
        how='outer',
        suffixes=('_actual', '_prior')
    ).fillna(0)

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

    # Top contributors by dimension
    for dim in breakout_dimensions[:2]:
        if dim in current_df.columns:
            actual_agg = current_df.groupby(dim)['revenue'].sum().reset_index()
            actual_agg.columns = [dim, 'actual']
            prior_agg = prior_df.groupby(dim)['revenue'].sum().reset_index()
            prior_agg.columns = [dim, 'prior']

            dim_merged = pd.merge(actual_agg, prior_agg, on=dim, how='outer').fillna(0)
            dim_merged['variance'] = dim_merged['actual'] - dim_merged['prior']
            dim_merged['variance_pct'] = (dim_merged['variance'] / dim_merged['prior'].replace(0, np.nan) * 100).fillna(0)
            dim_merged = dim_merged.sort_values('variance', key=abs, ascending=False)

            for _, row in dim_merged.head(3).iterrows():
                direction = "increased" if row['variance'] > 0 else "decreased"
                facts.append(f"{format_display_name(dim)} '{row[dim]}': {format_number(row['variance'])} ({row['variance_pct']:+.1f}%) - {direction}")

    return pvm_results, facts, current_df, prior_df


def create_waterfall_chart(pvm_results, current_period, prior_period, metric):
    """Create waterfall chart data"""

    categories = [prior_period, "Volume", "Price", "Mix", current_period]

    def get_color(value):
        return '#4ade80' if value >= 0 else '#ef4444'

    volume_val = pvm_results['volume_impact']
    price_val = pvm_results['price_impact']
    mix_val = pvm_results['mix_impact']

    scale = 1_000_000 if abs(pvm_results['starting_value']) >= 1_000_000 else 1_000

    data_series = [{
        'name': format_display_name(metric),
        'data': [
            {
                'name': prior_period,
                'y': pvm_results['starting_value'] / scale,
                'color': '#3b82f6',
                'dataLabels': {'enabled': True, 'format': format_number(pvm_results['starting_value'])}
            },
            {
                'name': 'Volume',
                'y': volume_val / scale,
                'color': get_color(volume_val),
                'dataLabels': {'enabled': True, 'format': format_number(volume_val)}
            },
            {
                'name': 'Price',
                'y': price_val / scale,
                'color': get_color(price_val),
                'dataLabels': {'enabled': True, 'format': format_number(price_val)}
            },
            {
                'name': 'Mix',
                'y': mix_val / scale,
                'color': get_color(mix_val),
                'dataLabels': {'enabled': True, 'format': format_number(mix_val)}
            },
            {
                'name': current_period,
                'isSum': True,
                'y': pvm_results['ending_value'] / scale,
                'color': '#3b82f6',
                'dataLabels': {'enabled': True, 'format': format_number(pvm_results['ending_value'])}
            }
        ],
        'dataLabels': {
            'enabled': True,
            'style': {'fontWeight': 'bold', 'color': '#000000', 'textOutline': 'none'}
        }
    }]

    scale_label = 'M' if scale == 1_000_000 else 'K'

    return {
        'chart_categories': categories,
        'chart_data': data_series,
        'chart_y_axis': {
            'title': {'text': f'{format_display_name(metric)} (${scale_label})'},
            'labels': {'format': '${value:,.0f}' + scale_label}
        }
    }


def create_summary_table(pvm_results, current_period, prior_period):
    """Create summary table"""

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
            "Volume Impact",
            "", "",
            format_number(pvm_results['volume_impact']),
            f"{pvm_results['volume_impact'] / abs(pvm_results['total_variance']) * 100:.1f}%" if pvm_results['total_variance'] != 0 else "N/A"
        ],
        [
            "Price Impact",
            "", "",
            format_number(pvm_results['price_impact']),
            f"{pvm_results['price_impact'] / abs(pvm_results['total_variance']) * 100:.1f}%" if pvm_results['total_variance'] != 0 else "N/A"
        ],
        [
            "Mix Impact",
            "", "",
            format_number(pvm_results['mix_impact']),
            f"{pvm_results['mix_impact'] / abs(pvm_results['total_variance']) * 100:.1f}%" if pvm_results['total_variance'] != 0 else "N/A"
        ]
    ]

    columns = [
        {'name': 'Metric'},
        {'name': current_period},
        {'name': prior_period},
        {'name': 'Change'},
        {'name': 'Change %'}
    ]

    return {'data': data, 'col_defs': columns}


@skill(
    name="Reckitt PVM Drivers",
    llm_name="reckitt_pvm_drivers",
    description=(
        "Price-Volume-Mix variance analysis for Reckitt Surface Care brands. "
        "Decomposes net revenue changes into Volume, Price, and Mix impacts using units as volume metric. "
        "Compares current period vs prior period (e.g., Q2'26 vs Q2'25 for YoY analysis)."
    ),
    capabilities=(
        "1) Price-Volume-Mix decomposition: Volume Impact (units change × prior price), "
        "Price Impact (price change × current units), Mix Impact (portfolio shift). "
        "2) Waterfall chart visualization showing variance bridge from prior to current period. "
        "3) Dimensional breakout by brand, sub_category, segment, channel, state. "
        "4) Top contributor identification - which brands/channels drove growth or decline. "
        "5) YoY and sequential quarter comparisons."
    ),
    limitations=(
        "Only analyzes Reckitt Benckiser brands (LYSOL, LYSOL POWER, LYSOL LAUNDRY SANITIZER). "
        "Finance metrics (net_revenue, gross_sales, gross_margin) only available for Reckitt brands. "
        "Data covers Q2'25 through Q2'26 (5 quarters)."
    ),
    example_questions=(
        "What drove the net revenue change in Q2'26 vs Q2'25? | "
        "Show me price-volume-mix analysis for Q4'25 vs Q3'25. | "
        "Which brands contributed most to revenue growth? | "
        "Break down the variance by channel. | "
        "What is the YoY revenue change for Lysol?"
    ),
    parameter_guidance=(
        "METRIC: net_revenue (default), gross_sales, or gross_margin. "
        "CURRENT_PERIOD: Quarter to analyze (e.g., Q2'26, Q1'26, Q4'25, Q3'25, Q2'25). "
        "PRIOR_PERIOD: Comparison quarter (e.g., Q2'25 for YoY, Q1'26 for sequential). "
        "BREAKOUT_DIMENSIONS: brand, sub_category, segment, channel, state_name. "
        "FILTERS: Narrow by brand, channel, sub_category, etc."
    ),
    parameters=[
        SkillParameter(
            name="metric",
            description="Revenue metric to analyze: net_revenue (default), gross_sales, or gross_margin",
            default_value="net_revenue"
        ),
        SkillParameter(
            name="current_period",
            description="Current period quarter (e.g., Q2'26, Q1'26, Q4'25)",
            default_value="Q2'26"
        ),
        SkillParameter(
            name="prior_period",
            description="Prior period for comparison (e.g., Q2'25 for YoY, Q1'26 for sequential)",
            default_value="Q2'25"
        ),
        SkillParameter(
            name="breakout_dimensions",
            constrained_to="dimensions",
            is_multi=True,
            description="Dimensions for breakout analysis: brand, sub_category, segment, channel, state_name"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            is_multi=True,
            description="Filters to narrow analysis (e.g., brand='LYSOL', channel='Grocery')"
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for executive summary",
            default_value=DEFAULT_MAX_PROMPT
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt for detailed insights",
            default_value=DEFAULT_INSIGHT_PROMPT
        )
    ]
)
def reckitt_pvm_drivers(parameters: SkillInput) -> SkillOutput:
    """Reckitt Price-Volume-Mix Driver Analysis"""

    logger.info(f"Skill received parameters: {parameters.arguments}")

    # Extract parameters
    metric = getattr(parameters.arguments, 'metric', 'net_revenue') or 'net_revenue'
    current_period = getattr(parameters.arguments, 'current_period', "Q2'26") or "Q2'26"
    prior_period = getattr(parameters.arguments, 'prior_period', "Q2'25") or "Q2'25"
    breakout_dimensions = getattr(parameters.arguments, 'breakout_dimensions', None) or ['brand', 'sub_category', 'segment', 'channel', 'state_name']
    other_filters = getattr(parameters.arguments, 'other_filters', []) or []
    max_prompt = getattr(parameters.arguments, 'max_prompt', DEFAULT_MAX_PROMPT)
    insight_prompt = getattr(parameters.arguments, 'insight_prompt', DEFAULT_INSIGHT_PROMPT)

    # Get client
    try:
        from answerrocket_client import AnswerRocketClient
        client = AnswerRocketClient()
    except Exception as e:
        logger.error(f"Failed to initialize AnswerRocketClient: {e}")
        return SkillOutput(
            final_prompt=f"Failed to initialize client: {str(e)}",
            narrative=f"**Error**: Could not connect to AnswerRocket. {str(e)}",
            visualizations=[],
            warnings=[str(e)]
        )

    # Run analysis
    try:
        pvm_results, facts, current_df, prior_df = run_pvm_analysis(
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
            narrative=f"**Error**: {str(e)}\n\nPlease verify the period values are valid (Q2'25, Q3'25, Q4'25, Q1'26, Q2'26).",
            visualizations=[],
            warnings=[str(e)]
        )

    # Generate insights
    import jinja2
    try:
        from ar_analytics import ArUtils
        ar_utils = ArUtils()
        insight_template = jinja2.Template(insight_prompt).render(facts=facts)
        insights = ar_utils.get_llm_response(insight_template)
    except:
        insights = "**Price-Volume-Mix Analysis Complete**\n\n" + "\n".join([f"- {f}" for f in facts])

    # Create visualization
    waterfall_data = create_waterfall_chart(pvm_results, current_period, prior_period, metric)
    summary_table = create_summary_table(pvm_results, current_period, prior_period)

    metric_display = format_display_name(metric)
    layout_vars = {
        "headline": f"{metric_display} Variance Analysis",
        "sub_headline": f"{current_period} vs {prior_period} | Reckitt Surface Care",
        **waterfall_data,
        **summary_table
    }

    rendered = wire_layout(json.loads(WATERFALL_CHART_LAYOUT), layout_vars)

    viz_list = [
        SkillVisualization(title=f"{metric_display} PVM Analysis", layout=rendered)
    ]

    # Max prompt for response
    max_response = jinja2.Template(max_prompt).render(facts=facts)

    return SkillOutput(
        final_prompt=max_response,
        narrative=insights,
        visualizations=viz_list
    )


if __name__ == '__main__':
    from skill_framework import preview_skill

    skill_input: SkillInput = reckitt_pvm_drivers.create_input(arguments={
        'metric': 'net_revenue',
        'current_period': "Q2'26",
        'prior_period': "Q2'25",
        'breakout_dimensions': ['brand', 'sub_category'],
        'other_filters': []
    })
    out = reckitt_pvm_drivers(skill_input)
    preview_skill(reckitt_pvm_drivers, out)
