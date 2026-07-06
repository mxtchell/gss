from __future__ import annotations
from types import SimpleNamespace
import pandas as pd
import numpy as np
from skill_framework import (
    SkillInput,
    SkillVisualization,
    skill,
    SkillParameter,
    SkillOutput,
    ParameterDisplayDescription
)
from skill_framework.skills import ExportData
from skill_framework.layouts import wire_layout
from ar_analytics import ArUtils
import jinja2
import json
import logging
import os

logger = logging.getLogger(__name__)

# Database Configuration
DATABASE_ID = os.getenv('DATABASE_ID', '')
DATA_FILE = 'reckitt_surface_care_poc.csv'


# Default prompts
DEFAULT_MAX_PROMPT = """
Based on the following variance analysis facts:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide a concise executive summary (2-3 sentences) highlighting the most significant variance drivers.
"""

DEFAULT_INSIGHT_PROMPT = """
Analyze the following variance analysis data:
{% for fact_list in facts %}
{% for fact in fact_list %}
- {{ fact }}
{% endfor %}
{% endfor %}

Provide detailed insights covering:
1. Key variance drivers (Price, Volume, Mix)
2. Top contributing dimensions
3. Actionable recommendations for stakeholders

Format the insights in clear markdown with bullet points.
"""


# Layout template for waterfall chart visualization - EXACT COPY FROM FPA
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
                    "border": "null",
                    "textDecoration": "null",
                    "writingMode": "horizontal-tb",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "minHeight": "600px",
                "chartOptions": {},
                "options": {
                    "chart": {
                        "type": "waterfall",
                        "height": 600
                    },
                    "title": {
                        "text": "",
                        "style": {
                            "fontSize": "18px",
                            "fontWeight": "bold"
                        }
                    },
                    "xAxis": {
                        "categories": [],
                        "title": {
                            "text": ""
                        }
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
                        "pointFormat": "<b>{point.name}</b>: {point.formatted}"
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
            "targets": [
                {
                    "elementName": "Paragraph0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Header0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "chart_categories",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.xAxis.categories"
                }
            ]
        },
        {
            "name": "chart_y_axis",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.yAxis"
                }
            ]
        },
        {
            "name": "chart_data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.series"
                }
            ]
        },
        {
            "name": "chart_title",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.title.text"
                }
            ]
        },
        {
            "name": "data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "data"
                }
            ]
        },
        {
            "name": "col_defs",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "columns"
                }
            ]
        }
    ]
}
"""

# Horizontal bar chart layout for dimensional breakouts - EXACT COPY FROM FPA
HORIZONTAL_BAR_LAYOUT = """
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
                "text": "Dimensional Breakout",
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
                "text": "Variance by Dimension",
                "style": {
                    "fontSize": "15px",
                    "fontWeight": "normal",
                    "textAlign": "center",
                    "verticalAlign": "start",
                    "color": "#fafafa",
                    "border": "null",
                    "textDecoration": "null",
                    "writingMode": "horizontal-tb",
                    "alignItems": "center"
                },
                "parentId": "CardContainer0",
                "hidden": false
            },
            {
                "name": "HighchartsChart0",
                "type": "HighchartsChart",
                "minHeight": "400px",
                "chartOptions": {},
                "options": {
                    "chart": {
                        "type": "bar"
                    },
                    "title": {
                        "text": "",
                        "style": {
                            "fontSize": "18px",
                            "fontWeight": "bold"
                        }
                    },
                    "xAxis": {
                        "categories": [],
                        "title": {
                            "text": ""
                        }
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
                        "enabled": true,
                        "align": "center",
                        "verticalAlign": "bottom",
                        "layout": "horizontal"
                    },
                    "plotOptions": {
                        "bar": {
                            "dataLabels": {
                                "enabled": false
                            }
                        }
                    },
                    "tooltip": {
                        "pointFormat": "<b>{series.name}</b>: {point.y:,.0f}"
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
            "targets": [
                {
                    "elementName": "Paragraph0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "headline",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "Header0",
                    "fieldName": "text"
                }
            ]
        },
        {
            "name": "chart_categories",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.xAxis.categories"
                }
            ]
        },
        {
            "name": "chart_y_axis",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.yAxis"
                }
            ]
        },
        {
            "name": "chart_data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.series"
                }
            ]
        },
        {
            "name": "chart_title",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "HighchartsChart0",
                    "fieldName": "options.title.text"
                }
            ]
        },
        {
            "name": "data",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "data"
                }
            ]
        },
        {
            "name": "col_defs",
            "isRequired": false,
            "defaultValue": null,
            "targets": [
                {
                    "elementName": "DataTable0",
                    "fieldName": "columns"
                }
            ]
        }
    ]
}
"""


def format_number(value, is_currency=True, decimals=1):
    """Format numbers with M/K/B abbreviations"""
    if pd.isna(value) or not isinstance(value, (int, float)):
        return str(value)

    abs_value = abs(value)

    if abs_value >= 1_000_000_000:
        formatted = f"{value / 1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        formatted = f"{value / 1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        formatted = f"{value / 1_000:.{decimals}f}K"
    else:
        formatted = f"{value:.{decimals}f}"

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
        'brand': 'Brand',
        'segment': 'Segment',
        'channel': 'Channel',
    }

    if name.lower() in special_cases:
        return special_cases[name.lower()]

    return name.replace('_', ' ').title()


class ReckittPVMAnalysis:
    """Reckitt PVM Variance Analysis using net_revenue and units"""

    def __init__(self, client, metric, current_period, prior_period, breakout_dimensions=None,
                 top_n=10, other_filters=None):
        self.client = client
        self.metric = metric
        self.current_period = current_period
        self.prior_period = prior_period
        self.breakout_dimensions = breakout_dimensions or []
        self.top_n = top_n
        self.other_filters = other_filters or []

        self.current_df = None
        self.prior_df = None
        self.pvm_results = None
        self.breakout_results = {}
        self.facts = []

    def build_filter_clause(self):
        """Build SQL WHERE clause from filters"""
        clauses = []

        if self.other_filters:
            for filter_dict in self.other_filters:
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

    def parse_period_to_date_range(self, period_str):
        """Convert period string to date range for month_new column query.

        Dataset has month_new with values like: 2025-04-01, 2025-05-01, etc.
        """
        if not period_str:
            raise ValueError("Period is required but was not provided")

        period_str = period_str.strip().upper()

        # Quarter mapping to month_new date ranges
        quarter_map = {
            "Q2'25": ("2025-04-01", "2025-06-01"),
            "Q3'25": ("2025-07-01", "2025-09-01"),
            "Q4'25": ("2025-10-01", "2025-12-01"),
            "Q1'26": ("2026-01-01", "2026-03-01"),
            "Q2'26": ("2026-04-01", "2026-06-01"),
            # Also support Q2 2025 format
            "Q2 2025": ("2025-04-01", "2025-06-01"),
            "Q3 2025": ("2025-07-01", "2025-09-01"),
            "Q4 2025": ("2025-10-01", "2025-12-01"),
            "Q1 2026": ("2026-01-01", "2026-03-01"),
            "Q2 2026": ("2026-04-01", "2026-06-01"),
        }

        if period_str in quarter_map:
            return quarter_map[period_str]

        # Try to parse as Q2'25 format
        if "'" in period_str:
            return quarter_map.get(period_str, (period_str, period_str))

        raise ValueError(f"Unknown period format: {period_str}. Use Q2'25 or Q2 2025 format.")

    def query_data(self):
        """Query current and prior period data from database"""
        logger.info(f"Querying data for metric: {self.metric}, periods: {self.current_period} vs {self.prior_period}")

        filter_clause = self.build_filter_clause()

        # Parse periods to date ranges for month_new column
        current_start, current_end = self.parse_period_to_date_range(self.current_period)
        prior_start, prior_end = self.parse_period_to_date_range(self.prior_period)

        logger.info(f"Current period: month_new BETWEEN '{current_start}' AND '{current_end}'")
        logger.info(f"Prior period: month_new BETWEEN '{prior_start}' AND '{prior_end}'")

        # Query current period (Reckitt brands only)
        current_query = f"""
        SELECT brand, sub_category, segment, channel, state_name,
               SUM({self.metric}) as revenue,
               SUM(units) as units
        FROM read_csv('{DATA_FILE}')
        WHERE manufacturer = 'RECKITT BENCKISER'
        AND month_new BETWEEN '{current_start}' AND '{current_end}'
        {filter_clause}
        GROUP BY brand, sub_category, segment, channel, state_name
        """

        logger.info(f"Current period query: {current_query}")
        result = self.client.data.execute_sql_query(
            database_id=DATABASE_ID,
            sql_query=current_query,
            row_limit=50000
        )
        if not result.success:
            raise ValueError(f"Current period query failed: {result.error}")
        self.current_df = result.df

        # Query prior period
        prior_query = f"""
        SELECT brand, sub_category, segment, channel, state_name,
               SUM({self.metric}) as revenue,
               SUM(units) as units
        FROM read_csv('{DATA_FILE}')
        WHERE manufacturer = 'RECKITT BENCKISER'
        AND month_new BETWEEN '{prior_start}' AND '{prior_end}'
        {filter_clause}
        GROUP BY brand, sub_category, segment, channel, state_name
        """

        logger.info(f"Prior period query: {prior_query}")
        result = self.client.data.execute_sql_query(
            database_id=DATABASE_ID,
            sql_query=prior_query,
            row_limit=50000
        )
        if not result.success:
            raise ValueError(f"Prior period query failed: {result.error}")
        self.prior_df = result.df

        logger.info(f"Current shape: {self.current_df.shape if self.current_df is not None else 'None'}")
        logger.info(f"Prior shape: {self.prior_df.shape if self.prior_df is not None else 'None'}")

        if self.current_df is None or self.current_df.empty:
            raise ValueError(f"No data found for current period: {self.current_period} (month_new BETWEEN '{current_start}' AND '{current_end}')")
        if self.prior_df is None or self.prior_df.empty:
            raise ValueError(f"No data found for prior period: {self.prior_period} (month_new BETWEEN '{prior_start}' AND '{prior_end}')")

    def calculate_price_volume_mix(self):
        """Calculate Price-Volume-Mix decomposition"""
        logger.info("Calculating Price-Volume-Mix decomposition")

        if self.current_df is None or self.prior_df is None:
            logger.error("Data not loaded. Call query_data() first.")
            return None

        # Use brand for mix calculation
        mix_dimension = 'brand'

        # Calculate totals
        actual_revenue = self.current_df['revenue'].sum()
        prior_revenue = self.prior_df['revenue'].sum()
        total_variance = actual_revenue - prior_revenue

        actual_units = self.current_df['units'].sum()
        prior_units = self.prior_df['units'].sum()

        # Category-level PVM calculation
        actual_by_cat = self.current_df.groupby(mix_dimension).agg({
            'revenue': 'sum',
            'units': 'sum'
        }).reset_index()
        actual_by_cat['price'] = actual_by_cat['revenue'] / actual_by_cat['units'].replace(0, np.nan)
        actual_by_cat['price'] = actual_by_cat['price'].fillna(0)

        prior_by_cat = self.prior_df.groupby(mix_dimension).agg({
            'revenue': 'sum',
            'units': 'sum'
        }).reset_index()
        prior_by_cat['price'] = prior_by_cat['revenue'] / prior_by_cat['units'].replace(0, np.nan)
        prior_by_cat['price'] = prior_by_cat['price'].fillna(0)

        merged = pd.merge(
            actual_by_cat,
            prior_by_cat,
            on=mix_dimension,
            how='outer',
            suffixes=('_actual', '_prior')
        ).fillna(0)

        total_actual_units = actual_by_cat['units'].sum()
        total_prior_units = prior_by_cat['units'].sum()

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

        logger.info(f"PVM: Volume={volume_impact:,.0f}, Price={price_impact:,.0f}, Mix={mix_impact:,.0f}")

        self.pvm_results = {
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

        # Create facts
        self.facts.append({
            'fact': f"Total variance: {format_number(total_variance)} ({total_variance/prior_revenue*100:.1f}%)" if prior_revenue != 0 else f"Total variance: {format_number(total_variance)}",
            'category': 'overall'
        })
        self.facts.append({
            'fact': f"Volume impact: {format_number(volume_impact)} ({volume_impact/abs(total_variance)*100 if total_variance != 0 else 0:.1f}% of variance)",
            'category': 'pvm'
        })
        self.facts.append({
            'fact': f"Price impact: {format_number(price_impact)} ({price_impact/abs(total_variance)*100 if total_variance != 0 else 0:.1f}% of variance)",
            'category': 'pvm'
        })
        self.facts.append({
            'fact': f"Mix impact: {format_number(mix_impact)} ({mix_impact/abs(total_variance)*100 if total_variance != 0 else 0:.1f}% of variance)",
            'category': 'pvm'
        })

        return self.pvm_results

    def calculate_dimensional_breakout(self, dimension):
        """Calculate variance attribution by dimension"""
        logger.info(f"Calculating breakout for dimension: {dimension}")

        if self.current_df is None or self.prior_df is None:
            return None

        if dimension not in self.current_df.columns:
            logger.warning(f"Dimension {dimension} not in data")
            return None

        # Aggregate by dimension
        actual_agg = self.current_df.groupby(dimension)['revenue'].sum().reset_index()
        actual_agg.columns = [dimension, 'actual']

        prior_agg = self.prior_df.groupby(dimension)['revenue'].sum().reset_index()
        prior_agg.columns = [dimension, 'prior']

        merged = pd.merge(actual_agg, prior_agg, on=dimension, how='outer').fillna(0)

        # Calculate variance
        merged['variance'] = merged['actual'] - merged['prior']
        merged['variance_pct'] = (merged['variance'] / merged['prior'].replace(0, np.nan) * 100).fillna(0)

        # Rank by absolute variance
        merged['abs_variance'] = merged['variance'].abs()
        merged = merged.sort_values('abs_variance', ascending=False)

        # Take top N
        top_n_df = merged.head(self.top_n).copy()

        self.breakout_results[dimension] = top_n_df

        # Add facts for top contributors
        for idx, row in top_n_df.head(3).iterrows():
            self.facts.append({
                'fact': f"{dimension} '{row[dimension]}': {format_number(row['variance'])} variance ({row['variance_pct']:.1f}%)",
                'category': f'breakout_{dimension}'
            })

        return top_n_df

    def create_waterfall_chart_data(self):
        """Create Highcharts waterfall chart configuration - EXACT SAME AS FPA"""
        if not self.pvm_results:
            return None

        categories = [
            self.prior_period,
            "Volume",
            "Price",
            "Mix",
            self.current_period
        ]

        metric_display = format_display_name(self.metric)

        def format_millions(value):
            return f"${value / 1_000_000:.2f}M"

        def format_thousands(value):
            return f"${value / 1_000:.1f}K"

        def get_color(value):
            return '#4ade80' if value >= 0 else '#ef4444'

        volume_val = int(self.pvm_results['volume_impact'])
        price_val = int(self.pvm_results['price_impact'])
        mix_val = int(self.pvm_results['mix_impact'])

        # Scale for display
        starting_m = self.pvm_results['starting_value'] / 1_000_000
        volume_m = volume_val / 1_000_000
        price_m = price_val / 1_000_000
        mix_m = mix_val / 1_000_000
        ending_m = self.pvm_results['ending_value'] / 1_000_000

        data_series = [{
            'name': metric_display,
            'data': [
                {
                    'name': self.prior_period,
                    'y': starting_m,
                    'color': '#3b82f6',
                    'dataLabels': {
                        'enabled': True,
                        'format': format_millions(self.pvm_results['starting_value'])
                    }
                },
                {
                    'name': 'Volume',
                    'y': volume_m,
                    'color': get_color(volume_val),
                    'dataLabels': {
                        'enabled': True,
                        'format': format_millions(volume_val) if abs(volume_val) >= 1_000_000 else format_thousands(volume_val)
                    }
                },
                {
                    'name': 'Price',
                    'y': price_m,
                    'color': get_color(price_val),
                    'dataLabels': {
                        'enabled': True,
                        'format': format_millions(price_val) if abs(price_val) >= 1_000_000 else format_thousands(price_val)
                    }
                },
                {
                    'name': 'Mix',
                    'y': mix_m,
                    'color': get_color(mix_val),
                    'dataLabels': {
                        'enabled': True,
                        'format': format_thousands(mix_val)
                    }
                },
                {
                    'name': self.current_period,
                    'isSum': True,
                    'y': ending_m,
                    'color': '#3b82f6',
                    'dataLabels': {
                        'enabled': True,
                        'format': format_millions(self.pvm_results['ending_value'])
                    }
                }
            ],
            'dataLabels': {
                'enabled': True,
                'style': {
                    'fontWeight': 'bold',
                    'color': '#000000',
                    'textOutline': 'none'
                }
            },
            'tooltip': {
                'pointFormat': '<b>{point.name}</b>: {point.y:.2f}M'
            }
        }]

        # Y-axis range
        min_val = min(starting_m, ending_m, starting_m + volume_m, starting_m + volume_m + price_m, starting_m + volume_m + price_m + mix_m)
        max_val = max(starting_m, ending_m, starting_m + volume_m, starting_m + volume_m + price_m, starting_m + volume_m + price_m + mix_m)
        padding = (max_val - min_val) * 0.1

        return {
            'chart_categories': categories,
            'chart_data': data_series,
            'chart_y_axis': {
                'title': {'text': metric_display},
                'labels': {'format': '${value:,.0f}M'},
                'min': min_val - padding,
                'max': max_val + padding
            },
            'chart_title': ''
        }

    def create_horizontal_bar_chart_data(self, dimension):
        """Create Highcharts horizontal bar chart for dimension breakout"""
        if dimension not in self.breakout_results:
            return None

        df = self.breakout_results[dimension]

        categories = df[dimension].tolist()
        actual_data = [x / 1_000_000 for x in df['actual'].tolist()]
        prior_data = [x / 1_000_000 for x in df['prior'].tolist()]

        return {
            'chart_categories': categories,
            'chart_data': [
                {
                    'name': self.current_period,
                    'data': actual_data,
                    'color': '#5DADE2'
                },
                {
                    'name': self.prior_period,
                    'data': prior_data,
                    'color': '#F8C471'
                }
            ],
            'chart_y_axis': {
                'title': {'text': format_display_name(self.metric)},
                'labels': {'format': '${value:,.0f}M'}
            },
            'chart_title': f'{format_display_name(dimension)} Variance Analysis'
        }

    def get_summary_table(self):
        """Create summary table with PVM breakdown"""
        if not self.pvm_results:
            return None

        data = [
            [
                format_display_name(self.metric),
                format_number(self.pvm_results['ending_value']),
                format_number(self.pvm_results['starting_value']),
                format_number(self.pvm_results['total_variance']),
                f"{self.pvm_results['total_variance'] / self.pvm_results['starting_value'] * 100:+.1f}%" if self.pvm_results['starting_value'] != 0 else "N/A"
            ],
            [
                "  Units",
                f"{self.pvm_results['actual_units']:,.0f}",
                f"{self.pvm_results['prior_units']:,.0f}",
                f"{self.pvm_results['actual_units'] - self.pvm_results['prior_units']:+,.0f}",
                f"{(self.pvm_results['actual_units'] - self.pvm_results['prior_units']) / self.pvm_results['prior_units'] * 100:+.1f}%" if self.pvm_results['prior_units'] != 0 else "N/A"
            ],
            [
                "  Price ($/Unit)",
                f"${self.pvm_results['actual_price']:.2f}",
                f"${self.pvm_results['prior_price']:.2f}",
                f"${self.pvm_results['actual_price'] - self.pvm_results['prior_price']:+.2f}",
                f"{(self.pvm_results['actual_price'] - self.pvm_results['prior_price']) / self.pvm_results['prior_price'] * 100:+.1f}%" if self.pvm_results['prior_price'] != 0 else "N/A"
            ],
            ["", "", "", "", ""],
            [
                "Volume Impact",
                "", "",
                format_number(self.pvm_results['volume_impact']),
                f"{self.pvm_results['volume_impact'] / abs(self.pvm_results['total_variance']) * 100:.1f}% of var" if self.pvm_results['total_variance'] != 0 else "N/A"
            ],
            [
                "Price Impact",
                "", "",
                format_number(self.pvm_results['price_impact']),
                f"{self.pvm_results['price_impact'] / abs(self.pvm_results['total_variance']) * 100:.1f}% of var" if self.pvm_results['total_variance'] != 0 else "N/A"
            ],
            [
                "Mix Impact",
                "", "",
                format_number(self.pvm_results['mix_impact']),
                f"{self.pvm_results['mix_impact'] / abs(self.pvm_results['total_variance']) * 100:.1f}% of var" if self.pvm_results['total_variance'] != 0 else "N/A"
            ]
        ]

        columns = [
            {'name': ''},
            {'name': self.current_period},
            {'name': self.prior_period},
            {'name': 'Change ($)'},
            {'name': 'Change (%)'}
        ]

        return {'data': data, 'col_defs': columns}

    def get_breakout_table(self, dimension):
        """Create variance table for dimension breakout"""
        if dimension not in self.breakout_results:
            return None

        df = self.breakout_results[dimension]

        data = []
        for _, row in df.iterrows():
            data.append([
                row[dimension],
                format_number(row['actual']),
                format_number(row['prior']),
                format_number(row['variance']),
                f"{row['variance_pct']:.1f}%"
            ])

        columns = [
            {'name': dimension},
            {'name': self.current_period},
            {'name': self.prior_period},
            {'name': 'Variance'},
            {'name': 'Variance %'}
        ]

        return {'data': data, 'col_defs': columns}

    def run_analysis(self):
        """Run complete variance analysis"""
        logger.info("Starting Reckitt PVM analysis")

        self.query_data()
        self.calculate_price_volume_mix()

        for dim in self.breakout_dimensions:
            self.calculate_dimensional_breakout(dim)

        logger.info("Analysis complete")
        return self


@skill(
    name="Reckitt PVM Drivers",
    llm_name="reckitt_pvm_drivers",
    description="Price-Volume-Mix variance analysis for Reckitt Surface Care brands. Decomposes net revenue changes into Volume, Price, and Mix impacts. Compares current period vs prior period.",
    capabilities="Price-Volume-Mix variance decomposition. Waterfall chart visualization. Dimensional breakout by brand, sub_category, segment, channel, state.",
    limitations="Only analyzes Reckitt Benckiser brands. Data covers Q2'25 through Q2'26.",
    example_questions="What drove the net revenue change in Q2'26 vs Q2'25? Show me price-volume-mix analysis. Which brands contributed most to revenue growth?",
    parameter_guidance="METRIC: net_revenue, gross_sales, gross_margin. CURRENT_PERIOD: Q2'26, Q1'26, Q4'25, Q3'25, Q2'25. PRIOR_PERIOD: Q2'25 for YoY.",
    parameters=[
        SkillParameter(name="metric", constrained_to="metrics", is_multi=False, description="Metric to analyze"),
        SkillParameter(name="current_period", description="Current quarter (e.g., Q2'26)", default_value="Q2'26"),
        SkillParameter(name="prior_period", description="Prior quarter for comparison (e.g., Q2'25)", default_value="Q2'25"),
        SkillParameter(name="breakout_dimensions", constrained_to="dimensions", is_multi=True, description="Dimensions for breakout"),
        SkillParameter(name="top_n", description="Number of top contributors", default_value=10),
        SkillParameter(name="other_filters", constrained_to="filters", is_multi=True, description="Additional filters"),
        SkillParameter(name="max_prompt", parameter_type="prompt", description="Prompt for summary", default_value=DEFAULT_MAX_PROMPT),
        SkillParameter(name="insight_prompt", parameter_type="prompt", description="Prompt for insights", default_value=DEFAULT_INSIGHT_PROMPT)
    ]
)
def reckitt_pvm_drivers(parameters: SkillInput):
    """Execute Reckitt PVM Variance Analysis"""

    logger.info(f"Skill received parameters: {parameters.arguments}")

    # Extract parameters
    metric = getattr(parameters.arguments, 'metric', 'net_revenue') or 'net_revenue'
    current_period = getattr(parameters.arguments, 'current_period', "Q2'26") or "Q2'26"
    prior_period = getattr(parameters.arguments, 'prior_period', "Q2'25") or "Q2'25"

    # HARDCODED: Always show these breakout dimensions
    breakout_dimensions = ['brand', 'sub_category', 'segment', 'channel', 'state_name']

    top_n = int(getattr(parameters.arguments, 'top_n', 10) or 10)
    other_filters = getattr(parameters.arguments, 'other_filters', [])
    max_prompt = getattr(parameters.arguments, 'max_prompt', DEFAULT_MAX_PROMPT)
    insight_prompt = getattr(parameters.arguments, 'insight_prompt', DEFAULT_INSIGHT_PROMPT)

    # Get AnswerRocketClient
    try:
        from answer_rocket import AnswerRocketClient
        client = AnswerRocketClient()
        ar_utils = ArUtils()
    except Exception as e:
        logger.error(f"Failed to initialize AnswerRocketClient: {e}")
        return SkillOutput(
            final_prompt=f"Failed to initialize client: {str(e)}",
            warnings=[str(e)]
        )

    # Run analysis
    analysis = ReckittPVMAnalysis(
        client=client,
        metric=metric,
        current_period=current_period,
        prior_period=prior_period,
        breakout_dimensions=breakout_dimensions,
        top_n=top_n,
        other_filters=other_filters
    )

    try:
        analysis.run_analysis()
    except ValueError as e:
        logger.error(f"Analysis failed: {e}")
        return SkillOutput(
            final_prompt=f"Analysis could not be completed: {str(e)}",
            narrative=f"**Error**: {str(e)}",
            visualizations=[],
            warnings=[str(e)]
        )

    # Generate insights
    facts_list = [pd.DataFrame(analysis.facts)]
    insight_template = jinja2.Template(insight_prompt).render(facts=[facts_list])
    max_response_prompt = jinja2.Template(max_prompt).render(facts=[facts_list])

    try:
        insights = ar_utils.get_llm_response(insight_template)
    except:
        insights = "Variance analysis complete. Review the waterfall chart and dimensional breakouts for detailed insights."

    # Create visualizations - EXACT SAME PATTERN AS FPA
    viz_list = []
    export_data = {}

    # Tab 1: Waterfall Chart + Summary Table
    waterfall_data = analysis.create_waterfall_chart_data()
    summary_table = analysis.get_summary_table()

    logger.info(f"Waterfall data: {waterfall_data}")
    logger.info(f"Summary table: {summary_table}")

    if waterfall_data and summary_table:
        metric_display = format_display_name(metric)
        general_vars = {
            "headline": f"{metric_display} Variance Analysis",
            "sub_headline": f"{current_period} vs {prior_period} | Price-Volume-Mix",
            "exec_summary": insights
        }

        layout_vars = {**general_vars, **waterfall_data, **summary_table}

        logger.info(f"Layout vars keys: {layout_vars.keys()}")
        logger.info(f"Chart data sample: {layout_vars.get('chart_data', 'MISSING')}")

        rendered = wire_layout(json.loads(WATERFALL_CHART_LAYOUT), layout_vars)
        viz_list.append(SkillVisualization(title=f"{metric_display} Analysis", layout=rendered))
        export_data["PVM_Summary"] = pd.DataFrame(summary_table['data'], columns=['', current_period, prior_period, 'Change ($)', 'Change (%)'])
    else:
        logger.error(f"Missing waterfall data or summary table")

    # Tab 2+: Horizontal Bar Charts for each dimension
    for dimension in breakout_dimensions:
        bar_data = analysis.create_horizontal_bar_chart_data(dimension)
        table_data = analysis.get_breakout_table(dimension)

        if bar_data and table_data:
            dimension_display = format_display_name(dimension)
            general_vars = {
                "headline": f"{dimension_display} Breakout",
                "sub_headline": f"Top {top_n} Contributors to Variance"
            }

            layout_vars = {**general_vars, **bar_data, **table_data}
            rendered = wire_layout(json.loads(HORIZONTAL_BAR_LAYOUT), layout_vars)
            viz_list.append(SkillVisualization(title=dimension_display, layout=rendered))
            export_data[f"{dimension}_Variance"] = analysis.breakout_results[dimension]

    # Parameter display
    metric_display = format_display_name(metric)
    dimensions_display = ", ".join([format_display_name(d) for d in breakout_dimensions])

    param_info = [
        ParameterDisplayDescription(key="", value=f"Metric: {metric_display}"),
        ParameterDisplayDescription(key="", value=f"Period: {current_period} vs {prior_period}"),
        ParameterDisplayDescription(key="", value=f"Dimensions: {dimensions_display}")
    ]

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=insights,
        visualizations=viz_list,
        parameter_display_descriptions=param_info,
        export_data=[ExportData(name=name, data=df) for name, df in export_data.items()]
    )
