"""
Reckitt Price-Volume-Mix Driver Analysis

Adapted from FPA metric_drivers.py for Reckitt Surface Care POC data.
Uses net_revenue with units as volume to derive price for PVM decomposition.

Data structure:
- Revenue: net_revenue (Reckitt brands only)
- Volume: units (from Nielsen)
- Price: net_revenue / units (derived)
- Dimensions: brand, sub_category, segment, channel, state_name
- Time: quarter (Q2'25, Q3'25, Q4'25, Q1'26, Q2'26)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Default data path
DEFAULT_DATA_PATH = "/Users/mitchelltravis/cursor/reckitt_poc_data/reckitt_surface_care_poc.csv"


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
        'country_name': 'Country',
        'household_penetration': 'HH Penetration',
    }

    if name.lower() in special_cases:
        return special_cases[name.lower()]

    return name.replace('_', ' ').title()


class ReckittPVMAnalysis:
    """Price-Volume-Mix Analysis for Reckitt Surface Care Data

    Uses net_revenue as revenue metric and units as volume to derive price.
    Formula:
    - Volume Impact = (Actual Units - Prior Units) × Prior Price
    - Price Impact = (Actual Price - Prior Price) × Actual Units
    - Mix Impact = Residual (captures portfolio shift)
    """

    def __init__(self, data_path=None, metric='net_revenue', current_period=None,
                 prior_period=None, breakout_dimensions=None, top_n=10, filters=None):
        self.data_path = data_path or DEFAULT_DATA_PATH
        self.metric = metric  # net_revenue, gross_sales, or gross_margin
        self.current_period = current_period  # e.g., "Q2'26"
        self.prior_period = prior_period  # e.g., "Q2'25" for YoY
        self.breakout_dimensions = breakout_dimensions or ['brand', 'sub_category', 'segment', 'channel', 'state_name']
        self.top_n = top_n
        self.filters = filters or {}

        self.df = None
        self.current_df = None
        self.prior_df = None
        self.pvm_results = None
        self.breakout_results = {}
        self.facts = []

    def load_data(self):
        """Load and filter data for Reckitt brands only (they have finance data)"""
        logger.info(f"Loading data from {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # Filter to Reckitt brands only (they have net_revenue)
        reckitt_df = self.df[self.df['manufacturer'] == 'RECKITT BENCKISER'].copy()

        # Apply any additional filters
        for col, val in self.filters.items():
            if col in reckitt_df.columns:
                if isinstance(val, list):
                    reckitt_df = reckitt_df[reckitt_df[col].isin(val)]
                else:
                    reckitt_df = reckitt_df[reckitt_df[col] == val]

        # Filter by period
        self.current_df = reckitt_df[reckitt_df['quarter'] == self.current_period].copy()
        self.prior_df = reckitt_df[reckitt_df['quarter'] == self.prior_period].copy()

        logger.info(f"Current period ({self.current_period}): {len(self.current_df):,} rows")
        logger.info(f"Prior period ({self.prior_period}): {len(self.prior_df):,} rows")

        if self.current_df.empty:
            raise ValueError(f"No data found for current period: {self.current_period}")
        if self.prior_df.empty:
            raise ValueError(f"No data found for prior period: {self.prior_period}")

    def calculate_price_volume_mix(self):
        """
        Calculate Price-Volume-Mix decomposition using category-level detail

        Uses units as volume and derives price = net_revenue / units

        Formula:
        - Volume Impact = (Actual Units - Prior Units) × Prior Avg Price
        - Mix Impact = Change in category shares × Prior prices × Actual volume
        - Price Impact = Residual
        """
        logger.info("Calculating Price-Volume-Mix decomposition")

        # Use the first breakout dimension for mix calculation (typically brand)
        mix_dimension = self.breakout_dimensions[0] if self.breakout_dimensions else 'brand'

        # Calculate totals
        actual_revenue = self.current_df[self.metric].sum()
        prior_revenue = self.prior_df[self.metric].sum()
        total_variance = actual_revenue - prior_revenue

        actual_units = self.current_df['units'].sum()
        prior_units = self.prior_df['units'].sum()

        # Category-level PVM calculation
        if mix_dimension in self.current_df.columns:
            logger.info(f"Using category-level PVM with dimension: {mix_dimension}")

            # Aggregate by category
            actual_by_cat = self.current_df.groupby(mix_dimension).agg({
                self.metric: 'sum',
                'units': 'sum'
            }).reset_index()
            actual_by_cat['price'] = actual_by_cat[self.metric] / actual_by_cat['units'].replace(0, np.nan)
            actual_by_cat['price'] = actual_by_cat['price'].fillna(0)

            prior_by_cat = self.prior_df.groupby(mix_dimension).agg({
                self.metric: 'sum',
                'units': 'sum'
            }).reset_index()
            prior_by_cat['price'] = prior_by_cat[self.metric] / prior_by_cat['units'].replace(0, np.nan)
            prior_by_cat['price'] = prior_by_cat['price'].fillna(0)

            # Merge to align categories
            merged = pd.merge(
                actual_by_cat,
                prior_by_cat,
                on=mix_dimension,
                how='outer',
                suffixes=('_actual', '_prior')
            ).fillna(0)

            # Calculate total volumes
            total_actual_units = actual_by_cat['units'].sum()
            total_prior_units = prior_by_cat['units'].sum()

            # Step 1: Mix Impact - change in category shares at prior prices
            mix_impact = 0
            for _, row in merged.iterrows():
                actual_share = row['units_actual'] / total_actual_units if total_actual_units > 0 else 0
                prior_share = row['units_prior'] / total_prior_units if total_prior_units > 0 else 0
                share_change = actual_share - prior_share
                mix_impact += share_change * row['price_prior'] * total_actual_units

            # Step 2: Volume Impact - total volume change at prior average price
            prior_avg_price = prior_revenue / total_prior_units if total_prior_units > 0 else 0
            volume_impact = (total_actual_units - total_prior_units) * prior_avg_price

            # Step 3: Price is residual
            price_impact = total_variance - volume_impact - mix_impact

        else:
            # Simple aggregate calculation
            logger.info("Using simple aggregate PVM calculation")

            actual_price = actual_revenue / actual_units if actual_units > 0 else 0
            prior_price = prior_revenue / prior_units if prior_units > 0 else 0

            volume_impact = (actual_units - prior_units) * prior_price
            price_impact = (actual_price - prior_price) * actual_units
            mix_impact = total_variance - volume_impact - price_impact

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
            'prior_price': prior_revenue / prior_units if prior_units > 0 else 0,
            'actual_price': actual_revenue / actual_units if actual_units > 0 else 0
        }

        # Create facts
        pct_change = (total_variance / prior_revenue * 100) if prior_revenue != 0 else 0
        self.facts.append({
            'fact': f"Total {self.metric.replace('_', ' ').title()} variance: {format_number(total_variance)} ({pct_change:+.1f}%)",
            'category': 'overall'
        })

        if total_variance != 0:
            self.facts.append({
                'fact': f"Volume impact: {format_number(volume_impact)} ({volume_impact/abs(total_variance)*100:.1f}% of variance)",
                'category': 'pvm'
            })
            self.facts.append({
                'fact': f"Price impact: {format_number(price_impact)} ({price_impact/abs(total_variance)*100:.1f}% of variance)",
                'category': 'pvm'
            })
            self.facts.append({
                'fact': f"Mix impact: {format_number(mix_impact)} ({mix_impact/abs(total_variance)*100:.1f}% of variance)",
                'category': 'pvm'
            })

        return self.pvm_results

    def calculate_dimensional_breakout(self, dimension):
        """Calculate variance attribution by dimension"""
        logger.info(f"Calculating breakout for dimension: {dimension}")

        if dimension not in self.current_df.columns:
            logger.warning(f"Dimension {dimension} not found in data")
            return None

        # Aggregate by dimension
        actual_agg = self.current_df.groupby(dimension).agg({
            self.metric: 'sum',
            'units': 'sum'
        }).reset_index()
        actual_agg.columns = [dimension, 'actual_revenue', 'actual_units']

        prior_agg = self.prior_df.groupby(dimension).agg({
            self.metric: 'sum',
            'units': 'sum'
        }).reset_index()
        prior_agg.columns = [dimension, 'prior_revenue', 'prior_units']

        merged = pd.merge(actual_agg, prior_agg, on=dimension, how='outer').fillna(0)

        # Calculate derived metrics
        merged['actual_price'] = merged['actual_revenue'] / merged['actual_units'].replace(0, np.nan)
        merged['prior_price'] = merged['prior_revenue'] / merged['prior_units'].replace(0, np.nan)
        merged['actual_price'] = merged['actual_price'].fillna(0)
        merged['prior_price'] = merged['prior_price'].fillna(0)

        merged['variance'] = merged['actual_revenue'] - merged['prior_revenue']
        merged['variance_pct'] = (merged['variance'] / merged['prior_revenue'].replace(0, np.nan) * 100).fillna(0)

        merged['units_change'] = merged['actual_units'] - merged['prior_units']
        merged['price_change'] = merged['actual_price'] - merged['prior_price']

        # Rank by absolute variance
        merged['abs_variance'] = merged['variance'].abs()
        merged = merged.sort_values('abs_variance', ascending=False)

        # Take top N
        top_n_df = merged.head(self.top_n).copy()
        self.breakout_results[dimension] = top_n_df

        # Add facts for top contributors
        for _, row in top_n_df.head(3).iterrows():
            direction = "increased" if row['variance'] > 0 else "decreased"
            self.facts.append({
                'fact': f"{dimension.replace('_', ' ').title()} '{row[dimension]}': {format_number(row['variance'])} ({row['variance_pct']:+.1f}%) - {direction}",
                'category': f'breakout_{dimension}'
            })

        return top_n_df

    def create_waterfall_chart_data(self):
        """Create Highcharts waterfall chart configuration"""
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

        def get_color(value):
            return '#4ade80' if value >= 0 else '#ef4444'

        volume_val = self.pvm_results['volume_impact']
        price_val = self.pvm_results['price_impact']
        mix_val = self.pvm_results['mix_impact']

        # Scale for display
        scale = 1_000_000 if abs(self.pvm_results['starting_value']) >= 1_000_000 else 1_000
        scale_label = 'M' if scale == 1_000_000 else 'K'

        starting_scaled = self.pvm_results['starting_value'] / scale
        volume_scaled = volume_val / scale
        price_scaled = price_val / scale
        mix_scaled = mix_val / scale
        ending_scaled = self.pvm_results['ending_value'] / scale

        data_series = [{
            'name': metric_display,
            'data': [
                {
                    'name': self.prior_period,
                    'y': starting_scaled,
                    'color': '#3b82f6',
                    'dataLabels': {'enabled': True, 'format': format_number(self.pvm_results['starting_value'])}
                },
                {
                    'name': 'Volume',
                    'y': volume_scaled,
                    'color': get_color(volume_val),
                    'dataLabels': {'enabled': True, 'format': format_number(volume_val)}
                },
                {
                    'name': 'Price',
                    'y': price_scaled,
                    'color': get_color(price_val),
                    'dataLabels': {'enabled': True, 'format': format_number(price_val)}
                },
                {
                    'name': 'Mix',
                    'y': mix_scaled,
                    'color': get_color(mix_val),
                    'dataLabels': {'enabled': True, 'format': format_number(mix_val)}
                },
                {
                    'name': self.current_period,
                    'isSum': True,
                    'y': ending_scaled,
                    'color': '#3b82f6',
                    'dataLabels': {'enabled': True, 'format': format_number(self.pvm_results['ending_value'])}
                }
            ],
            'dataLabels': {
                'enabled': True,
                'style': {'fontWeight': 'bold', 'color': '#000000', 'textOutline': 'none'}
            }
        }]

        return {
            'chart_categories': categories,
            'chart_data': data_series,
            'chart_y_axis': {
                'title': {'text': f'{metric_display} (${scale_label})'},
                'labels': {'format': '${value:,.0f}' + scale_label}
            },
            'chart_title': f'{metric_display} Price-Volume-Mix Bridge'
        }

    def create_horizontal_bar_chart_data(self, dimension):
        """Create horizontal bar chart for dimension breakout"""
        if dimension not in self.breakout_results:
            return None

        df = self.breakout_results[dimension]
        scale = 1_000_000 if df['actual_revenue'].abs().max() >= 1_000_000 else 1_000

        categories = df[dimension].tolist()
        actual_data = [x / scale for x in df['actual_revenue'].tolist()]
        prior_data = [x / scale for x in df['prior_revenue'].tolist()]

        return {
            'chart_categories': categories,
            'chart_data': [
                {'name': self.current_period, 'data': actual_data, 'color': '#5DADE2'},
                {'name': self.prior_period, 'data': prior_data, 'color': '#F8C471'}
            ],
            'chart_y_axis': {
                'title': {'text': format_display_name(self.metric)},
                'labels': {'format': '${value:,.0f}' + ('M' if scale == 1_000_000 else 'K')}
            },
            'chart_title': f'{format_display_name(dimension)} Variance Analysis'
        }

    def get_summary_table(self):
        """Create summary table with key metrics"""
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
                "Units",
                f"{self.pvm_results['actual_units']:,.0f}",
                f"{self.pvm_results['prior_units']:,.0f}",
                f"{self.pvm_results['actual_units'] - self.pvm_results['prior_units']:+,.0f}",
                f"{(self.pvm_results['actual_units'] - self.pvm_results['prior_units']) / self.pvm_results['prior_units'] * 100:+.1f}%" if self.pvm_results['prior_units'] != 0 else "N/A"
            ],
            [
                "Price ($/Unit)",
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
                f"{self.pvm_results['volume_impact'] / abs(self.pvm_results['total_variance']) * 100:.1f}% of variance" if self.pvm_results['total_variance'] != 0 else "N/A"
            ],
            [
                "Price Impact",
                "", "",
                format_number(self.pvm_results['price_impact']),
                f"{self.pvm_results['price_impact'] / abs(self.pvm_results['total_variance']) * 100:.1f}% of variance" if self.pvm_results['total_variance'] != 0 else "N/A"
            ],
            [
                "Mix Impact",
                "", "",
                format_number(self.pvm_results['mix_impact']),
                f"{self.pvm_results['mix_impact'] / abs(self.pvm_results['total_variance']) * 100:.1f}% of variance" if self.pvm_results['total_variance'] != 0 else "N/A"
            ]
        ]

        columns = [
            {'name': 'Metric'},
            {'name': self.current_period},
            {'name': self.prior_period},
            {'name': 'Change'},
            {'name': 'Change %'}
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
                format_number(row['actual_revenue']),
                format_number(row['prior_revenue']),
                format_number(row['variance']),
                f"{row['variance_pct']:+.1f}%",
                f"{row['units_change']:+,.0f}",
                f"${row['price_change']:+.2f}" if not pd.isna(row['price_change']) else "N/A"
            ])

        columns = [
            {'name': format_display_name(dimension)},
            {'name': self.current_period},
            {'name': self.prior_period},
            {'name': 'Variance'},
            {'name': 'Var %'},
            {'name': 'Units Δ'},
            {'name': 'Price Δ'}
        ]

        return {'data': data, 'col_defs': columns}

    def run_analysis(self):
        """Run complete PVM analysis"""
        logger.info(f"Starting Reckitt PVM analysis: {self.current_period} vs {self.prior_period}")

        self.load_data()
        self.calculate_price_volume_mix()

        for dim in self.breakout_dimensions:
            self.calculate_dimensional_breakout(dim)

        logger.info("Analysis complete")
        return self

    def print_summary(self):
        """Print analysis summary to console"""
        print("\n" + "="*60)
        print(f"RECKITT PRICE-VOLUME-MIX ANALYSIS")
        print(f"{self.current_period} vs {self.prior_period}")
        print("="*60)

        if self.pvm_results:
            print(f"\n{format_display_name(self.metric)} Bridge:")
            print(f"  {self.prior_period}:     {format_number(self.pvm_results['starting_value'])}")
            print(f"  + Volume:       {format_number(self.pvm_results['volume_impact'])}")
            print(f"  + Price:        {format_number(self.pvm_results['price_impact'])}")
            print(f"  + Mix:          {format_number(self.pvm_results['mix_impact'])}")
            print(f"  = {self.current_period}:     {format_number(self.pvm_results['ending_value'])}")
            print(f"\n  Total Variance: {format_number(self.pvm_results['total_variance'])} ({self.pvm_results['total_variance']/self.pvm_results['starting_value']*100:+.1f}%)")

        print("\n" + "-"*60)
        print("KEY INSIGHTS:")
        for fact in self.facts[:10]:
            print(f"  • {fact['fact']}")

        print("\n" + "="*60)


def run_reckitt_pvm(current_period="Q2'26", prior_period="Q2'25", metric='net_revenue',
                    breakout_dimensions=None, data_path=None, filters=None):
    """
    Convenience function to run Reckitt PVM analysis

    Args:
        current_period: Current period (e.g., "Q2'26")
        prior_period: Prior period for comparison (e.g., "Q2'25")
        metric: Revenue metric to analyze ('net_revenue', 'gross_sales', 'gross_margin')
        breakout_dimensions: List of dimensions for breakout analysis
        data_path: Path to CSV file
        filters: Dict of column->value filters

    Returns:
        ReckittPVMAnalysis instance with results
    """
    if breakout_dimensions is None:
        breakout_dimensions = ['brand', 'sub_category', 'segment', 'channel', 'state_name']

    analysis = ReckittPVMAnalysis(
        data_path=data_path,
        metric=metric,
        current_period=current_period,
        prior_period=prior_period,
        breakout_dimensions=breakout_dimensions,
        filters=filters or {}
    )

    analysis.run_analysis()
    analysis.print_summary()

    return analysis


if __name__ == "__main__":
    # Example: YoY comparison Q2'26 vs Q2'25
    analysis = run_reckitt_pvm(
        current_period="Q2'26",
        prior_period="Q2'25",
        metric='net_revenue'
    )

    # Show top brand contributors
    print("\nTop Brand Contributors:")
    if 'brand' in analysis.breakout_results:
        print(analysis.breakout_results['brand'][['brand', 'variance', 'variance_pct']].head(10).to_string())
