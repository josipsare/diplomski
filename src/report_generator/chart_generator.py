"""
Chart Generator for Reports

Generates charts for both interactive HTML reports and static PDF reports.
Uses Plotly for interactive charts and matplotlib for static images.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# Import plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation


class ReportChartGenerator:
    """
    Generates charts for reports in both interactive (Plotly) and static (matplotlib) formats.
    """

    # Color scheme for risk levels
    RISK_COLORS = {
        'low': '#28a745',      # Green
        'medium': '#ffc107',   # Yellow
        'high': '#fd7e14',     # Orange
        'critical': '#dc3545', # Red
        'unknown': '#6c757d'   # Gray
    }

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize chart generator.

        Args:
            output_dir: Directory to save generated chart images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_risk_distribution_pie(
        self,
        distribution: Dict[str, int],
        for_pdf: bool = False
    ) -> Union[str, 'go.Figure']:
        """
        Create risk distribution pie chart.

        Args:
            distribution: Dict mapping risk levels to counts
            for_pdf: If True, return file path; if False, return Plotly figure

        Returns:
            File path (str) if for_pdf=True, else Plotly Figure
        """
        labels = list(distribution.keys())
        values = list(distribution.values())
        colors = [self.RISK_COLORS.get(k, '#6c757d') for k in labels]

        # Capitalize labels for display
        display_labels = [l.capitalize() for l in labels]

        if for_pdf or not PLOTLY_AVAILABLE:
            # Use matplotlib for static image
            fig, ax = plt.subplots(figsize=(8, 6))

            wedges, texts, autotexts = ax.pie(
                values,
                labels=display_labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                pctdistance=0.75
            )

            # Draw circle for donut effect
            centre_circle = plt.Circle((0, 0), 0.50, fc='white')
            ax.add_patch(centre_circle)

            ax.set_title('Portfolio Risk Distribution', fontsize=14, fontweight='bold')

            # Make text more readable
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')

            plt.tight_layout()
            path = self.output_dir / "risk_distribution.png"
            plt.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(path)

        else:
            # Use Plotly for interactive chart
            fig = go.Figure(data=[go.Pie(
                labels=display_labels,
                values=values,
                marker_colors=colors,
                textinfo='label+percent',
                hole=0.4,
                textfont=dict(size=12)
            )])

            fig.update_layout(
                title=dict(text="Portfolio Risk Distribution", font=dict(size=16)),
                font=dict(family="Arial", size=12),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1)
            )

            return fig

    def create_mad_histogram(
        self,
        mad_values: List[float],
        for_pdf: bool = False
    ) -> Union[str, 'go.Figure']:
        """
        Create MAD distribution histogram with risk bands.

        Args:
            mad_values: List of MAD values
            for_pdf: If True, return file path; if False, return Plotly figure

        Returns:
            File path (str) if for_pdf=True, else Plotly Figure
        """
        mad_values = [v for v in mad_values if v > 0]

        if for_pdf or not PLOTLY_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create histogram
            n, bins, patches = ax.hist(mad_values, bins=40, edgecolor='black', alpha=0.7)

            # Color bins by risk level
            for i, patch in enumerate(patches):
                bin_center = (bins[i] + bins[i + 1]) / 2
                if bin_center < 1.5:
                    patch.set_facecolor(self.RISK_COLORS['low'])
                elif bin_center < 2.5:
                    patch.set_facecolor(self.RISK_COLORS['medium'])
                elif bin_center < 3.5:
                    patch.set_facecolor(self.RISK_COLORS['high'])
                else:
                    patch.set_facecolor(self.RISK_COLORS['critical'])

            # Add threshold lines
            ax.axvline(x=1.5, color='orange', linestyle='--', linewidth=2, label='Low/Medium (1.5)')
            ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='Medium/High (2.5)')

            ax.set_xlabel('MAD Value', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Distribution of MAD Values by Risk Level', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')

            # Add statistics box
            stats_text = f'Mean: {np.mean(mad_values):.3f}\nMedian: {np.median(mad_values):.3f}\nStd: {np.std(mad_values):.3f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            plt.tight_layout()
            path = self.output_dir / "mad_histogram.png"
            plt.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(path)

        else:
            # Plotly version
            fig = go.Figure()

            # Create histogram with colored bars
            fig.add_trace(go.Histogram(
                x=mad_values,
                nbinsx=40,
                marker_color='steelblue',
                opacity=0.7,
                name='MAD Distribution'
            ))

            # Add threshold lines
            fig.add_vline(x=1.5, line_dash="dash", line_color="orange",
                         annotation_text="Low/Medium")
            fig.add_vline(x=2.5, line_dash="dash", line_color="red",
                         annotation_text="Medium/High")

            fig.update_layout(
                title="Distribution of MAD Values",
                xaxis_title="MAD Value",
                yaxis_title="Count",
                bargap=0.05
            )

            return fig

    def create_trend_chart(
        self,
        yearly_stats: pd.DataFrame,
        for_pdf: bool = False
    ) -> Union[str, 'go.Figure']:
        """
        Create trend chart showing MAD over years.

        Args:
            yearly_stats: DataFrame with yearly statistics
            for_pdf: If True, return file path; if False, return Plotly figure

        Returns:
            File path (str) if for_pdf=True, else Plotly Figure
        """
        if yearly_stats.empty:
            return None

        if for_pdf or not PLOTLY_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))

            years = yearly_stats['year'].values
            mean_mad = yearly_stats['mean_MAD'].values
            std_mad = yearly_stats['std_MAD'].values

            ax.plot(years, mean_mad, marker='o', linewidth=2, color='blue', label='Mean MAD')
            ax.fill_between(years, mean_mad - std_mad, mean_mad + std_mad,
                           alpha=0.3, color='blue', label='±1 Std Dev')

            if 'median_MAD' in yearly_stats.columns:
                ax.plot(years, yearly_stats['median_MAD'].values, marker='s',
                       linewidth=2, color='green', linestyle='--', label='Median MAD')

            ax.axhline(y=1.5, color='orange', linestyle=':', label='Threshold (1.5)')

            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('MAD Value', fontsize=12)
            ax.set_title('Benford Conformance Trend Over Time', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right')
            ax.set_xticks(years)

            plt.tight_layout()
            path = self.output_dir / "trend_chart.png"
            plt.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(path)

        else:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=yearly_stats['year'],
                y=yearly_stats['mean_MAD'],
                mode='lines+markers',
                name='Mean MAD',
                line=dict(color='blue', width=2)
            ))

            # Add confidence band
            fig.add_trace(go.Scatter(
                x=list(yearly_stats['year']) + list(yearly_stats['year'][::-1]),
                y=list(yearly_stats['mean_MAD'] + yearly_stats['std_MAD']) +
                  list((yearly_stats['mean_MAD'] - yearly_stats['std_MAD'])[::-1]),
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev'
            ))

            fig.add_hline(y=1.5, line_dash="dot", line_color="orange",
                         annotation_text="Threshold")

            fig.update_layout(
                title="Benford Conformance Trend Over Time",
                xaxis_title="Year",
                yaxis_title="MAD Value"
            )

            return fig

    def create_top_companies_chart(
        self,
        companies: List[Dict],
        n: int = 20,
        for_pdf: bool = False
    ) -> Union[str, 'go.Figure']:
        """
        Create horizontal bar chart of top suspicious companies.

        Args:
            companies: List of company dicts with 'company_name' and 'avg_mad'
            n: Number of companies to show
            for_pdf: If True, return file path; if False, return Plotly figure

        Returns:
            File path (str) if for_pdf=True, else Plotly Figure
        """
        # Sort and limit
        sorted_companies = sorted(companies, key=lambda x: x.get('avg_mad', 0), reverse=True)[:n]

        names = [c.get('company_name', 'Unknown')[:35] for c in sorted_companies]
        values = [c.get('avg_mad', 0) for c in sorted_companies]
        risk_levels = [c.get('risk_level', 'unknown') for c in sorted_companies]
        colors = [self.RISK_COLORS.get(r, '#6c757d') for r in risk_levels]

        if for_pdf or not PLOTLY_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 10))

            y_pos = range(len(names))
            bars = ax.barh(y_pos, values, color=colors, alpha=0.8)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()

            ax.axvline(x=1.5, color='orange', linestyle='--', linewidth=1.5)
            ax.axvline(x=2.5, color='red', linestyle='--', linewidth=1.5)

            ax.set_xlabel('Average MAD', fontsize=12)
            ax.set_title(f'Top {n} Companies by Benford Deviation', fontsize=14, fontweight='bold')

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=8)

            plt.tight_layout()
            path = self.output_dir / "top_companies.png"
            plt.savefig(str(path), dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            return str(path)

        else:
            fig = go.Figure(go.Bar(
                x=values,
                y=names,
                orientation='h',
                marker_color=colors,
                text=[f'{v:.2f}' for v in values],
                textposition='outside'
            ))

            fig.add_vline(x=1.5, line_dash="dash", line_color="orange")
            fig.add_vline(x=2.5, line_dash="dash", line_color="red")

            fig.update_layout(
                title=f"Top {n} Companies by Benford Deviation",
                xaxis_title="Average MAD",
                yaxis=dict(autorange="reversed"),
                height=max(400, n * 25)
            )

            return fig

    def create_company_sparkline(
        self,
        years: List[int],
        values: List[float],
        threshold: float = 2.5
    ) -> str:
        """
        Create small sparkline chart for company trend.

        Args:
            years: List of years
            values: List of MAD values
            threshold: Threshold line value

        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(3, 1))

        ax.plot(years, values, color='steelblue', linewidth=1.5)
        ax.axhline(y=threshold, color='red', linestyle=':', linewidth=0.5, alpha=0.5)

        ax.set_xlim(min(years), max(years))
        ax.axis('off')

        path = self.output_dir / f"sparkline_{hash(tuple(values))}.png"
        plt.savefig(str(path), dpi=100, bbox_inches='tight', facecolor='white',
                   transparent=True, pad_inches=0.01)
        plt.close()
        return str(path)

    def generate_all_charts(
        self,
        summary: 'PortfolioSummary',
        companies: List['CompanyReport'],
        yearly_stats: pd.DataFrame,
        for_pdf: bool = False
    ) -> Dict[str, Union[str, 'go.Figure']]:
        """
        Generate all charts for a report.

        Args:
            summary: PortfolioSummary object
            companies: List of CompanyReport objects
            yearly_stats: DataFrame with yearly statistics
            for_pdf: Generate static images for PDF

        Returns:
            Dict mapping chart names to file paths or Plotly figures
        """
        charts = {}

        # Risk distribution pie
        charts['risk_pie'] = self.create_risk_distribution_pie(
            summary.risk_distribution, for_pdf=for_pdf
        )

        # MAD histogram
        mad_values = [c.avg_mad for c in companies if c.avg_mad > 0]
        if mad_values:
            charts['mad_histogram'] = self.create_mad_histogram(mad_values, for_pdf=for_pdf)

        # Trend chart
        if not yearly_stats.empty:
            charts['trend'] = self.create_trend_chart(yearly_stats, for_pdf=for_pdf)

        # Top companies
        company_data = [{'company_name': c.company_name, 'avg_mad': c.avg_mad,
                        'risk_level': c.risk_level} for c in companies]
        charts['top_companies'] = self.create_top_companies_chart(
            company_data, n=20, for_pdf=for_pdf
        )

        return charts
