"""
HTML Report Generator

Generates interactive HTML reports from Benford analysis data.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List
import json

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from .data_collector import ReportDataCollector, PortfolioSummary, CompanyReport
from .chart_generator import ReportChartGenerator


class HTMLReportGenerator:
    """
    Generates interactive HTML reports with embedded Plotly charts.
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        output_dir: str = "data/output/reports"
    ):
        """
        Initialize HTML report generator.

        Args:
            template_dir: Directory containing Jinja2 templates
            output_dir: Directory to save generated reports
        """
        if not JINJA2_AVAILABLE:
            raise ImportError("jinja2 is required for HTML report generation. "
                            "Install it with: pip install jinja2")

        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )

        # Register custom filters
        self.env.filters['format_number'] = self._format_number
        self.env.filters['format_pct'] = self._format_pct
        self.env.filters['risk_color'] = self._get_risk_color

    @staticmethod
    def _format_number(value, decimals=2):
        """Format number with commas and decimals."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):,.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _format_pct(value, decimals=1):
        """Format percentage value."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}%"
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _get_risk_color(risk_level: str) -> str:
        """Return CSS color for risk level."""
        colors = {
            'low': '#28a745',
            'medium': '#ffc107',
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        return colors.get(risk_level, '#6c757d')

    def _load_styles(self) -> str:
        """Load CSS styles from template directory."""
        styles_path = self.template_dir / "styles.css"
        if styles_path.exists():
            return styles_path.read_text()
        return ""

    def generate_report(
        self,
        data_collector: ReportDataCollector,
        chart_generator: ReportChartGenerator,
        output_filename: str = "benford_analysis_report.html",
        top_n_companies: int = 50,
        show_company_details: bool = True,
        interactive: bool = True
    ) -> str:
        """
        Generate complete HTML report.

        Args:
            data_collector: ReportDataCollector with loaded data
            chart_generator: ReportChartGenerator for creating charts
            output_filename: Name of output HTML file
            top_n_companies: Number of companies for detailed breakdown
            show_company_details: Include individual company sections
            interactive: Enable interactive Plotly charts

        Returns:
            Path to generated HTML file
        """
        # Collect data
        summary = data_collector.get_portfolio_summary()
        all_companies = data_collector.get_company_reports()
        top_companies = data_collector.get_top_suspicious_companies(n=min(20, len(all_companies)))
        detailed_companies = data_collector.get_company_reports(top_n=top_n_companies) if show_company_details else []
        yearly_stats = data_collector.get_yearly_statistics()

        # Generate charts
        charts = {}

        # For interactive mode, generate Plotly figures and convert to JSON
        if interactive:
            risk_pie_fig = chart_generator.create_risk_distribution_pie(
                summary.risk_distribution, for_pdf=False
            )
            if risk_pie_fig:
                charts['risk_pie_json'] = risk_pie_fig.to_json()

            mad_values = [c.avg_mad for c in all_companies if c.avg_mad > 0]
            if mad_values:
                mad_fig = chart_generator.create_mad_histogram(mad_values, for_pdf=False)
                if mad_fig:
                    charts['mad_histogram_json'] = mad_fig.to_json()

            if not yearly_stats.empty:
                trend_fig = chart_generator.create_trend_chart(yearly_stats, for_pdf=False)
                if trend_fig:
                    charts['trend_json'] = trend_fig.to_json()
        else:
            # Generate static images
            charts = chart_generator.generate_all_charts(
                summary, all_companies, yearly_stats, for_pdf=True
            )

        # Load CSS styles
        styles = self._load_styles()

        # Render template
        template = self.env.get_template("report.html")
        html_content = template.render(
            title="Benford's Law Analysis Report",
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            summary=summary,
            top_companies=[self._company_to_dict(c) for c in top_companies],
            companies=[self._company_to_dict(c) for c in detailed_companies],
            charts=charts,
            styles=styles,
            interactive=interactive,
            show_company_details=show_company_details
        )

        # Save to file
        output_path = self.output_dir / output_filename
        output_path.write_text(html_content, encoding='utf-8')

        return str(output_path)

    @staticmethod
    def _company_to_dict(company: CompanyReport) -> dict:
        """Convert CompanyReport to dictionary for template."""
        return {
            'cik': company.cik,
            'company_name': company.company_name,
            'symbol': company.symbol,
            'risk_level': company.risk_level,
            'anomaly_score': company.anomaly_score,
            'years': company.years,
            'mad_values': company.mad_values,
            'chi_square_values': company.chi_square_values,
            'p_values': company.p_values,
            'stock_returns': company.stock_returns,
            'avg_mad': company.avg_mad,
            'trend_direction': company.trend_direction
        }
