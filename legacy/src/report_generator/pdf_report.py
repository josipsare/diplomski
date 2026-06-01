"""
PDF Report Generator

Generates PDF reports from Benford analysis data using WeasyPrint.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List
import base64

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from .data_collector import ReportDataCollector, PortfolioSummary, CompanyReport
from .chart_generator import ReportChartGenerator


class PDFReportGenerator:
    """
    Generates PDF reports with embedded static charts using WeasyPrint.
    """

    # PDF-specific CSS for print layout
    PDF_CSS = '''
        @page {
            size: A4;
            margin: 2cm;
            @top-center {
                content: "Benford's Law Analysis Report";
                font-size: 9pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
            }
        }

        body {
            font-family: 'Helvetica', 'Arial', sans-serif;
            font-size: 10pt;
            line-height: 1.4;
        }

        h1 { font-size: 20pt; }
        h2 { font-size: 14pt; page-break-after: avoid; }
        h3 { font-size: 12pt; page-break-after: avoid; }

        .page-break {
            page-break-after: always;
        }

        .no-break, .company-card {
            page-break-inside: avoid;
        }

        table {
            page-break-inside: avoid;
        }

        .chart-container {
            page-break-inside: avoid;
            text-align: center;
        }

        .chart-container img {
            max-width: 100%;
            max-height: 400px;
        }

        .summary-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .summary-card {
            flex: 1;
            min-width: 120px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
            text-align: center;
        }

        .risk-badge {
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }
    '''

    def __init__(
        self,
        template_dir: Optional[str] = None,
        output_dir: str = "data/output/reports"
    ):
        """
        Initialize PDF report generator.

        Args:
            template_dir: Directory containing Jinja2 templates
            output_dir: Directory to save generated reports
        """
        if not WEASYPRINT_AVAILABLE:
            raise ImportError("weasyprint is required for PDF report generation. "
                            "Install it with: pip install weasyprint")

        if not JINJA2_AVAILABLE:
            raise ImportError("jinja2 is required for PDF report generation. "
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

    def _embed_image(self, image_path: str) -> str:
        """
        Convert image to base64 data URI for PDF embedding.

        Args:
            image_path: Path to image file

        Returns:
            Base64 data URI string
        """
        try:
            with open(image_path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')

            suffix = Path(image_path).suffix.lower()
            mime_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.svg': 'image/svg+xml',
                '.gif': 'image/gif'
            }
            mime = mime_types.get(suffix, 'image/png')

            return f"data:{mime};base64,{data}"
        except Exception as e:
            print(f"Warning: Could not embed image {image_path}: {e}")
            return ""

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
        output_filename: str = "benford_analysis_report.pdf",
        top_n_companies: int = 50,
        show_company_details: bool = True
    ) -> str:
        """
        Generate PDF report.

        Args:
            data_collector: ReportDataCollector with loaded data
            chart_generator: ReportChartGenerator for creating charts
            output_filename: Name of output PDF file
            top_n_companies: Number of companies for detailed breakdown
            show_company_details: Include individual company sections

        Returns:
            Path to generated PDF file
        """
        # Collect data
        summary = data_collector.get_portfolio_summary()
        all_companies = data_collector.get_company_reports()
        top_companies = data_collector.get_top_suspicious_companies(n=min(20, len(all_companies)))
        detailed_companies = data_collector.get_company_reports(top_n=top_n_companies) if show_company_details else []
        yearly_stats = data_collector.get_yearly_statistics()

        # Generate static chart images
        charts = {}

        # Risk distribution pie
        risk_pie_path = chart_generator.create_risk_distribution_pie(
            summary.risk_distribution, for_pdf=True
        )
        if risk_pie_path:
            charts['risk_pie'] = self._embed_image(risk_pie_path)

        # MAD histogram
        mad_values = [c.avg_mad for c in all_companies if c.avg_mad > 0]
        if mad_values:
            mad_hist_path = chart_generator.create_mad_histogram(mad_values, for_pdf=True)
            if mad_hist_path:
                charts['mad_histogram'] = self._embed_image(mad_hist_path)

        # Trend chart
        if not yearly_stats.empty:
            trend_path = chart_generator.create_trend_chart(yearly_stats, for_pdf=True)
            if trend_path:
                charts['trend'] = self._embed_image(trend_path)

        # Load base styles and add PDF-specific styles
        base_styles = self._load_styles()
        styles = base_styles + "\n" + self.PDF_CSS

        # Render HTML template
        template = self.env.get_template("report.html")
        html_content = template.render(
            title="Benford's Law Analysis Report",
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            summary=summary,
            top_companies=[self._company_to_dict(c) for c in top_companies],
            companies=[self._company_to_dict(c) for c in detailed_companies],
            charts=charts,
            styles=styles,
            interactive=False,  # Disable interactive features for PDF
            show_company_details=show_company_details
        )

        # Convert HTML to PDF
        output_path = self.output_dir / output_filename

        try:
            html_doc = HTML(string=html_content, base_url=str(self.template_dir))
            html_doc.write_pdf(str(output_path))
        except Exception as e:
            print(f"Error generating PDF: {e}")
            # Save HTML as fallback
            html_fallback = output_path.with_suffix('.html')
            html_fallback.write_text(html_content, encoding='utf-8')
            print(f"HTML saved as fallback: {html_fallback}")
            return str(html_fallback)

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
