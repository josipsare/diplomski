"""
Data Collector for Report Generation

Aggregates and prepares data from Benford analysis results for report generation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class CompanyReport:
    """Data structure for individual company report data."""
    cik: str
    company_name: str
    symbol: Optional[str] = None
    risk_level: str = 'unknown'  # 'low', 'medium', 'high', 'critical'
    anomaly_score: float = 0.0
    years: List[int] = field(default_factory=list)
    mad_values: Dict[int, float] = field(default_factory=dict)
    chi_square_values: Dict[int, float] = field(default_factory=dict)
    p_values: Dict[int, float] = field(default_factory=dict)
    ks_values: Dict[int, float] = field(default_factory=dict)
    stock_returns: Optional[Dict[int, float]] = None
    volatility: Optional[Dict[int, float]] = None
    avg_mad: float = 0.0
    trend_direction: str = 'stable'  # 'improving', 'stable', 'worsening'


@dataclass
class PortfolioSummary:
    """Aggregate portfolio statistics."""
    total_companies: int = 0
    analysis_period: str = ""
    risk_distribution: Dict[str, int] = field(default_factory=dict)
    avg_mad: float = 0.0
    median_mad: float = 0.0
    std_mad: float = 0.0
    companies_above_threshold: int = 0
    companies_critical: int = 0
    yoy_trend: str = 'stable'
    key_findings: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)


class ReportDataCollector:
    """
    Collects and prepares data for report generation.

    Loads Benford analysis results and optionally stock data,
    then prepares structured data for the report templates.
    """

    # Risk classification thresholds
    RISK_THRESHOLDS = {
        'low': 1.5,
        'medium': 2.5,
        'high': 3.5
    }

    def __init__(
        self,
        benford_file: Union[str, Path],
        combined_file: Optional[Union[str, Path]] = None,
        years: Optional[List[int]] = None
    ):
        """
        Initialize the data collector.

        Args:
            benford_file: Path to Benford analysis CSV file
            combined_file: Optional path to combined Benford + stock data CSV
            years: List of years to include (default: 2014-2024)
        """
        self.benford_df = pd.read_csv(benford_file, dtype={'cik': str})
        self.benford_df['cik'] = self.benford_df['cik'].astype(str).str.zfill(10)

        self.combined_df = None
        if combined_file and Path(combined_file).exists():
            self.combined_df = pd.read_csv(combined_file, dtype={'cik': str})
            self.combined_df['cik'] = self.combined_df['cik'].astype(str).str.zfill(10)

        self.years = years or list(range(2014, 2025))
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare and validate loaded data."""
        # Calculate average MAD per company
        mad_cols = [f'year_{y}_MAD' for y in self.years if f'year_{y}_MAD' in self.benford_df.columns]

        if mad_cols:
            self.benford_df['avg_MAD'] = self.benford_df[mad_cols].replace(0, np.nan).mean(axis=1)
        else:
            self.benford_df['avg_MAD'] = 0

        # Classify risk levels
        self.benford_df['risk_level'] = self.benford_df['avg_MAD'].apply(self._classify_risk)

    def _classify_risk(self, avg_mad: float) -> str:
        """Classify company risk based on average MAD."""
        if pd.isna(avg_mad) or avg_mad == 0:
            return 'unknown'
        if avg_mad < self.RISK_THRESHOLDS['low']:
            return 'low'
        elif avg_mad < self.RISK_THRESHOLDS['medium']:
            return 'medium'
        elif avg_mad < self.RISK_THRESHOLDS['high']:
            return 'high'
        return 'critical'

    def _calculate_trend(self, values: Dict[int, float]) -> str:
        """Calculate trend direction from yearly values."""
        if len(values) < 3:
            return 'stable'

        years_sorted = sorted(values.keys())
        recent_years = years_sorted[-3:]
        recent_values = [values[y] for y in recent_years if values.get(y, 0) > 0]

        if len(recent_values) < 2:
            return 'stable'

        # Simple trend based on first vs last
        diff = recent_values[-1] - recent_values[0]
        if diff < -0.3:
            return 'improving'
        elif diff > 0.3:
            return 'worsening'
        return 'stable'

    def get_portfolio_summary(self) -> PortfolioSummary:
        """
        Generate aggregate portfolio statistics.

        Returns:
            PortfolioSummary with overall analysis statistics
        """
        df = self.benford_df

        # Risk distribution
        risk_dist = df['risk_level'].value_counts().to_dict()

        # Calculate statistics
        valid_mad = df['avg_MAD'][df['avg_MAD'] > 0]

        # Generate key findings
        findings = []

        n_critical = risk_dist.get('critical', 0)
        n_high = risk_dist.get('high', 0)
        n_total = len(df)

        if n_critical > 0:
            pct = n_critical / n_total * 100
            findings.append(f"{n_critical} companies ({pct:.1f}%) show critical Benford deviation")

        if n_high > 0:
            pct = n_high / n_total * 100
            findings.append(f"{n_high} companies ({pct:.1f}%) show high deviation")

        pct_conforming = (risk_dist.get('low', 0) / n_total * 100) if n_total > 0 else 0
        findings.append(f"{pct_conforming:.1f}% of companies show good Benford conformance")

        if not valid_mad.empty:
            findings.append(f"Average MAD across portfolio: {valid_mad.mean():.3f}")

        # Calculate year-over-year trend
        yoy_trend = 'stable'
        yearly_avgs = []
        for year in self.years[-3:]:
            col = f'year_{year}_MAD'
            if col in df.columns:
                yearly_avg = df[col][df[col] > 0].mean()
                if not pd.isna(yearly_avg):
                    yearly_avgs.append(yearly_avg)

        if len(yearly_avgs) >= 2:
            if yearly_avgs[-1] < yearly_avgs[0] - 0.1:
                yoy_trend = 'improving'
            elif yearly_avgs[-1] > yearly_avgs[0] + 0.1:
                yoy_trend = 'worsening'

        return PortfolioSummary(
            total_companies=n_total,
            analysis_period=f"{min(self.years)}-{max(self.years)}",
            risk_distribution=risk_dist,
            avg_mad=valid_mad.mean() if not valid_mad.empty else 0,
            median_mad=valid_mad.median() if not valid_mad.empty else 0,
            std_mad=valid_mad.std() if not valid_mad.empty else 0,
            companies_above_threshold=n_high + n_critical,
            companies_critical=n_critical,
            yoy_trend=yoy_trend,
            key_findings=findings,
            years=self.years
        )

    def get_company_reports(
        self,
        top_n: Optional[int] = None,
        filter_risk: Optional[str] = None,
        sort_by: str = 'avg_mad'
    ) -> List[CompanyReport]:
        """
        Get individual company report data.

        Args:
            top_n: Limit to top N companies by average MAD
            filter_risk: Filter by risk level ('low', 'medium', 'high', 'critical')
            sort_by: Sort field ('avg_mad', 'company_name')

        Returns:
            List of CompanyReport objects
        """
        df = self.benford_df.copy()

        # Filter by risk level if specified
        if filter_risk:
            df = df[df['risk_level'] == filter_risk]

        # Sort
        if sort_by == 'avg_mad':
            df = df.sort_values('avg_MAD', ascending=False)
        elif sort_by == 'company_name':
            df = df.sort_values('company_name')

        # Limit
        if top_n:
            df = df.head(top_n)

        reports = []
        for _, row in df.iterrows():
            # Extract yearly metrics
            mad_values = {}
            chi_values = {}
            p_values = {}
            ks_values = {}

            for year in self.years:
                mad_col = f'year_{year}_MAD'
                chi_col = f'year_{year}_chi_square'
                p_col = f'year_{year}_p_value'
                ks_col = f'year_{year}_KS_test'

                if mad_col in row and row[mad_col] > 0:
                    mad_values[year] = round(row[mad_col], 4)
                if chi_col in row and pd.notna(row[chi_col]):
                    chi_values[year] = round(row[chi_col], 4)
                if p_col in row and pd.notna(row[p_col]):
                    p_values[year] = round(row[p_col], 4)
                if ks_col in row and pd.notna(row[ks_col]):
                    ks_values[year] = round(row[ks_col], 4)

            # Get stock data if available
            stock_returns = None
            volatility = None
            if self.combined_df is not None:
                combined_row = self.combined_df[self.combined_df['cik'] == row['cik']]
                if not combined_row.empty:
                    stock_returns = {}
                    volatility = {}
                    for year in self.years:
                        ret_col = f'year_{year}_stock_annual_return'
                        vol_col = f'year_{year}_stock_volatility'
                        if ret_col in combined_row.columns:
                            val = combined_row[ret_col].iloc[0]
                            if pd.notna(val):
                                stock_returns[year] = round(val, 2)
                        if vol_col in combined_row.columns:
                            val = combined_row[vol_col].iloc[0]
                            if pd.notna(val):
                                volatility[year] = round(val, 4)

            # Calculate anomaly score (0-100 scale)
            avg_mad = row['avg_MAD'] if pd.notna(row['avg_MAD']) else 0
            anomaly_score = min(100, avg_mad * 33.3)  # MAD=3 -> score=100

            report = CompanyReport(
                cik=row['cik'],
                company_name=row.get('company_name', 'Unknown'),
                symbol=row.get('symbol'),
                risk_level=row['risk_level'],
                anomaly_score=round(anomaly_score, 1),
                years=list(mad_values.keys()),
                mad_values=mad_values,
                chi_square_values=chi_values,
                p_values=p_values,
                ks_values=ks_values,
                stock_returns=stock_returns if stock_returns else None,
                volatility=volatility if volatility else None,
                avg_mad=round(avg_mad, 4),
                trend_direction=self._calculate_trend(mad_values)
            )
            reports.append(report)

        return reports

    def get_top_suspicious_companies(self, n: int = 20) -> List[CompanyReport]:
        """Get top N most suspicious companies by average MAD."""
        return self.get_company_reports(top_n=n, sort_by='avg_mad')

    def get_companies_by_risk(self, risk_level: str) -> List[CompanyReport]:
        """Get all companies with a specific risk level."""
        return self.get_company_reports(filter_risk=risk_level)

    def get_yearly_statistics(self) -> pd.DataFrame:
        """
        Get yearly aggregate statistics.

        Returns:
            DataFrame with yearly statistics
        """
        stats = []
        for year in self.years:
            col = f'year_{year}_MAD'
            if col in self.benford_df.columns:
                valid_data = self.benford_df[col][self.benford_df[col] > 0]
                if not valid_data.empty:
                    stats.append({
                        'year': year,
                        'mean_MAD': valid_data.mean(),
                        'median_MAD': valid_data.median(),
                        'std_MAD': valid_data.std(),
                        'min_MAD': valid_data.min(),
                        'max_MAD': valid_data.max(),
                        'company_count': len(valid_data),
                        'pct_above_threshold': (valid_data > 2.5).sum() / len(valid_data) * 100
                    })

        return pd.DataFrame(stats)
