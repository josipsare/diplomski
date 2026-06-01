"""
Sector/Industry Classification and Comparison Module

Provides SIC-based sector classification and peer comparison for Benford analysis.
Allows comparing companies against their industry peers rather than just Benford's
expected distribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from scipy.stats import norm


# SIC Division Ranges (SEC uses numeric SIC codes)
# Based on US Department of Labor Standard Industrial Classification
SIC_DIVISIONS = {
    (100, 999): {'code': 'A', 'name': 'Agriculture, Forestry & Fishing'},
    (1000, 1499): {'code': 'B', 'name': 'Mining'},
    (1500, 1799): {'code': 'C', 'name': 'Construction'},
    (2000, 3999): {'code': 'D', 'name': 'Manufacturing'},
    (4000, 4999): {'code': 'E', 'name': 'Transportation & Utilities'},
    (5000, 5199): {'code': 'F', 'name': 'Wholesale Trade'},
    (5200, 5999): {'code': 'G', 'name': 'Retail Trade'},
    (6000, 6799): {'code': 'H', 'name': 'Finance, Insurance & Real Estate'},
    (7000, 8999): {'code': 'I', 'name': 'Services'},
    (9100, 9999): {'code': 'J', 'name': 'Public Administration'},
}

# Common SIC Major Groups (2-digit) with names
SIC_MAJOR_GROUPS = {
    '01': 'Agricultural Production - Crops',
    '02': 'Agricultural Production - Livestock',
    '07': 'Agricultural Services',
    '08': 'Forestry',
    '09': 'Fishing, Hunting & Trapping',
    '10': 'Metal Mining',
    '12': 'Coal Mining',
    '13': 'Oil & Gas Extraction',
    '14': 'Mining of Nonmetallic Minerals',
    '15': 'Building Construction',
    '16': 'Heavy Construction',
    '17': 'Construction Special Trade',
    '20': 'Food & Kindred Products',
    '21': 'Tobacco Products',
    '22': 'Textile Mill Products',
    '23': 'Apparel & Other Finished Products',
    '24': 'Lumber & Wood Products',
    '25': 'Furniture & Fixtures',
    '26': 'Paper & Allied Products',
    '27': 'Printing, Publishing & Allied',
    '28': 'Chemicals & Allied Products',
    '29': 'Petroleum Refining & Related',
    '30': 'Rubber & Miscellaneous Plastics',
    '31': 'Leather & Leather Products',
    '32': 'Stone, Clay, Glass & Concrete',
    '33': 'Primary Metal Industries',
    '34': 'Fabricated Metal Products',
    '35': 'Industrial Machinery & Equipment',
    '36': 'Electronic & Electrical Equipment',
    '37': 'Transportation Equipment',
    '38': 'Measuring & Analyzing Instruments',
    '39': 'Miscellaneous Manufacturing',
    '40': 'Railroad Transportation',
    '41': 'Local & Suburban Transit',
    '42': 'Motor Freight Transportation',
    '43': 'United States Postal Service',
    '44': 'Water Transportation',
    '45': 'Transportation by Air',
    '46': 'Pipelines',
    '47': 'Transportation Services',
    '48': 'Communications',
    '49': 'Electric, Gas & Sanitary Services',
    '50': 'Wholesale Trade - Durable Goods',
    '51': 'Wholesale Trade - Nondurable Goods',
    '52': 'Building Materials & Hardware',
    '53': 'General Merchandise Stores',
    '54': 'Food Stores',
    '55': 'Automotive Dealers & Gas Stations',
    '56': 'Apparel & Accessory Stores',
    '57': 'Home Furniture & Equipment Stores',
    '58': 'Eating & Drinking Places',
    '59': 'Miscellaneous Retail',
    '60': 'Depository Institutions (Banks)',
    '61': 'Non-Depository Credit Institutions',
    '62': 'Security & Commodity Brokers',
    '63': 'Insurance Carriers',
    '64': 'Insurance Agents & Brokers',
    '65': 'Real Estate',
    '67': 'Holding & Investment Offices',
    '70': 'Hotels & Lodging Places',
    '72': 'Personal Services',
    '73': 'Business Services',
    '75': 'Automotive Repair & Services',
    '76': 'Miscellaneous Repair Services',
    '78': 'Motion Pictures',
    '79': 'Amusement & Recreation Services',
    '80': 'Health Services',
    '81': 'Legal Services',
    '82': 'Educational Services',
    '83': 'Social Services',
    '84': 'Museums & Botanical Gardens',
    '86': 'Membership Organizations',
    '87': 'Engineering & Management Services',
    '88': 'Private Households',
    '89': 'Miscellaneous Services',
    '91': 'Executive, Legislative & General',
    '92': 'Justice, Public Order & Safety',
    '93': 'Public Finance & Taxation',
    '94': 'Administration of Human Resources',
    '95': 'Administration of Environmental Quality',
    '96': 'Administration of Economic Programs',
    '97': 'National Security & International',
    '99': 'Nonclassifiable Establishments',
}


def get_sic_division(sic_code: Union[str, int]) -> Tuple[str, str]:
    """
    Get division code and name for a SIC code.

    Args:
        sic_code: 4-digit SIC code

    Returns:
        Tuple of (division_code, division_name)
    """
    try:
        sic_int = int(str(sic_code).strip())
    except (ValueError, TypeError):
        return ('X', 'Unknown')

    for (low, high), info in SIC_DIVISIONS.items():
        if low <= sic_int <= high:
            return (info['code'], info['name'])

    return ('X', 'Unknown')


def get_sic_major_group(sic_code: Union[str, int]) -> Tuple[str, str]:
    """
    Get 2-digit major group code and name for a SIC code.

    Args:
        sic_code: 4-digit SIC code

    Returns:
        Tuple of (major_group_code, major_group_name)
    """
    try:
        sic_str = str(int(str(sic_code).strip())).zfill(4)
        major_group = sic_str[:2]
    except (ValueError, TypeError):
        return ('XX', 'Unknown')

    name = SIC_MAJOR_GROUPS.get(major_group, f'Major Group {major_group}')
    return (major_group, name)


def get_sic_industry_group(sic_code: Union[str, int]) -> str:
    """
    Get 3-digit industry group code.

    Args:
        sic_code: 4-digit SIC code

    Returns:
        3-digit industry group code
    """
    try:
        sic_str = str(int(str(sic_code).strip())).zfill(4)
        return sic_str[:3]
    except (ValueError, TypeError):
        return 'XXX'


def classify_company(cik: str, company_name: str, sic_code: Union[str, int]) -> Dict:
    """
    Classify a company by SIC code at all hierarchy levels.

    Args:
        cik: Company CIK
        company_name: Company name
        sic_code: 4-digit SIC code

    Returns:
        Dictionary with complete sector classification
    """
    division_code, division_name = get_sic_division(sic_code)
    major_group, major_group_name = get_sic_major_group(sic_code)
    industry_group = get_sic_industry_group(sic_code)

    return {
        'cik': str(cik).zfill(10) if cik else '',
        'company_name': company_name,
        'sic_code': str(sic_code).zfill(4) if sic_code else 'XXXX',
        'sic_division': division_code,
        'division_name': division_name,
        'sic_major_group': major_group,
        'major_group_name': major_group_name,
        'sic_industry_group': industry_group,
    }


def calculate_sector_baselines(
    analysis_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    years: List[int],
    group_by: str = 'sic_major_group',
    min_companies: int = 3
) -> pd.DataFrame:
    """
    Calculate Benford baseline metrics for each sector.

    Args:
        analysis_df: DataFrame with company Benford metrics (from batch_analysis)
        sector_df: DataFrame with company SIC classifications
        years: List of years to analyze
        group_by: Grouping level ('sic_division', 'sic_major_group', 'sic_industry_group')
        min_companies: Minimum companies required to calculate baseline

    Returns:
        DataFrame with sector baselines for each year
    """
    # Ensure CIK columns are strings for merging
    analysis_df = analysis_df.copy()
    sector_df = sector_df.copy()

    analysis_df['cik'] = analysis_df['cik'].astype(str).str.zfill(10)
    sector_df['cik'] = sector_df['cik'].astype(str).str.zfill(10)

    # Merge analysis data with sector classifications
    merged = analysis_df.merge(
        sector_df[['cik', group_by, 'division_name', 'major_group_name']],
        on='cik',
        how='left'
    )

    baselines = []

    for year in years:
        mad_col = f'year_{year}_MAD'
        chi_col = f'year_{year}_chi_square'
        pval_col = f'year_{year}_p_value'
        ks_col = f'year_{year}_KS_test'

        if mad_col not in merged.columns:
            continue

        # Filter valid data (MAD > 0 means we have data for that year)
        year_data = merged[merged[mad_col] > 0].copy()

        # Group by sector
        for sector_code, group in year_data.groupby(group_by):
            if pd.isna(sector_code) or len(group) < min_companies:
                continue

            mad_values = group[mad_col].values

            # Get sector name
            if group_by == 'sic_major_group' and 'major_group_name' in group.columns:
                sector_name = group['major_group_name'].iloc[0]
            elif 'division_name' in group.columns:
                sector_name = group['division_name'].iloc[0]
            else:
                sector_name = str(sector_code)

            baseline = {
                'sic_level': group_by,
                'sic_code': sector_code,
                'sector_name': sector_name,
                'year': year,
                'company_count': len(group),

                # MAD statistics
                'mean_MAD': np.mean(mad_values),
                'median_MAD': np.median(mad_values),
                'std_MAD': np.std(mad_values) if len(mad_values) > 1 else 0,
                'min_MAD': np.min(mad_values),
                'max_MAD': np.max(mad_values),
                'percentile_25': np.percentile(mad_values, 25),
                'percentile_75': np.percentile(mad_values, 75),
                'percentile_90': np.percentile(mad_values, 90),
                'percentile_95': np.percentile(mad_values, 95),

                # Other metrics
                'mean_chi_square': group[chi_col].mean() if chi_col in group.columns else 0,
                'mean_p_value': group[pval_col].mean() if pval_col in group.columns else 0,
                'mean_KS_test': group[ks_col].mean() if ks_col in group.columns else 0,
            }
            baselines.append(baseline)

    return pd.DataFrame(baselines)


def compare_company_to_sector(
    company_metrics: Dict,
    sector_baseline: pd.Series,
    year: int
) -> Optional[Dict]:
    """
    Compare a single company's metrics to its sector baseline.

    Args:
        company_metrics: Dictionary with company's Benford metrics
        sector_baseline: Series with sector baseline for the year
        year: Year being compared

    Returns:
        Dictionary with comparison results, or None if insufficient data
    """
    company_mad = company_metrics.get(f'year_{year}_MAD', 0)

    if company_mad == 0 or pd.isna(sector_baseline['std_MAD']) or sector_baseline['std_MAD'] == 0:
        return None

    # Calculate z-score relative to sector
    z_score = (company_mad - sector_baseline['mean_MAD']) / sector_baseline['std_MAD']

    # Calculate percentile (using normal distribution approximation)
    percentile = norm.cdf(z_score) * 100

    return {
        'cik': str(company_metrics.get('cik', '')).zfill(10),
        'company_name': company_metrics.get('company_name', ''),
        'year': year,
        'sic_code': company_metrics.get('sic_code', 'XXXX'),
        'sector_name': sector_baseline['sector_name'],

        # Raw values
        'company_MAD': round(company_mad, 4),
        'sector_median_MAD': round(sector_baseline['median_MAD'], 4),
        'sector_mean_MAD': round(sector_baseline['mean_MAD'], 4),
        'sector_std_MAD': round(sector_baseline['std_MAD'], 4),

        # Deviation scores
        'z_score_vs_sector': round(z_score, 4),
        'percentile_in_sector': round(percentile, 2),
        'deviation_from_median': round(company_mad - sector_baseline['median_MAD'], 4),

        # Flags
        'is_outlier': abs(z_score) > 2,
        'is_extreme_outlier': abs(z_score) > 3,
        'sector_company_count': int(sector_baseline['company_count']),
    }


def run_sector_comparison(
    analysis_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    years: List[int],
    group_by: str = 'sic_major_group',
    min_companies: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete sector comparison analysis.

    Args:
        analysis_df: DataFrame with Benford analysis results
        sector_df: DataFrame with sector classifications
        years: Years to analyze
        group_by: Grouping level
        min_companies: Minimum companies per sector

    Returns:
        Tuple of (sector_baselines_df, company_comparisons_df)
    """
    # Calculate sector baselines
    baselines_df = calculate_sector_baselines(
        analysis_df, sector_df, years, group_by, min_companies
    )

    if baselines_df.empty:
        return baselines_df, pd.DataFrame()

    # Prepare for comparison
    analysis_df = analysis_df.copy()
    sector_df = sector_df.copy()

    analysis_df['cik'] = analysis_df['cik'].astype(str).str.zfill(10)
    sector_df['cik'] = sector_df['cik'].astype(str).str.zfill(10)

    # Merge analysis with sector info
    merged = analysis_df.merge(
        sector_df[['cik', 'sic_code', group_by, 'major_group_name']],
        on='cik',
        how='left'
    )

    # Compare each company to their sector
    comparisons = []

    for _, company in merged.iterrows():
        company_dict = company.to_dict()
        sector_code = company.get(group_by)

        if pd.isna(sector_code):
            continue

        for year in years:
            # Find sector baseline for this year
            baseline = baselines_df[
                (baselines_df['sic_code'] == sector_code) &
                (baselines_df['year'] == year)
            ]

            if baseline.empty:
                continue

            comparison = compare_company_to_sector(
                company_dict,
                baseline.iloc[0],
                year
            )
            if comparison:
                comparisons.append(comparison)

    comparisons_df = pd.DataFrame(comparisons)

    # Calculate rank within sector for each year
    if not comparisons_df.empty:
        comparisons_df['sector_rank'] = comparisons_df.groupby(
            ['sector_name', 'year']
        )['company_MAD'].rank(method='min').astype(int)

    return baselines_df, comparisons_df


def identify_sector_outliers(
    comparisons_df: pd.DataFrame,
    z_threshold: float = 2.0
) -> pd.DataFrame:
    """
    Identify companies that deviate significantly from their sector.

    Args:
        comparisons_df: DataFrame from run_sector_comparison
        z_threshold: Z-score threshold for flagging (default 2.0)

    Returns:
        DataFrame with outlier companies
    """
    if comparisons_df.empty:
        return pd.DataFrame()

    outliers = comparisons_df[
        abs(comparisons_df['z_score_vs_sector']) > z_threshold
    ].copy()

    outliers['deviation_direction'] = outliers['z_score_vs_sector'].apply(
        lambda z: 'worse_than_peers' if z > 0 else 'better_than_peers'
    )

    return outliers.sort_values('z_score_vs_sector', ascending=False)


def get_sector_summary(baselines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics across all sectors.

    Args:
        baselines_df: DataFrame from calculate_sector_baselines

    Returns:
        DataFrame with sector summary statistics
    """
    if baselines_df.empty:
        return pd.DataFrame()

    summary = baselines_df.groupby('sector_name').agg({
        'median_MAD': ['mean', 'std', 'min', 'max'],
        'company_count': ['mean', 'sum'],
        'year': 'nunique'
    }).reset_index()

    summary.columns = [
        'sector_name',
        'avg_median_MAD', 'std_median_MAD', 'min_median_MAD', 'max_median_MAD',
        'avg_companies', 'total_observations',
        'years_covered'
    ]

    return summary.sort_values('avg_median_MAD', ascending=False)


if __name__ == '__main__':
    # Example usage
    print("Sector Classification Module")
    print("=" * 60)

    # Test SIC code classification
    test_sics = [
        ('0000320193', 'Apple Inc.', '3571'),
        ('0000789019', 'Microsoft Corp', '7372'),
        ('0001318605', 'Tesla Inc', '3711'),
        ('0001018724', 'Amazon.com Inc', '5961'),
        ('0000051143', 'IBM Corp', '7370'),
    ]

    print("\nSIC Classification Examples:")
    print("-" * 60)

    for cik, name, sic in test_sics:
        classification = classify_company(cik, name, sic)
        print(f"\n{name}:")
        print(f"  SIC Code: {classification['sic_code']}")
        print(f"  Division: {classification['sic_division']} - {classification['division_name']}")
        print(f"  Major Group: {classification['sic_major_group']} - {classification['major_group_name']}")
        print(f"  Industry Group: {classification['sic_industry_group']}")
