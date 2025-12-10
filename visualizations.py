"""
Benford's Law Analysis Visualizations
=====================================

This script generates 8 different visualizations for analyzing Benford's Law
conformance across ~500 public companies over 11 years (2014-2024).

Each visualization serves a specific analytical purpose - see VISUALIZATION_GUIDE.md
for detailed explanations.

Author: Generated for Diploma Thesis
Data Source: SEC EDGAR Financial Statements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory for graphs
OUTPUT_DIR = Path("graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_prepare_data(filepath="benford_500_companies_analysis.csv"):
    """Load the Benford analysis data and prepare it for visualization."""
    df = pd.read_csv(filepath)

    # Extract years from column names
    years = list(range(2014, 2025))

    # Create long-format dataframes for easier plotting
    mad_data = []
    chi_data = []
    pvalue_data = []
    ks_data = []

    for _, row in df.iterrows():
        cik = row['cik']
        company = row['company_name']

        for year in years:
            mad_col = f'year_{year}_MAD'
            chi_col = f'year_{year}_chi_square'
            pval_col = f'year_{year}_p_value'
            ks_col = f'year_{year}_KS_test'

            if mad_col in df.columns:
                mad_val = row[mad_col]
                chi_val = row[chi_col]
                pval_val = row[pval_col]
                ks_val = row[ks_col]

                # Skip zero/missing values
                if mad_val > 0:
                    mad_data.append({
                        'cik': cik,
                        'company': company,
                        'year': year,
                        'MAD': mad_val,
                        'chi_square': chi_val,
                        'p_value': pval_val,
                        'KS_test': ks_val
                    })

    df_long = pd.DataFrame(mad_data)

    # Calculate average metrics per company
    company_avg = df_long.groupby(['cik', 'company']).agg({
        'MAD': 'mean',
        'chi_square': 'mean',
        'p_value': 'mean',
        'KS_test': 'mean'
    }).reset_index()
    company_avg.columns = ['cik', 'company', 'avg_MAD', 'avg_chi_square', 'avg_p_value', 'avg_KS_test']

    return df, df_long, company_avg, years


def plot_1_heatmap(df, years):
    """
    GRAPH 1: HEATMAP
    ----------------
    Purpose: Provide a bird's-eye view of ALL companies across ALL years

    What it shows:
    - Each row = one company
    - Each column = one year (2014-2024)
    - Color intensity = MAD value (darker red = higher deviation from Benford's Law)

    How to interpret:
    - Green/light colors = Good conformance (MAD < 1.5)
    - Yellow = Moderate deviation (MAD 1.5-2.5)
    - Red/dark colors = High deviation (MAD > 2.5) - potential anomaly
    - White/blank = No data for that year

    Use case: Quickly identify which companies and years have anomalies
    """
    print("Creating Graph 1: Heatmap...")

    # Prepare matrix for heatmap
    companies = df['company_name'].tolist()

    # Create matrix
    matrix = np.zeros((len(companies), len(years)))

    for i, row in df.iterrows():
        for j, year in enumerate(years):
            col = f'year_{year}_MAD'
            if col in df.columns:
                val = row[col]
                matrix[i, j] = val if val > 0 else np.nan

    # Sort by average MAD for better visualization
    avg_mad = np.nanmean(matrix, axis=1)
    sort_idx = np.argsort(avg_mad)[::-1]  # Descending
    matrix_sorted = matrix[sort_idx]
    companies_sorted = [companies[i] for i in sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 20))

    # Create heatmap
    im = ax.imshow(matrix_sorted, aspect='auto', cmap='RdYlGn_r',
                   vmin=0, vmax=3.5)

    # Labels
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Companies (sorted by average MAD)', fontsize=12)
    ax.set_title('Benford\'s Law Conformance Heatmap\n(MAD Values: Green=Conforming, Red=Deviating)',
                 fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label('MAD Value', fontsize=11)

    # Remove y-tick labels (too many companies)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_heatmap_all_companies.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/1_heatmap_all_companies.png")


def plot_2_boxplots_by_year(df_long, years):
    """
    GRAPH 2: BOX PLOTS BY YEAR
    --------------------------
    Purpose: Show the DISTRIBUTION of MAD values across all companies for each year

    What it shows:
    - Each box = distribution of MAD values for one year
    - Box = 25th to 75th percentile (middle 50% of companies)
    - Line in box = median MAD
    - Whiskers = extend to 1.5x IQR
    - Dots = outliers (companies with extreme values)

    How to interpret:
    - Higher boxes = more companies deviating that year
    - Wider boxes = more variance in conformance
    - Many outliers = some companies with very high deviations
    - Trend over time: Is conformance improving or worsening?

    Use case: Identify if certain years had more anomalies (e.g., COVID-2020)
    """
    print("Creating Graph 2: Box Plots by Year...")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create box plot
    sns.boxplot(data=df_long, x='year', y='MAD', ax=ax, palette='viridis')

    # Add reference lines
    ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, label='Moderate threshold (1.5)')
    ax.axhline(y=2.5, color='red', linestyle='--', linewidth=2, label='High deviation threshold (2.5)')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('MAD Value', fontsize=12)
    ax.set_title('Distribution of Benford\'s Law Conformance by Year\n(Box Plot of MAD Values Across All Companies)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_boxplots_by_year.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/2_boxplots_by_year.png")


def plot_3_top_bottom_companies(company_avg):
    """
    GRAPH 3: TOP/BOTTOM PERFORMERS BAR CHART
    ----------------------------------------
    Purpose: Identify the BEST and WORST conforming companies

    What it shows:
    - Left panel: Top 20 companies with LOWEST average MAD (best conformance)
    - Right panel: Top 20 companies with HIGHEST average MAD (worst conformance)

    How to interpret:
    - Green bars (left) = Companies whose financials closely follow Benford's Law
    - Red bars (right) = Companies with consistent deviations - warrant investigation
    - Bar length = average MAD over all available years

    Use case: Identify specific companies for deeper analysis or investigation
    """
    print("Creating Graph 3: Top/Bottom Performers...")

    # Sort companies
    best = company_avg.nsmallest(20, 'avg_MAD')
    worst = company_avg.nlargest(20, 'avg_MAD')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Best conforming (green)
    colors_best = plt.cm.Greens(np.linspace(0.4, 0.8, len(best)))
    ax1.barh(range(len(best)), best['avg_MAD'].values, color=colors_best)
    ax1.set_yticks(range(len(best)))
    ax1.set_yticklabels([name[:40] for name in best['company'].values], fontsize=9)
    ax1.set_xlabel('Average MAD', fontsize=11)
    ax1.set_title('Top 20 Best Conforming Companies\n(Lowest Average MAD)',
                  fontsize=12, fontweight='bold', color='green')
    ax1.invert_yaxis()

    # Worst conforming (red)
    colors_worst = plt.cm.Reds(np.linspace(0.4, 0.8, len(worst)))
    ax2.barh(range(len(worst)), worst['avg_MAD'].values, color=colors_worst)
    ax2.set_yticks(range(len(worst)))
    ax2.set_yticklabels([name[:40] for name in worst['company'].values], fontsize=9)
    ax2.set_xlabel('Average MAD', fontsize=11)
    ax2.set_title('Top 20 Worst Conforming Companies\n(Highest Average MAD)',
                  fontsize=12, fontweight='bold', color='red')
    ax2.invert_yaxis()

    plt.suptitle('Company Rankings by Benford\'s Law Conformance', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_top_bottom_companies.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/3_top_bottom_companies.png")


def plot_4_time_series(df_long, company_avg):
    """
    GRAPH 4: TIME SERIES LINE CHARTS
    --------------------------------
    Purpose: Track INDIVIDUAL company conformance OVER TIME

    What it shows:
    - X-axis = Years (2014-2024)
    - Y-axis = MAD value
    - Each line = one company's MAD trajectory
    - Shows: 5 best + 5 worst + 5 most volatile companies

    How to interpret:
    - Flat lines = consistent behavior
    - Rising lines = deteriorating conformance (concerning)
    - Falling lines = improving conformance
    - Spiky lines = inconsistent reporting quality

    Use case: Deep dive into specific companies' reporting patterns
    """
    print("Creating Graph 4: Time Series...")

    # Select interesting companies
    best_5 = company_avg.nsmallest(5, 'avg_MAD')['company'].tolist()
    worst_5 = company_avg.nlargest(5, 'avg_MAD')['company'].tolist()

    # Calculate volatility (std of MAD over time)
    volatility = df_long.groupby('company')['MAD'].std().reset_index()
    volatility.columns = ['company', 'volatility']
    volatile_5 = volatility.nlargest(5, 'volatility')['company'].tolist()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Best companies
    for company in best_5:
        data = df_long[df_long['company'] == company].sort_values('year')
        axes[0].plot(data['year'], data['MAD'], marker='o', label=company[:30], linewidth=2)
    axes[0].set_ylabel('MAD', fontsize=11)
    axes[0].set_title('Best Conforming Companies (Lowest Avg MAD)', fontsize=12, fontweight='bold', color='green')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_ylim(0, 2)

    # Worst companies
    for company in worst_5:
        data = df_long[df_long['company'] == company].sort_values('year')
        axes[1].plot(data['year'], data['MAD'], marker='o', label=company[:30], linewidth=2)
    axes[1].set_ylabel('MAD', fontsize=11)
    axes[1].set_title('Worst Conforming Companies (Highest Avg MAD)', fontsize=12, fontweight='bold', color='red')
    axes[1].legend(loc='upper right', fontsize=8)

    # Most volatile companies
    for company in volatile_5:
        data = df_long[df_long['company'] == company].sort_values('year')
        axes[2].plot(data['year'], data['MAD'], marker='o', label=company[:30], linewidth=2)
    axes[2].set_xlabel('Year', fontsize=11)
    axes[2].set_ylabel('MAD', fontsize=11)
    axes[2].set_title('Most Volatile Companies (Highest Std Dev in MAD)', fontsize=12, fontweight='bold', color='orange')
    axes[2].legend(loc='upper right', fontsize=8)

    plt.suptitle('Time Series Analysis of Benford\'s Law Conformance', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '4_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/4_time_series.png")


def plot_5_histogram(df_long):
    """
    GRAPH 5: HISTOGRAM / DISTRIBUTION CHART
    ---------------------------------------
    Purpose: Show the OVERALL distribution of MAD values across all data points

    What it shows:
    - X-axis = MAD value ranges (bins)
    - Y-axis = Count of company-year observations
    - Vertical lines = threshold markers

    How to interpret:
    - Peak location = most common MAD value
    - Right tail = anomalous observations
    - Green zone (MAD < 1.5) = Conforming
    - Yellow zone (1.5-2.5) = Marginal
    - Red zone (> 2.5) = Non-conforming

    Use case: Understand overall conformance distribution across the dataset
    """
    print("Creating Graph 5: Histogram...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create histogram
    n, bins, patches = ax.hist(df_long['MAD'], bins=50, edgecolor='black', alpha=0.7)

    # Color bars by conformance level
    for i, patch in enumerate(patches):
        if bins[i] < 1.5:
            patch.set_facecolor('green')
        elif bins[i] < 2.5:
            patch.set_facecolor('orange')
        else:
            patch.set_facecolor('red')

    # Add threshold lines
    ax.axvline(x=1.5, color='orange', linestyle='--', linewidth=2, label='Moderate threshold (1.5)')
    ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2, label='High deviation threshold (2.5)')

    ax.set_xlabel('MAD Value', fontsize=12)
    ax.set_ylabel('Count (Company-Year Observations)', fontsize=12)
    ax.set_title('Distribution of MAD Values Across All Companies and Years\n(Green=Conforming, Orange=Marginal, Red=Non-conforming)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    # Add statistics annotation
    stats_text = f'Total observations: {len(df_long)}\n'
    stats_text += f'Mean MAD: {df_long["MAD"].mean():.3f}\n'
    stats_text += f'Median MAD: {df_long["MAD"].median():.3f}\n'
    stats_text += f'Std Dev: {df_long["MAD"].std():.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '5_histogram_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/5_histogram_distribution.png")


def plot_6_scatter(df_long):
    """
    GRAPH 6: SCATTER PLOT (Chi-Square vs MAD)
    -----------------------------------------
    Purpose: Compare TWO METRICS to validate they correlate and find outliers

    What it shows:
    - X-axis = Chi-square statistic
    - Y-axis = MAD value
    - Each point = one company-year observation
    - Color = p-value (statistical significance)

    How to interpret:
    - Points should generally follow a diagonal trend (metrics correlate)
    - Outliers = points far from the main cluster
    - Dark points (low p-value) = statistically significant deviations
    - Light points (high p-value) = deviations could be due to chance

    Use case: Validate that different metrics agree, identify unusual cases
    """
    print("Creating Graph 6: Scatter Plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Sample data if too large
    plot_data = df_long.sample(min(2000, len(df_long)), random_state=42)

    scatter = ax.scatter(plot_data['chi_square'], plot_data['MAD'],
                        c=plot_data['p_value'], cmap='viridis_r',
                        alpha=0.6, s=30)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('P-value (lower = more significant)', fontsize=11)

    ax.set_xlabel('Chi-Square Statistic', fontsize=12)
    ax.set_ylabel('MAD Value', fontsize=12)
    ax.set_title('Correlation Between Chi-Square and MAD Metrics\n(Color = P-value: Dark = Significant Deviation)',
                 fontsize=14, fontweight='bold')

    # Add correlation coefficient
    corr = df_long['chi_square'].corr(df_long['MAD'])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '6_scatter_chi_vs_mad.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/6_scatter_chi_vs_mad.png")


def plot_7_trend_analysis(df_long, years):
    """
    GRAPH 7: TREND ANALYSIS CHART
    -----------------------------
    Purpose: Show AGGREGATE conformance trends over time across ALL companies

    What it shows:
    - X-axis = Years (2014-2024)
    - Y-axis = Average MAD value
    - Line = mean MAD per year across all companies
    - Shaded area = standard deviation band (variability)

    How to interpret:
    - Rising trend = Financial reporting quality degrading over time
    - Falling trend = Improving conformance (better quality)
    - Wider bands = more variance between companies
    - Spikes = unusual years (e.g., COVID impact in 2020)

    Use case: Understand macro trends in financial reporting quality
    """
    print("Creating Graph 7: Trend Analysis...")

    # Calculate yearly statistics
    yearly_stats = df_long.groupby('year').agg({
        'MAD': ['mean', 'std', 'median', 'count'],
        'chi_square': 'mean'
    }).reset_index()
    yearly_stats.columns = ['year', 'mean_MAD', 'std_MAD', 'median_MAD', 'count', 'mean_chi']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot mean with std band
    ax.plot(yearly_stats['year'], yearly_stats['mean_MAD'],
            marker='o', linewidth=2, color='blue', label='Mean MAD')
    ax.fill_between(yearly_stats['year'],
                    yearly_stats['mean_MAD'] - yearly_stats['std_MAD'],
                    yearly_stats['mean_MAD'] + yearly_stats['std_MAD'],
                    alpha=0.3, color='blue', label='Standard Deviation')

    # Plot median
    ax.plot(yearly_stats['year'], yearly_stats['median_MAD'],
            marker='s', linewidth=2, color='green', linestyle='--', label='Median MAD')

    # Add sample size annotation
    for i, row in yearly_stats.iterrows():
        ax.annotate(f'n={int(row["count"])}',
                   (row['year'], row['mean_MAD'] + row['std_MAD'] + 0.1),
                   ha='center', fontsize=8, alpha=0.7)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('MAD Value', fontsize=12)
    ax.set_title('Aggregate Trend of Benford\'s Law Conformance Over Time\n(All Companies Combined)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xticks(years)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '7_trend_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/7_trend_analysis.png")


def plot_8_pvalue_significance(df_long, years):
    """
    GRAPH 8: P-VALUE SIGNIFICANCE CHART
    -----------------------------------
    Purpose: Show what PERCENTAGE of companies have SIGNIFICANT deviations per year

    What it shows:
    - X-axis = Years (2014-2024)
    - Y-axis = Percentage of companies
    - Bars = % with significant deviation (p < 0.05)

    How to interpret:
    - Higher bars = more companies with statistically significant anomalies
    - If >50% are significant = widespread deviation from Benford's Law
    - Trend over years = is anomaly prevalence increasing?

    Use case: Quantify how prevalent significant deviations are
    """
    print("Creating Graph 8: P-value Significance...")

    # Calculate percentage significant per year
    significance = df_long.groupby('year').apply(
        lambda x: (x['p_value'] < 0.05).sum() / len(x) * 100
    ).reset_index()
    significance.columns = ['year', 'pct_significant']

    # Also calculate different thresholds
    sig_01 = df_long.groupby('year').apply(
        lambda x: (x['p_value'] < 0.01).sum() / len(x) * 100
    ).reset_index()
    sig_01.columns = ['year', 'pct_sig_01']

    significance = significance.merge(sig_01, on='year')

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(significance))
    width = 0.35

    bars1 = ax.bar(x - width/2, significance['pct_significant'], width,
                   label='p < 0.05 (Significant)', color='orange', alpha=0.8)
    bars2 = ax.bar(x + width/2, significance['pct_sig_01'], width,
                   label='p < 0.01 (Highly Significant)', color='red', alpha=0.8)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Percentage of Companies (%)', fontsize=12)
    ax.set_title('Percentage of Companies with Statistically Significant\nDeviations from Benford\'s Law',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(significance['year'].values)
    ax.legend(loc='upper right')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(0.02, 51, 'Majority threshold (50%)', fontsize=9, alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '8_pvalue_significance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: graphs/8_pvalue_significance.png")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("BENFORD'S LAW ANALYSIS - VISUALIZATION GENERATOR")
    print("=" * 60)
    print()

    # Load data
    print("Loading data...")
    df, df_long, company_avg, years = load_and_prepare_data()
    print(f"  Loaded {len(df)} companies")
    print(f"  {len(df_long)} total company-year observations")
    print()

    # Generate all graphs
    print("Generating visualizations...")
    print("-" * 40)

    plot_1_heatmap(df, years)
    plot_2_boxplots_by_year(df_long, years)
    plot_3_top_bottom_companies(company_avg)
    plot_4_time_series(df_long, company_avg)
    plot_5_histogram(df_long)
    plot_6_scatter(df_long)
    plot_7_trend_analysis(df_long, years)
    plot_8_pvalue_significance(df_long, years)

    print("-" * 40)
    print()
    print("ALL VISUALIZATIONS COMPLETE!")
    print(f"Graphs saved to: {OUTPUT_DIR.absolute()}")
    print()
    print("See VISUALIZATION_GUIDE.md for detailed explanations of each graph.")
    print("=" * 60)


if __name__ == "__main__":
    main()
