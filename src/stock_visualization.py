"""
Stock Analysis Visualizations

Creates visualizations combining Benford's Law analysis with stock performance.
Includes the Benford Arrow Plot and other correlation visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_benford_arrow_plot(
    data: pd.DataFrame,
    x_col: str = 'MAD',
    y_col: str = 'stock_annual_return',
    output_path: Optional[str] = None,
    title: str = "Benford's Law vs Stock Performance",
    year: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create the Benford Arrow Plot - primary visualization.

    Shows each company as a point with an arrow indicating next year's
    stock price direction. Companies with high Benford MAD (suspicious)
    are on the right side of the plot.

    Args:
        data: DataFrame with columns: symbol, year, x (benford metric),
              y (stock return), next_year_up (bool), next_year_return
        x_col: Column name for X-axis (Benford metric)
        y_col: Column name for Y-axis (stock return)
        output_path: If provided, save figure to this path
        title: Plot title
        year: If provided, filter to specific year
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    df = data.copy()

    # Filter by year if specified
    if year is not None:
        df = df[df['year'] == year]
        title = f"{title} ({year})"

    if len(df) == 0:
        print("No data to plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Separate up and down arrows
    up_mask = df['next_year_up'] == True
    down_mask = df['next_year_up'] == False

    # Plot points with arrows
    # UP arrows (green)
    if up_mask.any():
        up_data = df[up_mask]
        ax.scatter(
            up_data['x'], up_data['y'],
            c='green', alpha=0.6, s=50, label='Next year UP',
            marker='^', edgecolors='darkgreen', linewidths=0.5
        )

        # Draw arrows for significant moves
        for _, row in up_data.iterrows():
            arrow_len = min(row['arrow_magnitude'] / 10, 5)  # Scale arrow length
            ax.annotate(
                '', xy=(row['x'], row['y'] + arrow_len),
                xytext=(row['x'], row['y']),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.4, lw=1)
            )

    # DOWN arrows (red)
    if down_mask.any():
        down_data = df[down_mask]
        ax.scatter(
            down_data['x'], down_data['y'],
            c='red', alpha=0.6, s=50, label='Next year DOWN',
            marker='v', edgecolors='darkred', linewidths=0.5
        )

        # Draw arrows for significant moves
        for _, row in down_data.iterrows():
            arrow_len = min(row['arrow_magnitude'] / 10, 5)
            ax.annotate(
                '', xy=(row['x'], row['y'] - arrow_len),
                xytext=(row['x'], row['y']),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.4, lw=1)
            )

    # Add vertical lines for MAD thresholds
    ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.7, label='MAD = 1.5 (threshold)')
    ax.axvline(x=2.5, color='red', linestyle='--', alpha=0.7, label='MAD = 2.5 (concerning)')

    # Add horizontal line at 0% return
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Add regions with shading
    ax.axvspan(2.5, ax.get_xlim()[1] if ax.get_xlim()[1] > 2.5 else 5,
               alpha=0.1, color='red', label='Suspicious zone')

    # Labels and title
    ax.set_xlabel(f'Benford {x_col} (higher = more suspicious)', fontsize=12)
    ax.set_ylabel(f'Stock Return % (Year N)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    # Add annotation explaining the plot
    textstr = 'Hypothesis: Companies on the right\n(high MAD) should have more\nred/down arrows'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Statistics
    if len(df) > 0:
        high_mad = df[df['x'] > 2.0]
        low_mad = df[df['x'] <= 1.5]

        high_mad_down_pct = (high_mad['next_year_up'] == False).mean() * 100 if len(high_mad) > 0 else 0
        low_mad_down_pct = (low_mad['next_year_up'] == False).mean() * 100 if len(low_mad) > 0 else 0

        stats_text = f'High MAD (>2.0): {high_mad_down_pct:.1f}% down next year (n={len(high_mad)})\n'
        stats_text += f'Low MAD (≤1.5): {low_mad_down_pct:.1f}% down next year (n={len(low_mad)})'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_arrow_plot_by_year(
    data: pd.DataFrame,
    years: List[int],
    x_col: str = 'MAD',
    y_col: str = 'stock_annual_return',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12)
) -> plt.Figure:
    """
    Create multi-panel arrow plot showing different years.

    Args:
        data: DataFrame with arrow plot data
        years: List of years to show (e.g., [2016, 2018, 2020, 2022])
        x_col: Column for X-axis
        y_col: Column for Y-axis
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_years = len(years)
    n_cols = min(3, n_years)
    n_rows = (n_years + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, year in enumerate(years):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        year_data = data[data['year'] == year]

        if len(year_data) == 0:
            ax.text(0.5, 0.5, f'No data for {year}', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f'{year}')
            continue

        # Plot UP (green triangles)
        up_data = year_data[year_data['next_year_up'] == True]
        if len(up_data) > 0:
            ax.scatter(up_data['x'], up_data['y'], c='green', alpha=0.6,
                      s=30, marker='^', label='Up')

        # Plot DOWN (red triangles)
        down_data = year_data[year_data['next_year_up'] == False]
        if len(down_data) > 0:
            ax.scatter(down_data['x'], down_data['y'], c='red', alpha=0.6,
                      s=30, marker='v', label='Down')

        # Threshold lines
        ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(x=2.5, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Title with stats
        high_mad = year_data[year_data['x'] > 2.0]
        pct_down = (high_mad['next_year_up'] == False).mean() * 100 if len(high_mad) > 0 else 0
        ax.set_title(f'{year} (High MAD: {pct_down:.0f}% down)', fontsize=11)

        ax.set_xlabel('MAD', fontsize=9)
        ax.set_ylabel('Return %', fontsize=9)

    # Remove empty subplots
    for idx in range(len(years), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    # Add legend to first plot
    axes[0, 0].legend(loc='upper right', fontsize=8)

    fig.suptitle("Benford Arrow Plot by Year\n(High MAD companies should have more red/down arrows)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_volume_benford_heatmap(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    top_n: int = 50,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create heatmap of volume Benford MAD across companies and years.

    Args:
        data: DataFrame with volume_benford_MAD columns
        output_path: Path to save figure
        top_n: Number of companies to show
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Extract volume MAD columns
    volume_cols = [c for c in data.columns if 'volume_benford_MAD' in c]
    years = sorted([int(c.split('_')[1]) for c in volume_cols])

    # Create pivot data
    pivot_data = data[['symbol'] + volume_cols].copy()

    # Calculate average MAD per company
    pivot_data['avg_MAD'] = pivot_data[volume_cols].mean(axis=1)
    pivot_data = pivot_data.nlargest(top_n, 'avg_MAD')

    # Prepare heatmap matrix
    heatmap_data = pivot_data.set_index('symbol')[volume_cols]
    heatmap_data.columns = years

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap='RdYlGn_r',  # Red = high MAD (bad), Green = low MAD (good)
        center=1.5,
        annot=True,
        fmt='.1f',
        ax=ax,
        cbar_kws={'label': 'Volume Benford MAD'}
    )

    ax.set_title(f'Trading Volume Benford Analysis (Top {top_n} by Avg MAD)\n'
                 f'Red = Unusual volume patterns, Green = Normal',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Company', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_combined_time_series(
    data: pd.DataFrame,
    symbols: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Plot combined time series of Benford MAD and stock price.

    Args:
        data: Long format DataFrame with year, MAD, stock metrics
        symbols: List of symbols to plot
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    n_symbols = len(symbols)
    n_cols = min(2, n_symbols)
    n_rows = (n_symbols + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, symbol in enumerate(symbols):
        row = idx // n_cols
        col = idx % n_cols
        ax1 = axes[row, col]

        symbol_data = data[data['symbol'] == symbol].sort_values('year')

        if len(symbol_data) == 0:
            ax1.text(0.5, 0.5, f'No data for {symbol}', ha='center', va='center',
                    transform=ax1.transAxes)
            continue

        years = symbol_data['year'].values

        # Plot MAD on left axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Benford MAD', color=color1)
        ax1.plot(years, symbol_data['MAD'], color=color1, marker='o', label='MAD')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5)
        ax1.axhline(y=2.5, color='red', linestyle='--', alpha=0.5)

        # Plot stock return on right axis
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Stock Return %', color=color2)
        ax2.bar(years, symbol_data['stock_annual_return'], alpha=0.3, color=color2, label='Return')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        ax1.set_title(f'{symbol}', fontsize=11, fontweight='bold')

    # Remove empty subplots
    for idx in range(len(symbols), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    fig.suptitle('Benford MAD (blue) vs Stock Returns (green bars) Over Time',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_correlation_summary(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create correlation summary visualization.

    Shows correlations between Benford metrics and stock metrics.

    Args:
        data: Long format DataFrame with all metrics
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    # Select numeric columns for correlation
    benford_cols = ['MAD', 'chi_square', 'p_value', 'KS_test']
    stock_cols = ['stock_annual_return', 'stock_volatility', 'stock_max_drawdown',
                  'stock_sharpe_ratio', 'stock_volume_benford_MAD']

    # Filter to existing columns
    benford_cols = [c for c in benford_cols if c in data.columns]
    stock_cols = [c for c in stock_cols if c in data.columns]

    if not benford_cols or not stock_cols:
        print("Not enough columns for correlation")
        return None

    # Calculate correlations
    all_cols = benford_cols + stock_cols
    corr_matrix = data[all_cols].corr()

    # Extract cross-correlations (Benford vs Stock)
    cross_corr = corr_matrix.loc[benford_cols, stock_cols]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Correlation heatmap
    sns.heatmap(
        cross_corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        ax=axes[0],
        vmin=-0.5,
        vmax=0.5
    )
    axes[0].set_title('Benford vs Stock Metrics Correlation', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Stock Metrics')
    axes[0].set_ylabel('Benford Metrics')

    # Plot 2: Summary statistics
    ax2 = axes[1]

    # Calculate percentages
    high_mad = data[data['MAD'] > 2.0] if 'MAD' in data.columns else pd.DataFrame()
    low_mad = data[data['MAD'] <= 1.5] if 'MAD' in data.columns else pd.DataFrame()

    categories = ['High MAD (>2.0)', 'Low MAD (≤1.5)']
    if 'stock_annual_return' in data.columns:
        high_pct_neg = (high_mad['stock_annual_return'] < 0).mean() * 100 if len(high_mad) > 0 else 0
        low_pct_neg = (low_mad['stock_annual_return'] < 0).mean() * 100 if len(low_mad) > 0 else 0
    else:
        high_pct_neg = 0
        low_pct_neg = 0

    values = [high_pct_neg, low_pct_neg]
    colors = ['red', 'green']

    bars = ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_ylabel('% with Negative Returns')
    ax2.set_title('Companies with Negative Stock Returns\nby Benford MAD Category',
                  fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)

    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', fontsize=11)

    # Add sample sizes
    ax2.text(0.02, 0.98, f'High MAD: n={len(high_mad)}\nLow MAD: n={len(low_mad)}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def generate_all_stock_visualizations(
    combined_data: pd.DataFrame,
    output_dir: str,
    years: List[int] = None
):
    """
    Generate all 5 core stock visualizations.

    Args:
        combined_data: Merged Benford + stock data in long format
        output_dir: Directory to save visualizations
        years: Years to include (default: 2016, 2018, 2020, 2022)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if years is None:
        years = [2016, 2018, 2020, 2022]

    print("\n" + "="*60)
    print("Generating Stock Analysis Visualizations")
    print("="*60 + "\n")

    # Prepare arrow plot data
    from .stock_metrics import get_arrow_plot_data
    arrow_data = get_arrow_plot_data(combined_data)

    # Graph 1: Main Benford Arrow Plot (all years combined)
    print("1. Generating Benford Arrow Plot...")
    plot_benford_arrow_plot(
        arrow_data,
        output_path=str(output_path / "01_benford_arrow_plot.png")
    )

    # Graph 2: Arrow Plot by Year
    print("2. Generating Arrow Plot by Year...")
    plot_arrow_plot_by_year(
        arrow_data,
        years=years,
        output_path=str(output_path / "02_arrow_plot_by_year.png")
    )

    # Graph 3: Volume Benford Heatmap
    print("3. Generating Volume Benford Heatmap...")
    # Need wide format for this
    # plot_volume_benford_heatmap(...)  # Implement when data available

    # Graph 4: Combined Time Series
    print("4. Generating Combined Time Series...")
    # Get top suspicious companies
    avg_mad = combined_data.groupby('symbol')['MAD'].mean().nlargest(10)
    top_symbols = avg_mad.index.tolist()
    plot_combined_time_series(
        combined_data,
        symbols=top_symbols[:6],
        output_path=str(output_path / "04_time_series_suspicious.png")
    )

    # Graph 5: Correlation Summary
    print("5. Generating Correlation Summary...")
    plot_correlation_summary(
        combined_data,
        output_path=str(output_path / "05_correlation_summary.png")
    )

    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_path}")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)

    n_companies = 100
    years = list(range(2016, 2024))

    sample_data = []
    for i in range(n_companies):
        symbol = f"COMP{i:03d}"
        for year in years:
            sample_data.append({
                'symbol': symbol,
                'year': year,
                'MAD': np.random.exponential(1.5),
                'chi_square': np.random.exponential(10),
                'p_value': np.random.uniform(0, 1),
                'stock_annual_return': np.random.normal(10, 30),
                'stock_volatility': np.random.exponential(20),
            })

    df = pd.DataFrame(sample_data)

    # Prepare arrow data
    from stock_metrics import get_arrow_plot_data
    arrow_df = get_arrow_plot_data(df)

    # Test arrow plot
    print("Testing arrow plot...")
    fig = plot_benford_arrow_plot(arrow_df, output_path="test_arrow_plot.png")
    plt.show()
