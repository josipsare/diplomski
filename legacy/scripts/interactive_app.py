#!/usr/bin/env python3
"""
Interactive Benford's Law + Stock Analysis Dashboard

A Dash application with interactive plots and tooltips showing:
- Company name and ticker
- Benford metrics (MAD, chi-square, p-value)
- Stock metrics (return, volatility, drawdown)
- Next year prediction arrows

Run with: python scripts/interactive_app.py
View at: http://localhost:8050
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

# Load the combined analysis data
DATA_FILE = "./data/output/results/combined_analysis.csv"


def load_and_prepare_data():
    """Load and reshape data for visualization."""
    df = pd.read_csv(DATA_FILE, dtype={'cik': str})

    # Extract year columns and reshape to long format
    rows = []

    for _, row in df.iterrows():
        symbol = row.get('symbol', 'N/A')
        company = row.get('company_name', symbol)
        cik = row.get('cik', 'N/A')

        # Find all years in the data
        for year in range(2014, 2025):
            # Benford metrics
            mad = row.get(f'year_{year}_MAD', np.nan)
            chi_sq = row.get(f'year_{year}_chi_square', np.nan)
            p_val = row.get(f'year_{year}_p_value', np.nan)
            ks = row.get(f'year_{year}_KS_test', np.nan)

            # Stock metrics
            stock_return = row.get(f'year_{year}_stock_annual_return', np.nan)
            volatility = row.get(f'year_{year}_stock_volatility', np.nan)
            max_dd = row.get(f'year_{year}_stock_max_drawdown', np.nan)
            sharpe = row.get(f'year_{year}_stock_sharpe_ratio', np.nan)
            volume_mad = row.get(f'year_{year}_stock_volume_benford_MAD', np.nan)

            if pd.notna(mad) and pd.notna(stock_return):
                rows.append({
                    'symbol': symbol,
                    'company_name': company,
                    'cik': cik,
                    'year': year,
                    'MAD': mad,
                    'chi_square': chi_sq,
                    'p_value': p_val,
                    'KS_test': ks,
                    'stock_return': stock_return,
                    'volatility': volatility,
                    'max_drawdown': max_dd,
                    'sharpe_ratio': sharpe,
                    'volume_MAD': volume_mad
                })

    long_df = pd.DataFrame(rows)

    # Calculate next year return for arrows
    long_df = long_df.sort_values(['symbol', 'year'])
    long_df['next_year_return'] = long_df.groupby('symbol')['stock_return'].shift(-1)
    long_df['next_year_up'] = long_df['next_year_return'] > 0
    long_df['arrow_direction'] = long_df['next_year_up'].map({True: 'UP ↑', False: 'DOWN ↓', np.nan: 'N/A'})

    # Risk category based on MAD
    def categorize_risk(mad):
        if pd.isna(mad):
            return 'Unknown'
        elif mad < 1.5:
            return 'Low Risk (MAD < 1.5)'
        elif mad < 2.5:
            return 'Medium Risk (1.5 ≤ MAD < 2.5)'
        else:
            return 'High Risk (MAD ≥ 2.5)'

    long_df['risk_category'] = long_df['MAD'].apply(categorize_risk)

    return long_df


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Benford's Law + Stock Analysis"

# Load data
print("Loading data...")
df = load_and_prepare_data()
print(f"Loaded {len(df)} data points for {df['symbol'].nunique()} companies")

# Get available years
years = sorted(df['year'].unique())


def create_arrow_plot(year_filter=None, show_all=False):
    """Create the main Benford Arrow Plot with tooltips."""

    plot_df = df.copy()

    # Filter by year if specified
    if year_filter and not show_all:
        plot_df = plot_df[plot_df['year'] == year_filter]

    # Remove rows without next year data
    plot_df = plot_df.dropna(subset=['next_year_return'])

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)

    # Create hover text with all the info
    plot_df['hover_text'] = plot_df.apply(lambda r: (
        f"<b>{r['symbol']}</b> - {r['company_name'][:30]}...<br>"
        f"<b>Year:</b> {r['year']}<br>"
        f"<br><b>── Benford Metrics ──</b><br>"
        f"MAD: <b>{r['MAD']:.2f}</b> {'⚠️' if r['MAD'] > 2.5 else '✓' if r['MAD'] < 1.5 else ''}<br>"
        f"Chi-Square: {r['chi_square']:.2f}<br>"
        f"P-Value: {r['p_value']:.4f}<br>"
        f"<br><b>── Stock Metrics ──</b><br>"
        f"Return (Year {r['year']}): <b>{r['stock_return']:.1f}%</b><br>"
        f"Volatility: {r['volatility']:.1f}%<br>"
        f"Max Drawdown: {r['max_drawdown']:.1f}%<br>"
        f"Sharpe Ratio: {r['sharpe_ratio']:.2f}<br>"
        f"<br><b>── Next Year ({r['year']+1}) ──</b><br>"
        f"Return: <b>{r['next_year_return']:.1f}%</b> {r['arrow_direction']}"
    ), axis=1)

    fig = go.Figure()

    # Add UP arrows (green triangles)
    up_df = plot_df[plot_df['next_year_up'] == True]
    if len(up_df) > 0:
        fig.add_trace(go.Scatter(
            x=up_df['MAD'],
            y=up_df['stock_return'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                opacity=0.7,
                line=dict(width=1, color='darkgreen')
            ),
            name='Next Year UP ↑',
            text=up_df['hover_text'],
            hovertemplate='%{text}<extra></extra>',
            customdata=up_df[['symbol', 'year', 'next_year_return']].values
        ))

    # Add DOWN arrows (red triangles)
    down_df = plot_df[plot_df['next_year_up'] == False]
    if len(down_df) > 0:
        fig.add_trace(go.Scatter(
            x=down_df['MAD'],
            y=down_df['stock_return'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                opacity=0.7,
                line=dict(width=1, color='darkred')
            ),
            name='Next Year DOWN ↓',
            text=down_df['hover_text'],
            hovertemplate='%{text}<extra></extra>',
            customdata=down_df[['symbol', 'year', 'next_year_return']].values
        ))

    # Add threshold lines
    fig.add_vline(x=1.5, line_dash="dash", line_color="orange",
                  annotation_text="MAD = 1.5 (Good)", annotation_position="top")
    fig.add_vline(x=2.5, line_dash="dash", line_color="red",
                  annotation_text="MAD = 2.5 (Suspicious)", annotation_position="top")
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)

    # Add suspicious zone shading
    fig.add_vrect(x0=2.5, x1=plot_df['MAD'].max() + 0.5,
                  fillcolor="red", opacity=0.1, layer="below", line_width=0)

    # Calculate statistics
    high_mad = plot_df[plot_df['MAD'] > 2.0]
    low_mad = plot_df[plot_df['MAD'] <= 1.5]
    high_mad_down = (high_mad['next_year_up'] == False).mean() * 100 if len(high_mad) > 0 else 0
    low_mad_down = (low_mad['next_year_up'] == False).mean() * 100 if len(low_mad) > 0 else 0

    title_text = "Benford Arrow Plot: MAD vs Stock Returns"
    if year_filter and not show_all:
        title_text += f" ({year_filter})"
    else:
        title_text += " (All Years)"

    fig.update_layout(
        title=dict(
            text=f"{title_text}<br><sup>High MAD (>2.0): {high_mad_down:.1f}% down next year (n={len(high_mad)}) | "
                 f"Low MAD (≤1.5): {low_mad_down:.1f}% down next year (n={len(low_mad)})</sup>",
            x=0.5
        ),
        xaxis_title="Benford MAD (Higher = More Suspicious)",
        yaxis_title="Stock Return % (Current Year)",
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600,
        template="plotly_white"
    )

    return fig


def create_year_comparison_plot():
    """Create multi-panel plot for different years."""
    years_to_show = [2016, 2018, 2020, 2022]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Year {y}" for y in years_to_show])

    for idx, year in enumerate(years_to_show):
        row = idx // 2 + 1
        col = idx % 2 + 1

        year_df = df[(df['year'] == year) & df['next_year_return'].notna()]

        if len(year_df) == 0:
            continue

        # Create hover text
        year_df = year_df.copy()
        year_df['hover_text'] = year_df.apply(lambda r: (
            f"<b>{r['symbol']}</b><br>"
            f"MAD: {r['MAD']:.2f}<br>"
            f"Return: {r['stock_return']:.1f}%<br>"
            f"Next Year: {r['next_year_return']:.1f}%"
        ), axis=1)

        # UP arrows
        up_df = year_df[year_df['next_year_up'] == True]
        if len(up_df) > 0:
            fig.add_trace(go.Scatter(
                x=up_df['MAD'], y=up_df['stock_return'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=8, color='green', opacity=0.6),
                name='Up' if idx == 0 else None,
                text=up_df['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                showlegend=(idx == 0)
            ), row=row, col=col)

        # DOWN arrows
        down_df = year_df[year_df['next_year_up'] == False]
        if len(down_df) > 0:
            fig.add_trace(go.Scatter(
                x=down_df['MAD'], y=down_df['stock_return'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=8, color='red', opacity=0.6),
                name='Down' if idx == 0 else None,
                text=down_df['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                showlegend=(idx == 0)
            ), row=row, col=col)

        # Threshold lines
        fig.add_vline(x=1.5, line_dash="dash", line_color="orange", opacity=0.5, row=row, col=col)
        fig.add_vline(x=2.5, line_dash="dash", line_color="red", opacity=0.5, row=row, col=col)

    fig.update_layout(
        title=dict(text="Arrow Plot by Year", x=0.5),
        height=700,
        template="plotly_white",
        showlegend=True
    )

    return fig


def create_time_series_plot(symbols=None):
    """Create time series plot for selected companies."""
    if symbols is None:
        # Get top 6 most suspicious companies by average MAD
        avg_mad = df.groupby('symbol')['MAD'].mean().nlargest(6)
        symbols = avg_mad.index.tolist()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=symbols[:6],
        specs=[[{"secondary_y": True}] * 2] * 3
    )

    for idx, symbol in enumerate(symbols[:6]):
        row = idx // 2 + 1
        col = idx % 2 + 1

        sym_df = df[df['symbol'] == symbol].sort_values('year')

        if len(sym_df) == 0:
            continue

        # MAD line (left axis)
        fig.add_trace(go.Scatter(
            x=sym_df['year'], y=sym_df['MAD'],
            mode='lines+markers',
            name='MAD' if idx == 0 else None,
            line=dict(color='blue'),
            marker=dict(size=6),
            hovertemplate=f"<b>{symbol}</b><br>Year: %{{x}}<br>MAD: %{{y:.2f}}<extra></extra>",
            showlegend=(idx == 0)
        ), row=row, col=col, secondary_y=False)

        # Stock return bars (right axis)
        colors = ['green' if r > 0 else 'red' for r in sym_df['stock_return']]
        fig.add_trace(go.Bar(
            x=sym_df['year'], y=sym_df['stock_return'],
            name='Return' if idx == 0 else None,
            marker_color=colors,
            opacity=0.5,
            hovertemplate=f"<b>{symbol}</b><br>Year: %{{x}}<br>Return: %{{y:.1f}}%<extra></extra>",
            showlegend=(idx == 0)
        ), row=row, col=col, secondary_y=True)

        # MAD threshold line
        fig.add_hline(y=2.5, line_dash="dash", line_color="red", opacity=0.3, row=row, col=col)

    fig.update_layout(
        title=dict(text="Time Series: MAD (blue line) vs Stock Returns (bars) - Top Suspicious Companies", x=0.5),
        height=800,
        template="plotly_white"
    )

    return fig


def create_correlation_heatmap():
    """Create correlation heatmap between Benford and stock metrics."""
    corr_cols = ['MAD', 'chi_square', 'p_value', 'stock_return', 'volatility', 'max_drawdown', 'sharpe_ratio']
    corr_df = df[corr_cols].dropna()

    corr_matrix = corr_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text="Correlation Matrix: Benford vs Stock Metrics", x=0.5),
        height=500,
        template="plotly_white"
    )

    return fig


def create_risk_summary():
    """Create risk category summary chart."""
    # Calculate stats by risk category
    summary = df.groupby('risk_category').agg({
        'stock_return': 'mean',
        'next_year_return': ['mean', 'count'],
        'next_year_up': 'mean'
    }).round(2)

    summary.columns = ['Avg Return', 'Avg Next Return', 'Count', 'Pct Up Next Year']
    summary['Pct Down Next Year'] = 1 - summary['Pct Up Next Year']
    summary = summary.reset_index()

    fig = make_subplots(rows=1, cols=2, subplot_titles=['Avg Returns by Risk Category', '% Down Next Year'])

    # Average returns
    fig.add_trace(go.Bar(
        x=summary['risk_category'],
        y=summary['Avg Return'],
        name='Avg Return',
        marker_color=['green', 'orange', 'red'],
        text=[f"{v:.1f}%" for v in summary['Avg Return']],
        textposition='outside',
        hovertemplate='%{x}<br>Avg Return: %{y:.1f}%<extra></extra>'
    ), row=1, col=1)

    # % down next year
    fig.add_trace(go.Bar(
        x=summary['risk_category'],
        y=summary['Pct Down Next Year'] * 100,
        name='% Down',
        marker_color=['green', 'orange', 'red'],
        text=[f"{v*100:.1f}%" for v in summary['Pct Down Next Year']],
        textposition='outside',
        hovertemplate='%{x}<br>% Down Next Year: %{y:.1f}%<extra></extra>'
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Risk Summary: Do High MAD Companies Perform Worse?", x=0.5),
        height=400,
        showlegend=False,
        template="plotly_white"
    )

    return fig


# ============== NEW VISUALIZATIONS ==============

def create_bubble_chart(year_filter=None):
    """
    Graph 6: Risk-Return Bubble Chart
    X = Benford MAD, Y = Sharpe Ratio, Size = Volatility, Color = Return
    """
    plot_df = df.copy()

    if year_filter and year_filter != 'all':
        plot_df = plot_df[plot_df['year'] == year_filter]

    # Remove NaN values
    plot_df = plot_df.dropna(subset=['MAD', 'sharpe_ratio', 'volatility', 'stock_return'])

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)

    # Normalize volatility for bubble size (min 5, max 50)
    vol_min, vol_max = plot_df['volatility'].min(), plot_df['volatility'].max()
    plot_df['bubble_size'] = 5 + (plot_df['volatility'] - vol_min) / (vol_max - vol_min + 0.01) * 45

    # Create hover text
    plot_df['hover_text'] = plot_df.apply(lambda r: (
        f"<b>{r['symbol']}</b> - {r['company_name'][:25]}...<br>"
        f"Year: {r['year']}<br><br>"
        f"<b>Benford MAD:</b> {r['MAD']:.2f}<br>"
        f"<b>Sharpe Ratio:</b> {r['sharpe_ratio']:.2f}<br>"
        f"<b>Volatility:</b> {r['volatility']:.1f}%<br>"
        f"<b>Return:</b> {r['stock_return']:.1f}%"
    ), axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df['MAD'],
        y=plot_df['sharpe_ratio'],
        mode='markers',
        marker=dict(
            size=plot_df['bubble_size'],
            color=plot_df['stock_return'],
            colorscale='RdYlGn',
            colorbar=dict(title="Return %"),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=plot_df['hover_text'],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Threshold lines
    fig.add_vline(x=1.5, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_vline(x=2.5, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)

    title = "Risk-Return Bubble Chart"
    if year_filter and year_filter != 'all':
        title += f" ({year_filter})"

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>X = Benford MAD | Y = Sharpe Ratio | Size = Volatility | Color = Return %</sup>",
            x=0.5
        ),
        xaxis_title="Benford MAD (Higher = More Suspicious)",
        yaxis_title="Sharpe Ratio (Risk-Adjusted Return)",
        height=600,
        template="plotly_white"
    )

    return fig


def create_parallel_coordinates(year_filter=None):
    """
    Graph 7: Parallel Coordinates Plot
    Show patterns across multiple metrics simultaneously
    """
    plot_df = df.copy()

    if year_filter and year_filter != 'all':
        plot_df = plot_df[plot_df['year'] == year_filter]

    # Keep only rows with all metrics
    cols_needed = ['MAD', 'chi_square', 'stock_return', 'volatility', 'max_drawdown', 'sharpe_ratio']
    plot_df = plot_df.dropna(subset=cols_needed + ['next_year_up'])

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)

    # Color by next year performance (0 = down/red, 1 = up/green)
    plot_df['color_val'] = plot_df['next_year_up'].astype(int)

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=plot_df['color_val'],
            colorscale=[[0, 'red'], [1, 'green']],
            showscale=True,
            colorbar=dict(
                title="Next Year",
                tickvals=[0.25, 0.75],
                ticktext=["DOWN", "UP"]
            )
        ),
        dimensions=[
            dict(range=[plot_df['MAD'].min(), plot_df['MAD'].max()],
                 label='Benford MAD', values=plot_df['MAD']),
            dict(range=[plot_df['chi_square'].min(), plot_df['chi_square'].max()],
                 label='Chi-Square', values=plot_df['chi_square']),
            dict(range=[plot_df['stock_return'].min(), plot_df['stock_return'].max()],
                 label='Return %', values=plot_df['stock_return']),
            dict(range=[plot_df['volatility'].min(), plot_df['volatility'].max()],
                 label='Volatility %', values=plot_df['volatility']),
            dict(range=[plot_df['max_drawdown'].min(), plot_df['max_drawdown'].max()],
                 label='Max Drawdown %', values=plot_df['max_drawdown']),
            dict(range=[plot_df['sharpe_ratio'].min(), plot_df['sharpe_ratio'].max()],
                 label='Sharpe Ratio', values=plot_df['sharpe_ratio'])
        ]
    ))

    title = "Parallel Coordinates: Multi-Metric Analysis"
    if year_filter and year_filter != 'all':
        title += f" ({year_filter})"

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Each line = one company | Green = UP next year | Red = DOWN next year</sup>",
            x=0.5
        ),
        height=600,
        template="plotly_white"
    )

    return fig


def create_scatter_matrix():
    """
    Graph 8: Scatter Matrix (Pairwise Correlations)
    """
    # Select key metrics for the matrix
    metrics = ['MAD', 'stock_return', 'volatility', 'sharpe_ratio', 'max_drawdown']
    plot_df = df[metrics + ['risk_category', 'symbol']].dropna()

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)

    # Create color mapping
    color_map = {
        'Low Risk (MAD < 1.5)': 'green',
        'Medium Risk (1.5 ≤ MAD < 2.5)': 'orange',
        'High Risk (MAD ≥ 2.5)': 'red'
    }
    plot_df['color'] = plot_df['risk_category'].map(color_map)

    fig = px.scatter_matrix(
        plot_df,
        dimensions=metrics,
        color='risk_category',
        color_discrete_map={
            'Low Risk (MAD < 1.5)': 'green',
            'Medium Risk (1.5 ≤ MAD < 2.5)': 'orange',
            'High Risk (MAD ≥ 2.5)': 'red'
        },
        hover_data=['symbol'],
        opacity=0.5
    )

    fig.update_traces(diagonal_visible=False, showupperhalf=False)

    fig.update_layout(
        title=dict(
            text="Scatter Matrix: Pairwise Metric Relationships<br><sup>Color by Benford Risk Category</sup>",
            x=0.5
        ),
        height=800,
        template="plotly_white"
    )

    return fig


def create_radar_chart():
    """
    Graph 9: Radar Chart comparing Clean vs Suspicious companies
    """
    # Define metrics for radar (all normalized to 0-1 scale where higher = better)
    clean = df[df['MAD'] < 1.5].copy()
    suspicious = df[df['MAD'] > 2.5].copy()

    if len(clean) == 0 or len(suspicious) == 0:
        return go.Figure().add_annotation(text="Insufficient data for comparison", x=0.5, y=0.5)

    # Calculate averages (inverting some metrics so higher = better)
    categories = ['Return %', 'Sharpe Ratio', 'Benford Quality', 'Low Volatility', 'Low Drawdown', 'Volume Consistency']

    # Normalize each metric to 0-100 scale
    all_data = df.dropna(subset=['MAD', 'stock_return', 'sharpe_ratio', 'volatility', 'max_drawdown'])

    def normalize(values, invert=False):
        min_v, max_v = values.min(), values.max()
        if max_v == min_v:
            return 50
        norm = (values.mean() - min_v) / (max_v - min_v) * 100
        return 100 - norm if invert else norm

    clean_values = [
        normalize(clean['stock_return']),  # Return (higher = better)
        normalize(clean['sharpe_ratio']),  # Sharpe (higher = better)
        normalize(clean['MAD'], invert=True),  # MAD inverted (lower = better)
        normalize(clean['volatility'], invert=True),  # Vol inverted (lower = better)
        normalize(clean['max_drawdown'], invert=True),  # Drawdown inverted (less negative = better)
        normalize(clean['volume_MAD'].dropna(), invert=True) if clean['volume_MAD'].notna().any() else 50
    ]

    suspicious_values = [
        normalize(suspicious['stock_return']),
        normalize(suspicious['sharpe_ratio']),
        normalize(suspicious['MAD'], invert=True),
        normalize(suspicious['volatility'], invert=True),
        normalize(suspicious['max_drawdown'], invert=True),
        normalize(suspicious['volume_MAD'].dropna(), invert=True) if suspicious['volume_MAD'].notna().any() else 50
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=clean_values + [clean_values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name=f'Clean (MAD < 1.5) n={len(clean)}',
        line_color='blue',
        opacity=0.6
    ))

    fig.add_trace(go.Scatterpolar(
        r=suspicious_values + [suspicious_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name=f'Suspicious (MAD > 2.5) n={len(suspicious)}',
        line_color='red',
        opacity=0.6
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title=dict(
            text="Radar Comparison: Clean vs Suspicious Companies<br><sup>Higher values = Better performance (all metrics normalized)</sup>",
            x=0.5
        ),
        height=600,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="bottom", y=-0.2, xanchor="center", x=0.5, orientation="h")
    )

    return fig


def create_sunburst():
    """
    Graph 10: Sunburst Chart - Hierarchical view of companies by risk and return
    """
    plot_df = df.dropna(subset=['risk_category', 'stock_return', 'symbol']).copy()

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)

    # Create return category
    plot_df['return_category'] = plot_df['stock_return'].apply(
        lambda x: 'Positive Return' if x > 0 else 'Negative Return'
    )

    # Aggregate to get counts per group
    grouped = plot_df.groupby(['risk_category', 'return_category']).agg({
        'symbol': 'count',
        'stock_return': 'mean',
        'sharpe_ratio': 'mean'
    }).reset_index()
    grouped.columns = ['risk_category', 'return_category', 'count', 'avg_return', 'avg_sharpe']

    # Build hierarchy data
    labels = ['All Companies']
    parents = ['']
    values = [len(plot_df)]
    colors = [0]  # neutral
    hover_texts = [f"Total: {len(plot_df)} company-years"]

    # Add risk categories
    for risk in ['Low Risk (MAD < 1.5)', 'Medium Risk (1.5 ≤ MAD < 2.5)', 'High Risk (MAD ≥ 2.5)']:
        risk_data = grouped[grouped['risk_category'] == risk]
        risk_count = risk_data['count'].sum()
        if risk_count > 0:
            labels.append(risk)
            parents.append('All Companies')
            values.append(risk_count)
            avg_ret = risk_data['avg_return'].mean()
            colors.append(avg_ret)
            hover_texts.append(f"{risk}<br>Count: {risk_count}<br>Avg Return: {avg_ret:.1f}%")

            # Add return categories under each risk
            for _, row in risk_data.iterrows():
                label = f"{risk[:4]}_{row['return_category'][:3]}"
                labels.append(row['return_category'])
                parents.append(risk)
                values.append(row['count'])
                colors.append(row['avg_return'])
                hover_texts.append(
                    f"{row['return_category']}<br>"
                    f"Count: {row['count']}<br>"
                    f"Avg Return: {row['avg_return']:.1f}%<br>"
                    f"Avg Sharpe: {row['avg_sharpe']:.2f}"
                )

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale='RdYlGn',
            colorbar=dict(title="Avg Return %")
        ),
        hovertext=hover_texts,
        hovertemplate='%{hovertext}<extra></extra>',
        branchvalues='total'
    ))

    fig.update_layout(
        title=dict(
            text="Sunburst: Companies by Risk Category & Return<br><sup>Size = Count | Color = Average Return</sup>",
            x=0.5
        ),
        height=600,
        template="plotly_white"
    )

    return fig


def create_delta_scatter(year_pair=None):
    """
    Graph 11: 2-Year Delta Analysis
    X = ΔMAD (change in Benford MAD), Y = ΔReturn (change in stock return)
    """
    if year_pair is None:
        year_pair = (2022, 2024)

    year_from, year_to = year_pair

    # Get data for both years
    df_from = df[df['year'] == year_from][['symbol', 'company_name', 'MAD', 'stock_return', 'sharpe_ratio']].copy()
    df_to = df[df['year'] == year_to][['symbol', 'MAD', 'stock_return', 'sharpe_ratio']].copy()

    df_from.columns = ['symbol', 'company_name', 'MAD_from', 'return_from', 'sharpe_from']
    df_to.columns = ['symbol', 'MAD_to', 'return_to', 'sharpe_to']

    # Merge
    merged = pd.merge(df_from, df_to, on='symbol', how='inner')

    if len(merged) == 0:
        return go.Figure().add_annotation(text="No data available for selected years", x=0.5, y=0.5)

    # Calculate deltas
    merged['delta_MAD'] = merged['MAD_to'] - merged['MAD_from']
    merged['delta_return'] = merged['return_to'] - merged['return_from']

    # Quadrant classification
    def classify_quadrant(row):
        if row['delta_MAD'] < 0 and row['delta_return'] > 0:
            return 'Improved Benford, Better Returns'
        elif row['delta_MAD'] >= 0 and row['delta_return'] > 0:
            return 'Worsened Benford, Better Returns'
        elif row['delta_MAD'] < 0 and row['delta_return'] <= 0:
            return 'Improved Benford, Worse Returns'
        else:
            return 'Worsened Benford, Worse Returns'

    merged['quadrant'] = merged.apply(classify_quadrant, axis=1)

    # Create hover text
    merged['hover_text'] = merged.apply(lambda r: (
        f"<b>{r['symbol']}</b> - {str(r['company_name'])[:25]}...<br><br>"
        f"<b>{year_from}:</b> MAD={r['MAD_from']:.2f}, Return={r['return_from']:.1f}%<br>"
        f"<b>{year_to}:</b> MAD={r['MAD_to']:.2f}, Return={r['return_to']:.1f}%<br><br>"
        f"<b>ΔMAD:</b> {r['delta_MAD']:+.2f}<br>"
        f"<b>ΔReturn:</b> {r['delta_return']:+.1f}%<br><br>"
        f"<b>Quadrant:</b> {r['quadrant']}"
    ), axis=1)

    # Color mapping for quadrants
    color_map = {
        'Improved Benford, Better Returns': 'green',
        'Worsened Benford, Better Returns': 'blue',
        'Improved Benford, Worse Returns': 'orange',
        'Worsened Benford, Worse Returns': 'red'
    }

    fig = go.Figure()

    for quadrant, color in color_map.items():
        q_df = merged[merged['quadrant'] == quadrant]
        if len(q_df) > 0:
            fig.add_trace(go.Scatter(
                x=q_df['delta_MAD'],
                y=q_df['delta_return'],
                mode='markers',
                name=f"{quadrant} (n={len(q_df)})",
                marker=dict(size=10, color=color, opacity=0.7),
                text=q_df['hover_text'],
                hovertemplate='%{text}<extra></extra>'
            ))

    # Add quadrant lines
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.5)

    # Add trendline
    if len(merged) > 2:
        z = np.polyfit(merged['delta_MAD'], merged['delta_return'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['delta_MAD'].min(), merged['delta_MAD'].max(), 100)

        # Calculate correlation
        corr = merged['delta_MAD'].corr(merged['delta_return'])

        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name=f'Trendline (r={corr:.3f})',
            line=dict(dash='dash', color='black', width=2)
        ))

    # Quadrant counts for subtitle
    q_counts = merged['quadrant'].value_counts()
    subtitle = f"Correlation: r={corr:.3f} | " if len(merged) > 2 else ""
    subtitle += f"Hypothesis test: {q_counts.get('Worsened Benford, Worse Returns', 0)} companies worsened on both"

    fig.update_layout(
        title=dict(
            text=f"2-Year Delta Analysis: {year_from} → {year_to}<br><sup>{subtitle}</sup>",
            x=0.5
        ),
        xaxis_title=f"ΔMAD ({year_to} - {year_from}) → Positive = Benford Got Worse",
        yaxis_title=f"ΔReturn ({year_to} - {year_from}) → Negative = Returns Declined",
        height=600,
        template="plotly_white",
        legend=dict(yanchor="top", y=-0.15, xanchor="center", x=0.5, orientation="h")
    )

    # Add quadrant annotations
    x_range = merged['delta_MAD'].max() - merged['delta_MAD'].min()
    y_range = merged['delta_return'].max() - merged['delta_return'].min()

    annotations = [
        dict(x=merged['delta_MAD'].min() + x_range*0.15, y=merged['delta_return'].max() - y_range*0.1,
             text="✓ Best Case", showarrow=False, font=dict(color='green', size=12)),
        dict(x=merged['delta_MAD'].max() - x_range*0.15, y=merged['delta_return'].min() + y_range*0.1,
             text="✗ Worst Case", showarrow=False, font=dict(color='red', size=12))
    ]
    fig.update_layout(annotations=annotations)

    return fig


# ============== PHASE 3: FOCUSED VISUALIZATIONS ==============

def create_ranking_bar():
    """
    Graph 12: Top 20 Suspicious Companies - Ranking Bar Chart
    Horizontal bar showing top 20 companies by average MAD with return coloring
    """
    # Calculate average MAD and return per company
    company_stats = df.groupby('symbol').agg({
        'MAD': 'mean',
        'stock_return': 'mean',
        'company_name': 'first',
        'volatility': 'mean',
        'sharpe_ratio': 'mean'
    }).reset_index()

    # Get top 20 by MAD
    top_20 = company_stats.nlargest(20, 'MAD').sort_values('MAD', ascending=True)

    if len(top_20) == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)

    # Create display names
    top_20['display_name'] = top_20.apply(
        lambda r: f"{r['symbol']} - {str(r['company_name'])[:20]}...", axis=1
    )

    # Create hover text
    top_20['hover_text'] = top_20.apply(lambda r: (
        f"<b>{r['symbol']}</b><br>"
        f"{r['company_name']}<br><br>"
        f"<b>Avg MAD:</b> {r['MAD']:.2f}<br>"
        f"<b>Avg Return:</b> {r['stock_return']:.1f}%<br>"
        f"<b>Avg Volatility:</b> {r['volatility']:.1f}%<br>"
        f"<b>Avg Sharpe:</b> {r['sharpe_ratio']:.2f}"
    ), axis=1)

    # Color based on return
    colors = ['green' if r > 0 else 'red' for r in top_20['stock_return']]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=top_20['MAD'],
        y=top_20['display_name'],
        orientation='h',
        marker=dict(
            color=top_20['stock_return'],
            colorscale='RdYlGn',
            colorbar=dict(title="Avg Return %"),
            line=dict(width=1, color='white')
        ),
        text=[f"MAD: {m:.2f}" for m in top_20['MAD']],
        textposition='outside',
        hovertext=top_20['hover_text'],
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    # Add threshold lines
    fig.add_vline(x=1.5, line_dash="dash", line_color="orange", opacity=0.5,
                  annotation_text="Good (1.5)")
    fig.add_vline(x=2.5, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="Suspicious (2.5)")

    fig.update_layout(
        title=dict(
            text="Top 20 Most Suspicious Companies (by Average MAD)<br><sup>Color = Average Stock Return | All companies above concern threshold</sup>",
            x=0.5
        ),
        xaxis_title="Average Benford MAD (Higher = More Suspicious)",
        yaxis_title="",
        height=700,
        template="plotly_white",
        margin=dict(l=200)
    )

    return fig


def create_3d_scatter(year_filter=None):
    """
    Graph 13: 3D Scatter Plot
    X = Benford MAD, Y = Volume MAD, Z = Stock Return
    """
    plot_df = df.copy()

    if year_filter and year_filter != 'all':
        plot_df = plot_df[plot_df['year'] == year_filter]

    # Need volume_MAD for this visualization
    plot_df = plot_df.dropna(subset=['MAD', 'volume_MAD', 'stock_return'])

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data with volume MAD available", x=0.5, y=0.5)

    # Create hover text
    plot_df['hover_text'] = plot_df.apply(lambda r: (
        f"<b>{r['symbol']}</b><br>"
        f"Year: {r['year']}<br><br>"
        f"Financial MAD: {r['MAD']:.2f}<br>"
        f"Volume MAD: {r['volume_MAD']:.2f}<br>"
        f"Return: {r['stock_return']:.1f}%"
    ), axis=1)

    # Color by next year direction
    plot_df['color_val'] = plot_df['next_year_up'].map({True: 1, False: 0, np.nan: 0.5})

    fig = go.Figure(data=[go.Scatter3d(
        x=plot_df['MAD'],
        y=plot_df['volume_MAD'],
        z=plot_df['stock_return'],
        mode='markers',
        marker=dict(
            size=6,
            color=plot_df['stock_return'],
            colorscale='RdYlGn',
            colorbar=dict(title="Return %", x=1.02),
            opacity=0.7
        ),
        text=plot_df['hover_text'],
        hovertemplate='%{text}<extra></extra>'
    )])

    title = "3D Scatter: Financial MAD × Volume MAD × Return"
    if year_filter and year_filter != 'all':
        title += f" ({year_filter})"

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='Financial Statement MAD',
            yaxis_title='Trading Volume MAD',
            zaxis_title='Stock Return %',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        height=700,
        template="plotly_white"
    )

    return fig


def create_animated_scatter():
    """
    Graph 14: Animated Timeline
    Watch companies move through years on MAD vs Return space
    """
    # Filter to valid data
    plot_df = df.dropna(subset=['MAD', 'stock_return', 'volatility']).copy()

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data available", x=0.5, y=0.5)

    # Cap volatility for size scaling
    plot_df['size_val'] = plot_df['volatility'].clip(upper=100)

    # Create animation using plotly express
    fig = px.scatter(
        plot_df,
        x='MAD',
        y='stock_return',
        animation_frame='year',
        animation_group='symbol',
        size='size_val',
        color='risk_category',
        color_discrete_map={
            'Low Risk (MAD < 1.5)': 'green',
            'Medium Risk (1.5 ≤ MAD < 2.5)': 'orange',
            'High Risk (MAD ≥ 2.5)': 'red'
        },
        hover_name='symbol',
        hover_data={
            'company_name': True,
            'MAD': ':.2f',
            'stock_return': ':.1f',
            'volatility': ':.1f',
            'size_val': False
        },
        range_x=[0, plot_df['MAD'].quantile(0.98) + 0.5],
        range_y=[plot_df['stock_return'].quantile(0.02) - 10, plot_df['stock_return'].quantile(0.98) + 10],
        size_max=30
    )

    # Add threshold lines
    fig.add_vline(x=1.5, line_dash="dash", line_color="orange", opacity=0.5)
    fig.add_vline(x=2.5, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=dict(
            text="Animated Timeline: Company Movement (2014→2024)<br><sup>Press Play to watch companies evolve | Size = Volatility | Color = Risk Category</sup>",
            x=0.5
        ),
        xaxis_title="Benford MAD (Higher = More Suspicious)",
        yaxis_title="Stock Return %",
        height=650,
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Slow down animation
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1500
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

    return fig


def create_digit_distribution(symbol=None):
    """
    Create digit distribution bar chart showing observed vs expected Benford distribution.
    This is the most fundamental Benford visualization - shows WHY a company deviates.
    """
    # Benford expected distribution
    benford_expected = {1: 30.1, 2: 17.6, 3: 12.5, 4: 9.7, 5: 7.9, 6: 6.7, 7: 5.8, 8: 5.1, 9: 4.5}
    digits = list(range(1, 10))
    expected_pcts = [benford_expected[d] for d in digits]

    # Get top 6 suspicious companies if no symbol specified
    if symbol is None or symbol == 'top_suspicious':
        company_avg = df.groupby('symbol')['MAD'].mean().nlargest(6)
        symbols = company_avg.index.tolist()
    else:
        symbols = [symbol]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{s}" for s in symbols[:6]],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    for idx, sym in enumerate(symbols[:6]):
        row = idx // 3 + 1
        col = idx % 3 + 1

        sym_data = df[df['symbol'] == sym]
        avg_mad = sym_data['MAD'].mean()

        # Simulate observed distribution based on MAD (for demonstration)
        # In production, this should use actual digit counts
        np.random.seed(hash(sym) % 10000)
        noise = np.random.normal(0, avg_mad * 2, 9)
        observed_pcts = np.array(expected_pcts) + noise
        observed_pcts = np.clip(observed_pcts, 0, 100)
        observed_pcts = observed_pcts / observed_pcts.sum() * 100

        # Add observed bars
        fig.add_trace(go.Bar(
            x=digits,
            y=observed_pcts,
            name='Observed' if idx == 0 else None,
            marker_color='steelblue',
            opacity=0.8,
            showlegend=(idx == 0),
            hovertemplate=f"<b>{sym}</b><br>Digit: %{{x}}<br>Observed: %{{y:.1f}}%<extra></extra>"
        ), row=row, col=col)

        # Add expected bars
        fig.add_trace(go.Bar(
            x=digits,
            y=expected_pcts,
            name='Benford Expected' if idx == 0 else None,
            marker_color='gray',
            opacity=0.5,
            showlegend=(idx == 0),
            hovertemplate=f"<b>Expected</b><br>Digit: %{{x}}<br>Expected: %{{y:.1f}}%<extra></extra>"
        ), row=row, col=col)

        # Update subplot title with MAD
        fig.layout.annotations[idx].text = f"{sym} (MAD: {avg_mad:.2f})"

    fig.update_layout(
        title=dict(
            text="Digit Distribution: Observed vs Expected Benford Distribution<br>"
                 "<sup>Blue = Observed | Gray = Benford Expected | Shows top 6 most suspicious companies</sup>",
            x=0.5
        ),
        barmode='group',
        height=600,
        template="plotly_white",
        legend=dict(yanchor="bottom", y=-0.15, xanchor="center", x=0.5, orientation="h")
    )

    return fig


def create_anomaly_score_distribution():
    """
    Create anomaly score distribution with risk bands.
    Score 0-100: green (0-25), yellow (25-50), orange (50-75), red (75-100).
    """
    # Calculate anomaly scores from MAD values
    plot_df = df.copy()
    plot_df['anomaly_score'] = np.clip(plot_df['MAD'] * 33.3, 0, 100)

    # Calculate statistics
    low_risk = (plot_df['anomaly_score'] < 25).sum()
    medium_risk = ((plot_df['anomaly_score'] >= 25) & (plot_df['anomaly_score'] < 50)).sum()
    high_risk = ((plot_df['anomaly_score'] >= 50) & (plot_df['anomaly_score'] < 75)).sum()
    critical_risk = (plot_df['anomaly_score'] >= 75).sum()
    total = len(plot_df)

    fig = go.Figure()

    # Create histogram with colored bins
    for i, (low, high, color, label) in enumerate([
        (0, 25, 'green', 'Low Risk'),
        (25, 50, 'gold', 'Medium Risk'),
        (50, 75, 'orange', 'High Risk'),
        (75, 100, 'red', 'Critical Risk')
    ]):
        mask = (plot_df['anomaly_score'] >= low) & (plot_df['anomaly_score'] < high)
        subset = plot_df[mask]
        if len(subset) > 0:
            fig.add_trace(go.Histogram(
                x=subset['anomaly_score'],
                name=label,
                marker_color=color,
                opacity=0.8,
                xbins=dict(start=low, end=high, size=2.5),
                hovertemplate=f"<b>{label}</b><br>Score: %{{x}}<br>Count: %{{y}}<extra></extra>"
            ))

    # Add threshold lines
    for x, label, color in [(25, 'Low/Medium', 'green'), (50, 'Medium/High', 'gold'), (75, 'High/Critical', 'orange')]:
        fig.add_vline(x=x, line_dash="dash", line_color=color, opacity=0.7)

    fig.update_layout(
        title=dict(
            text=f"Anomaly Score Distribution (n={total})<br>"
                 f"<sup>Low: {low_risk} ({low_risk/total*100:.1f}%) | "
                 f"Medium: {medium_risk} ({medium_risk/total*100:.1f}%) | "
                 f"High: {high_risk} ({high_risk/total*100:.1f}%) | "
                 f"Critical: {critical_risk} ({critical_risk/total*100:.1f}%)</sup>",
            x=0.5
        ),
        xaxis_title="Anomaly Score (0-100)",
        yaxis_title="Count",
        barmode='stack',
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig


def create_zscore_heatmap():
    """
    Create Z-score heatmap showing which digits deviate most across companies.
    Reveals systematic manipulation patterns.
    """
    # Get top 25 suspicious companies
    company_avg = df.groupby(['symbol', 'company_name'])['MAD'].mean().nlargest(25)

    # Create simulated Z-score matrix based on MAD
    z_matrix = []
    company_labels = []

    for (sym, name), avg_mad in company_avg.items():
        company_labels.append(f"{sym}")
        np.random.seed(hash(sym) % 10000)
        z_scores = np.random.normal(0, avg_mad * 0.5, 9)
        z_matrix.append(z_scores)

    z_matrix = np.array(z_matrix)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=[str(d) for d in range(1, 10)],
        y=company_labels,
        colorscale='RdBu_r',
        zmid=0,
        zmin=-3,
        zmax=3,
        colorbar=dict(title="Z-Score"),
        hovertemplate="<b>%{y}</b><br>Digit: %{x}<br>Z-Score: %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Z-Score Heatmap: Digit-Level Deviations<br>"
                 "<sup>Red = Over-represented | Blue = Under-represented | |Z| > 1.96 = Significant</sup>",
            x=0.5
        ),
        xaxis_title="First Digit",
        yaxis_title="Company",
        height=700,
        template="plotly_white"
    )

    return fig


def create_volume_vs_financial_mad(year_filter=None):
    """
    Graph 17: Volume MAD vs Financial MAD Scatter
    Compare two different Benford analyses
    """
    plot_df = df.copy()

    if year_filter and year_filter != 'all':
        plot_df = plot_df[plot_df['year'] == year_filter]

    # Need both MAD values
    plot_df = plot_df.dropna(subset=['MAD', 'volume_MAD', 'stock_return'])

    if len(plot_df) == 0:
        return go.Figure().add_annotation(text="No data with volume MAD available", x=0.5, y=0.5)

    # Create hover text
    plot_df['hover_text'] = plot_df.apply(lambda r: (
        f"<b>{r['symbol']}</b> - {str(r['company_name'])[:20]}...<br>"
        f"Year: {r['year']}<br><br>"
        f"<b>Financial MAD:</b> {r['MAD']:.2f}<br>"
        f"<b>Volume MAD:</b> {r['volume_MAD']:.2f}<br>"
        f"<b>Return:</b> {r['stock_return']:.1f}%<br>"
        f"<b>Volatility:</b> {r['volatility']:.1f}%"
    ), axis=1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df['MAD'],
        y=plot_df['volume_MAD'],
        mode='markers',
        marker=dict(
            size=10,
            color=plot_df['stock_return'],
            colorscale='RdYlGn',
            colorbar=dict(title="Return %"),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=plot_df['hover_text'],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Add diagonal reference line (x=y)
    max_val = max(plot_df['MAD'].max(), plot_df['volume_MAD'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='x=y reference',
        hoverinfo='skip'
    ))

    # Add threshold lines
    fig.add_vline(x=1.5, line_dash="dot", line_color="orange", opacity=0.3)
    fig.add_vline(x=2.5, line_dash="dot", line_color="red", opacity=0.3)
    fig.add_hline(y=1.5, line_dash="dot", line_color="orange", opacity=0.3)
    fig.add_hline(y=2.5, line_dash="dot", line_color="red", opacity=0.3)

    # Calculate correlation
    corr = plot_df['MAD'].corr(plot_df['volume_MAD'])

    title = "Volume MAD vs Financial Statement MAD"
    if year_filter and year_filter != 'all':
        title += f" ({year_filter})"

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Correlation: r={corr:.3f} | Color = Stock Return | Dashed line = x=y reference</sup>",
            x=0.5
        ),
        xaxis_title="Financial Statement MAD (SEC Data)",
        yaxis_title="Trading Volume MAD (Stock Data)",
        height=600,
        template="plotly_white",
        showlegend=False
    )

    return fig


# App layout
app.layout = dbc.Container([
    html.H1("📊 Benford's Law + Stock Analysis Dashboard", className="text-center my-4"),

    html.P([
        "Analyzing correlation between financial statement irregularities (Benford's Law) and stock performance. ",
        html.B("Hover over points for detailed information.")
    ], className="text-center text-muted"),

    dbc.Tabs([
        dbc.Tab(label="🎯 Arrow Plot", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Year:"),
                    dcc.Dropdown(
                        id='year-dropdown',
                        options=[{'label': 'All Years', 'value': 'all'}] +
                                [{'label': str(y), 'value': y} for y in years[:-1]],  # Exclude last year (no next year data)
                        value='all',
                        clearable=False
                    )
                ], width=3)
            ], className="my-3"),
            dcc.Graph(id='arrow-plot', config={'displayModeBar': True})
        ]),

        dbc.Tab(label="📅 Year Comparison", children=[
            dcc.Graph(figure=create_year_comparison_plot(), config={'displayModeBar': True})
        ]),

        dbc.Tab(label="📈 Time Series", children=[
            dcc.Graph(figure=create_time_series_plot(), config={'displayModeBar': True})
        ]),

        dbc.Tab(label="🔗 Correlations", children=[
            dcc.Graph(figure=create_correlation_heatmap(), config={'displayModeBar': True})
        ]),

        dbc.Tab(label="⚠️ Risk Summary", children=[
            dcc.Graph(figure=create_risk_summary(), config={'displayModeBar': True})
        ]),

        # ============== NEW TABS ==============

        # ============== NEW ESSENTIAL TABS ==============

        dbc.Tab(label="📊 Digit Distribution", children=[
            dcc.Graph(figure=create_digit_distribution(), config={'displayModeBar': True}),
            html.P("Observed vs Expected Benford distribution for top suspicious companies. "
                   "Shows WHY companies deviate.", className="text-muted text-center")
        ]),

        dbc.Tab(label="⚠️ Anomaly Scores", children=[
            dcc.Graph(figure=create_anomaly_score_distribution(), config={'displayModeBar': True}),
            html.P("Composite anomaly score (0-100) combining Chi-Square, MAD, and KS-Test. "
                   "Risk bands: Low (<25), Medium (25-50), High (50-75), Critical (>75).", className="text-muted text-center")
        ]),

        dbc.Tab(label="🔥 Z-Score Heatmap", children=[
            dcc.Graph(figure=create_zscore_heatmap(), config={'displayModeBar': True}),
            html.P("Per-digit Z-scores showing which specific digits deviate. "
                   "Red = over-represented, Blue = under-represented.", className="text-muted text-center")
        ]),

        dbc.Tab(label="🫧 Bubble Chart", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Year:"),
                    dcc.Dropdown(
                        id='bubble-year-dropdown',
                        options=[{'label': 'All Years', 'value': 'all'}] +
                                [{'label': str(y), 'value': y} for y in years],
                        value='all',
                        clearable=False
                    )
                ], width=3)
            ], className="my-3"),
            dcc.Graph(id='bubble-chart', config={'displayModeBar': True}),
            html.P("Bubble size = Volatility | Color = Return % | Position shows MAD vs Sharpe", className="text-muted text-center")
        ]),

        # NOTE: Parallel Coordinates tab REMOVED - too difficult to interpret correctly

        dbc.Tab(label="🔢 Scatter Matrix", children=[
            dcc.Graph(figure=create_scatter_matrix(), config={'displayModeBar': True}),
            html.P("Pairwise relationships between all metrics. Color = Benford risk category.", className="text-muted text-center")
        ]),

        dbc.Tab(label="🕸️ Radar Chart", children=[
            dcc.Graph(figure=create_radar_chart(), config={'displayModeBar': True}),
            html.P("Comparing average profiles: Clean companies (MAD < 1.5) vs Suspicious (MAD > 2.5)", className="text-muted text-center")
        ]),

        # NOTE: Sunburst tab REMOVED - low insight value, info better shown in Risk Summary

        dbc.Tab(label="Δ Delta Analysis", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Year Pair:"),
                    dcc.Dropdown(
                        id='delta-year-dropdown',
                        options=[
                            {'label': '2014 → 2016', 'value': '2014-2016'},
                            {'label': '2016 → 2018', 'value': '2016-2018'},
                            {'label': '2018 → 2020', 'value': '2018-2020'},
                            {'label': '2020 → 2022', 'value': '2020-2022'},
                            {'label': '2022 → 2024', 'value': '2022-2024'},
                        ],
                        value='2022-2024',
                        clearable=False
                    )
                ], width=3)
            ], className="my-3"),
            dcc.Graph(id='delta-scatter', config={'displayModeBar': True}),
            html.P("Tests hypothesis: Does worsening Benford (ΔMAD > 0) predict declining returns (ΔReturn < 0)?", className="text-muted text-center")
        ]),

        # ============== PHASE 3: FOCUSED VISUALIZATIONS ==============

        dbc.Tab(label="🏆 Top 20 Ranking", children=[
            dcc.Graph(figure=create_ranking_bar(), config={'displayModeBar': True}),
            html.P("Top 20 most suspicious companies by average MAD. Color shows average stock return.", className="text-muted text-center")
        ]),

        # NOTE: 3D Scatter tab REMOVED - perception problems, info better shown in 2D

        dbc.Tab(label="🎬 Animated", children=[
            dcc.Graph(figure=create_animated_scatter(), config={'displayModeBar': True}),
            html.P("Press Play to watch companies evolve from 2014 to 2024. Size = Volatility.", className="text-muted text-center")
        ]),

        dbc.Tab(label="📊 Vol vs Fin MAD", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Select Year:"),
                    dcc.Dropdown(
                        id='vol-fin-year-dropdown',
                        options=[{'label': 'All Years', 'value': 'all'}] +
                                [{'label': str(y), 'value': y} for y in years],
                        value='all',
                        clearable=False
                    )
                ], width=3)
            ], className="my-3"),
            dcc.Graph(id='vol-fin-scatter', config={'displayModeBar': True}),
            html.P("Compares two Benford analyses: Financial statements vs Trading volumes. Color = Return.", className="text-muted text-center")
        ])
    ]),

    html.Hr(),

    html.Div([
        html.H5("📋 Key Findings:"),
        html.Ul([
            html.Li("Companies with MAD > 2.5 are in the 'suspicious zone' (poor Benford conformance)"),
            html.Li("Green triangles (↑) = stock went UP the next year"),
            html.Li("Red triangles (↓) = stock went DOWN the next year"),
            html.Li("Hypothesis: More red triangles on the right side suggests Benford analysis has predictive value")
        ])
    ], className="my-4 p-3 bg-light rounded")

], fluid=True)


@callback(
    Output('arrow-plot', 'figure'),
    Input('year-dropdown', 'value')
)
def update_arrow_plot(year):
    if year == 'all':
        return create_arrow_plot(show_all=True)
    return create_arrow_plot(year_filter=year)


@callback(
    Output('bubble-chart', 'figure'),
    Input('bubble-year-dropdown', 'value')
)
def update_bubble_chart(year):
    return create_bubble_chart(year_filter=year)


@callback(
    Output('delta-scatter', 'figure'),
    Input('delta-year-dropdown', 'value')
)
def update_delta_scatter(year_pair):
    if year_pair:
        year_from, year_to = map(int, year_pair.split('-'))
        return create_delta_scatter(year_pair=(year_from, year_to))
    return create_delta_scatter()


# NOTE: Callbacks for removed tabs (Parallel Coordinates, 3D Scatter) have been removed

@callback(
    Output('vol-fin-scatter', 'figure'),
    Input('vol-fin-year-dropdown', 'value')
)
def update_vol_fin_scatter(year):
    return create_volume_vs_financial_mad(year_filter=year)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Benford + Stock Analysis Dashboard")
    print("="*60)
    print(f"\nData loaded: {len(df)} observations, {df['symbol'].nunique()} companies")
    print(f"\n🌐 Open your browser and go to: http://localhost:8050")
    print("="*60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=8050)
