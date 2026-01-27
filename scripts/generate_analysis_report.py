#!/usr/bin/env python3
"""
Generate Analysis Report

This script runs all analysis functions on sample data and outputs
a comprehensive report with explanations and interpretations.

Output: analysis_report.txt (in current directory)
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.benford import (
    calculate_benford_metrics,
    calculate_digit_zscores,
    calculate_anomaly_score,
    get_digit_distribution,
    interpret_results,
    BENFORD_EXPECTED
)


def generate_report(data: pd.Series, company_name: str = "Sample Company") -> str:
    """Generate a comprehensive analysis report for given data."""

    lines = []
    lines.append("=" * 80)
    lines.append("BENFORD'S LAW ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Company: {company_name}")
    lines.append("")

    # =====================================================================
    # SECTION 1: BASIC METRICS
    # =====================================================================
    lines.append("-" * 80)
    lines.append("SECTION 1: BASIC BENFORD METRICS")
    lines.append("-" * 80)
    lines.append("")

    metrics = calculate_benford_metrics(data)
    interp = interpret_results(metrics)

    lines.append(f"Sample Size: {metrics['n_samples']:,} values")
    lines.append("")
    lines.append("Metric          Value       Threshold    Status")
    lines.append("-" * 55)

    # Chi-square
    chi_status = "✓ PASS" if interp['chi_square']['conforms'] else "✗ FAIL"
    lines.append(f"Chi-Square      {metrics['chi_square']:8.4f}    < 15.507     {chi_status}")
    lines.append("  └─ Measures overall deviation from expected distribution")
    lines.append("  └─ Lower values indicate better conformance")
    lines.append("")

    # P-value
    p_status = "✗ Significant" if interp['p_value']['significant'] else "✓ Not significant"
    lines.append(f"P-Value         {metrics['p_value']:8.4f}    > 0.05       {p_status}")
    lines.append("  └─ Probability that deviation is due to chance")
    lines.append("  └─ p < 0.05 suggests non-random deviation")
    lines.append("")

    # MAD
    mad_status = "✓ PASS" if interp['MAD']['conforms'] else "✗ FAIL"
    lines.append(f"MAD             {metrics['MAD']:8.4f}    < 1.5        {mad_status}")
    lines.append(f"  └─ Conformity level: {interp['MAD']['level']}")
    lines.append("  └─ Mean Absolute Deviation from expected percentages")
    lines.append("")

    # KS test
    lines.append(f"KS-Test         {metrics['KS_test']:8.4f}    < 0.10       {'✓' if metrics['KS_test'] < 0.1 else '!'}")
    lines.append("  └─ Maximum point-wise deviation from Benford CDF")
    lines.append("")

    # =====================================================================
    # SECTION 2: Z-SCORE ANALYSIS
    # =====================================================================
    lines.append("-" * 80)
    lines.append("SECTION 2: Z-SCORE ANALYSIS (Per-Digit Deviation)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Purpose: Identifies which SPECIFIC digits deviate from expected")
    lines.append("         |Z| > 1.96 indicates statistically significant deviation")
    lines.append("")

    z_result = calculate_digit_zscores(data)

    lines.append("Digit  Observed%  Expected%  Diff%     Z-Score   Significance")
    lines.append("-" * 70)

    for analysis in z_result['digit_analysis']:
        d = analysis['digit']
        obs = analysis['observed_pct']
        exp = analysis['expected_pct']
        diff = analysis['difference_pct']
        z = analysis['z_score']
        sig = analysis['significance']
        direction = analysis['direction']

        # Status indicator
        if sig == 'highly_significant':
            status = "⚠️  HIGHLY SIG"
        elif sig == 'significant':
            status = "⚠️  SIGNIFICANT"
        else:
            status = "✓  Normal"

        lines.append(f"  {d}      {obs:6.2f}%    {exp:5.2f}%   {diff:+6.2f}%   {z:+7.4f}   {status}")

    lines.append("")
    lines.append(f"Summary:")
    lines.append(f"  • Most deviant digit: {z_result['most_deviant'][0]} (Z = {z_result['most_deviant'][1]:+.4f})")
    lines.append(f"  • Significant digits (|Z| > 1.96): {z_result['significant_digits'] or 'None'}")
    lines.append("")

    # Interpretation
    lines.append("Interpretation:")
    if z_result['significant_digits']:
        for d in z_result['significant_digits']:
            z = z_result['z_scores'][d]
            if z > 0:
                lines.append(f"  • Digit {d} is OVER-represented (+{z:.2f}σ)")
                if d == 5:
                    lines.append("    └─ Common pattern: rounding to 'nice' numbers")
                elif d == 9:
                    lines.append("    └─ Common pattern: just-below-threshold pricing")
            else:
                lines.append(f"  • Digit {d} is UNDER-represented ({z:.2f}σ)")
                if d == 1:
                    lines.append("    └─ Classic fraud indicator: numbers being inflated")
    else:
        lines.append("  • No digits show statistically significant deviation")
        lines.append("  • Data appears to conform well to Benford's Law")
    lines.append("")

    # =====================================================================
    # SECTION 3: ANOMALY SCORE
    # =====================================================================
    lines.append("-" * 80)
    lines.append("SECTION 3: COMPOSITE ANOMALY SCORE")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Purpose: Single score (0-100) combining all metrics for easy ranking")
    lines.append("")

    anomaly = calculate_anomaly_score(data)

    # Risk level visualization
    score = anomaly['anomaly_score']
    risk = anomaly['risk_level']

    if risk == 'low':
        risk_bar = "🟢" * int(score/10) + "⬜" * (10 - int(score/10))
        risk_color = "LOW"
    elif risk == 'medium':
        risk_bar = "🟡" * int(score/10) + "⬜" * (10 - int(score/10))
        risk_color = "MEDIUM"
    elif risk == 'high':
        risk_bar = "🟠" * int(score/10) + "⬜" * (10 - int(score/10))
        risk_color = "HIGH"
    else:
        risk_bar = "🔴" * int(score/10) + "⬜" * (10 - int(score/10))
        risk_color = "CRITICAL"

    lines.append(f"ANOMALY SCORE: {score:.2f} / 100")
    lines.append(f"RISK LEVEL:    {risk_color}")
    lines.append(f"Score Bar:     [{risk_bar}]")
    lines.append(f"               0        25       50       75      100")
    lines.append(f"               |   Low   | Medium |  High  |Critical|")
    lines.append("")

    lines.append("Component Breakdown:")
    lines.append("-" * 50)
    lines.append("Component       Raw Value   Normalized   Weight   Contrib")
    for comp, norm_score in anomaly['component_scores'].items():
        raw = anomaly['metrics'].get(comp, 0) if comp != 'KS_test' else anomaly['metrics'].get('KS_test', 0)
        weight = 0.4 if comp in ['chi_square', 'MAD'] else 0.2
        contrib = norm_score * weight
        lines.append(f"{comp:15} {raw:8.4f}    {norm_score:6.2f}       {weight:.0%}     {contrib:6.2f}")
    lines.append(f"{'':15} {'':8}    {'':6}       {'':4}     ------")
    lines.append(f"{'TOTAL':15} {'':8}    {'':6}       {'':4}     {score:6.2f}")
    lines.append("")

    lines.append("Risk Level Thresholds:")
    lines.append("  • 0-25:   LOW      - Likely conforming, no action needed")
    lines.append("  • 25-50:  MEDIUM   - Monitor, review if resources allow")
    lines.append("  • 50-75:  HIGH     - Should be investigated")
    lines.append("  • 75-100: CRITICAL - Immediate investigation recommended")
    lines.append("")

    # =====================================================================
    # SECTION 4: DIGIT DISTRIBUTION
    # =====================================================================
    lines.append("-" * 80)
    lines.append("SECTION 4: DIGIT FREQUENCY DISTRIBUTION")
    lines.append("-" * 80)
    lines.append("")
    lines.append("Purpose: Raw data for visualization and detailed pattern analysis")
    lines.append("")

    dist = get_digit_distribution(data)

    lines.append("Digit  Count    Observed%   Expected%   Difference   Visual Comparison")
    lines.append("-" * 80)

    for i, d in enumerate(dist['digits']):
        count = dist['counts'][i]
        obs = dist['observed'][i]
        exp = dist['expected'][i]
        diff = obs - exp

        # Create simple bar visualization
        obs_bar = "█" * int(obs / 2)
        exp_bar = "░" * int(exp / 2)

        lines.append(f"  {d}     {count:5}    {obs:6.2f}%     {exp:5.2f}%     {diff:+6.2f}%    {obs_bar}")
        lines.append(f"         {'':5}    {'':6}      {'':5}      {'':6}     {exp_bar} (expected)")

    lines.append("")
    lines.append(f"Total samples: {dist['n_samples']}")
    lines.append(f"Checksum: Observed sum = {sum(dist['observed']):.2f}% (should be ~100%)")
    lines.append("")

    # =====================================================================
    # SECTION 5: CONCLUSION
    # =====================================================================
    lines.append("-" * 80)
    lines.append("SECTION 5: OVERALL ASSESSMENT")
    lines.append("-" * 80)
    lines.append("")

    # Determine overall assessment
    issues = []
    if not interp['chi_square']['conforms']:
        issues.append("Chi-square exceeds threshold")
    if interp['p_value']['significant']:
        issues.append("Statistically significant deviation detected")
    if not interp['MAD']['conforms']:
        issues.append("MAD exceeds acceptable threshold")
    if z_result['significant_digits']:
        issues.append(f"Significant digit deviations: {z_result['significant_digits']}")

    if not issues:
        lines.append("✅ ASSESSMENT: DATA APPEARS TO CONFORM TO BENFORD'S LAW")
        lines.append("")
        lines.append("   All metrics are within acceptable ranges. No significant")
        lines.append("   digit-level deviations detected. This data pattern is")
        lines.append("   consistent with naturally occurring financial figures.")
    elif len(issues) <= 2 and risk in ['low', 'medium']:
        lines.append("⚠️  ASSESSMENT: MINOR CONCERNS DETECTED")
        lines.append("")
        lines.append("   Some metrics show deviation from expected patterns:")
        for issue in issues:
            lines.append(f"   • {issue}")
        lines.append("")
        lines.append("   Recommendation: Monitor in future periods, but likely")
        lines.append("   within normal business variation.")
    else:
        lines.append("🚨 ASSESSMENT: SIGNIFICANT DEVIATION FROM BENFORD'S LAW")
        lines.append("")
        lines.append("   Multiple indicators suggest non-conformance:")
        for issue in issues:
            lines.append(f"   • {issue}")
        lines.append("")
        lines.append("   Recommendation: Further investigation warranted.")
        lines.append("   Consider detailed review of source data and accounting practices.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Generate sample report using test data."""

    print("Generating Benford Analysis Report...")
    print("")

    # Generate sample data (lognormal - follows Benford's Law)
    np.random.seed(42)
    sample_data = pd.Series(np.random.lognormal(mean=10, sigma=3, size=1000))

    # Generate report
    report = generate_report(sample_data, "Test Company (Lognormal Data)")

    # Save to file
    output_file = Path("analysis_report.txt")
    output_file.write_text(report)

    print(f"Report saved to: {output_file.absolute()}")
    print("")
    print("Preview (first 50 lines):")
    print("-" * 40)
    for line in report.split("\n")[:50]:
        print(line)
    print("...")
    print(f"\n[Full report: {len(report.split(chr(10)))} lines]")


if __name__ == "__main__":
    main()
