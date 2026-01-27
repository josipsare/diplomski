"""
Benford's Law Statistical Metrics Calculator
Calculates multiple statistical tests for conformance to Benford's Law
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest


# Benford's Law expected distribution (percentages) - First Digit
BENFORD_EXPECTED = {
    1: 30.1, 2: 17.6, 3: 12.5, 4: 9.7, 5: 7.9,
    6: 6.7, 7: 5.8, 8: 5.1, 9: 4.5
}

# Benford's Law expected distribution (percentages) - Second Digit
# Source: Nigrini (2012) - digits 0-9
BENFORD_SECOND_DIGIT_EXPECTED = {
    0: 11.97, 1: 11.39, 2: 10.88, 3: 10.43, 4: 10.03,
    5: 9.67, 6: 9.34, 7: 9.04, 8: 8.76, 9: 8.50
}


def get_first_digit(number):
    """
    Extract first significant digit from a number

    Args:
        number: Numerical value

    Returns:
        int: First significant digit (1-9) or None
    """
    if pd.isna(number) or number == 0:
        return None

    # Convert to string and extract first non-zero digit
    for char in str(int(abs(number))):
        if char != '0':
            return int(char)

    return None


def get_second_digit(number):
    """
    Extract second significant digit from a number.

    Args:
        number: Numerical value

    Returns:
        int: Second significant digit (0-9) or None if number has only 1 digit
    """
    if pd.isna(number) or number == 0:
        return None

    # Get absolute value and convert to string of digits
    abs_str = str(int(abs(number)))

    # Remove leading zeros and find significant digits
    significant = abs_str.lstrip('0')

    # Need at least 2 significant digits
    if len(significant) < 2:
        return None

    return int(significant[1])


def benford_cdf(x):
    """
    Cumulative distribution function for Benford's Law
    Vectorized to handle both scalars and arrays

    Args:
        x: Value (digit) - can be scalar or array

    Returns:
        float or array: Cumulative probability
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # For x < 1, result is 0 (already initialized)
    # For x >= 9, result is 1
    result[x >= 9] = 1.0

    # For 1 <= x < 9, use Benford's formula
    mask = (x >= 1) & (x < 9)
    result[mask] = np.log10(1 + 1/np.floor(x[mask]))

    return result if result.shape else float(result)


def benford_second_digit_cdf(x):
    """
    Cumulative distribution function for second-digit Benford's Law.
    Uses pre-computed expected values for efficiency.

    Args:
        x: Value (digit) - can be scalar or array

    Returns:
        float or array: Cumulative probability
    """
    x = np.asarray(x)

    # Pre-compute cumulative probabilities
    cumulative_probs = np.cumsum([BENFORD_SECOND_DIGIT_EXPECTED[i] / 100 for i in range(10)])

    result = np.zeros_like(x, dtype=float)

    # Handle edge cases
    result[x < 0] = 0.0
    result[x >= 9] = 1.0

    # For each digit 0-8, assign cumulative probability
    for d in range(0, 9):
        mask = (x >= d) & (x < d + 1)
        result[mask] = cumulative_probs[d]

    return result if result.shape else float(result)


def calculate_benford_metrics(numbers_series):
    """
    Calculate comprehensive Benford's Law metrics for a series of numbers

    Args:
        numbers_series: pandas Series of numerical values

    Returns:
        dict: Dictionary containing:
            - chi_square: Chi-square test statistic
            - p_value: P-value for chi-square test
            - MAD: Mean Absolute Deviation
            - KS_test: Kolmogorov-Smirnov test statistic
            - n_samples: Number of samples analyzed
    """
    # Handle empty or invalid input
    if numbers_series is None or len(numbers_series) == 0:
        return {
            'chi_square': 0,
            'p_value': 0,
            'MAD': 0,
            'KS_test': 0,
            'n_samples': 0
        }

    # Create DataFrame for processing
    df = pd.DataFrame({'value': numbers_series})
    df['abs_value'] = df['value'].abs()

    # Extract first digits
    df['first_digit'] = df['abs_value'].apply(get_first_digit)
    df = df.dropna(subset=['first_digit'])

    # If no valid digits found
    if len(df) == 0:
        return {
            'chi_square': 0,
            'p_value': 0,
            'MAD': 0,
            'KS_test': 0,
            'n_samples': 0
        }

    n_samples = len(df)

    # Count frequency of each digit
    digit_counts = df['first_digit'].value_counts()

    # Calculate observed frequencies (percentages)
    observed_freq = []
    observed_counts = []
    expected_freq = []

    for digit in range(1, 10):
        count = digit_counts.get(digit, 0)
        observed_counts.append(count)
        observed_freq.append((count / n_samples) * 100)
        expected_freq.append(BENFORD_EXPECTED[digit])

    observed_freq = np.array(observed_freq)
    expected_freq = np.array(expected_freq)

    # 1. Chi-Square Test
    expected_counts = [(n_samples * BENFORD_EXPECTED[i] / 100) for i in range(1, 10)]
    chi_square = sum((o - e)**2 / e for o, e in zip(observed_counts, expected_counts))

    # 2. P-Value (chi-square distribution with 8 degrees of freedom)
    p_value = 1 - chi2.cdf(chi_square, df=8)

    # 3. MAD - Mean Absolute Deviation
    # Industry standard: MAD < 0.015 indicates conformity
    mad = np.mean(np.abs(observed_freq - expected_freq))

    # 4. Kolmogorov-Smirnov Test
    # Test if first digits follow Benford's distribution
    first_digits = df['first_digit'].values
    ks_statistic, ks_pvalue = kstest(first_digits, benford_cdf)

    return {
        'chi_square': round(chi_square, 4),
        'p_value': round(p_value, 4),
        'MAD': round(mad, 4),
        'KS_test': round(ks_statistic, 4),
        'n_samples': n_samples
    }


def calculate_second_digit_benford_metrics(numbers_series):
    """
    Calculate comprehensive second-digit Benford's Law metrics for a series of numbers.

    Second-digit analysis requires numbers with at least 2 significant digits.
    Uses digits 0-9 (chi-square has 9 degrees of freedom).

    Args:
        numbers_series: pandas Series of numerical values

    Returns:
        dict: Dictionary containing:
            - chi_square: Chi-square test statistic
            - p_value: P-value for chi-square test
            - MAD: Mean Absolute Deviation
            - KS_test: Kolmogorov-Smirnov test statistic
            - n_samples: Number of samples analyzed
    """
    # Handle empty or invalid input
    if numbers_series is None or len(numbers_series) == 0:
        return {
            'chi_square': 0,
            'p_value': 0,
            'MAD': 0,
            'KS_test': 0,
            'n_samples': 0
        }

    # Create DataFrame for processing
    df = pd.DataFrame({'value': numbers_series})
    df['abs_value'] = df['value'].abs()

    # Extract second digits (filters numbers with <2 significant digits)
    df['second_digit'] = df['abs_value'].apply(get_second_digit)
    df = df.dropna(subset=['second_digit'])

    # If no valid digits found
    if len(df) == 0:
        return {
            'chi_square': 0,
            'p_value': 0,
            'MAD': 0,
            'KS_test': 0,
            'n_samples': 0
        }

    n_samples = len(df)

    # Count frequency of each digit (0-9)
    digit_counts = df['second_digit'].value_counts()

    # Calculate observed frequencies (percentages)
    observed_freq = []
    observed_counts = []
    expected_freq = []

    for digit in range(0, 10):  # 0-9 for second digit
        count = digit_counts.get(digit, 0)
        observed_counts.append(count)
        observed_freq.append((count / n_samples) * 100)
        expected_freq.append(BENFORD_SECOND_DIGIT_EXPECTED[digit])

    observed_freq = np.array(observed_freq)
    expected_freq = np.array(expected_freq)

    # 1. Chi-Square Test
    expected_counts = [(n_samples * BENFORD_SECOND_DIGIT_EXPECTED[i] / 100) for i in range(0, 10)]
    chi_square = sum((o - e)**2 / e for o, e in zip(observed_counts, expected_counts) if e > 0)

    # 2. P-Value (chi-square distribution with 9 degrees of freedom)
    # 10 categories - 1 = 9 df
    p_value = 1 - chi2.cdf(chi_square, df=9)

    # 3. MAD - Mean Absolute Deviation
    mad = np.mean(np.abs(observed_freq - expected_freq))

    # 4. Kolmogorov-Smirnov Test
    second_digits = df['second_digit'].values
    ks_statistic, ks_pvalue = kstest(second_digits, benford_second_digit_cdf)

    return {
        'chi_square': round(chi_square, 4),
        'p_value': round(p_value, 4),
        'MAD': round(mad, 4),
        'KS_test': round(ks_statistic, 4),
        'n_samples': n_samples
    }


def interpret_results(metrics):
    """
    Interpret first-digit Benford's Law metrics

    Args:
        metrics: Dictionary from calculate_benford_metrics

    Returns:
        dict: Interpretation of each metric
    """
    interpretations = {}

    # Chi-square interpretation (8 df, α=0.05: critical value = 15.507)
    interpretations['chi_square'] = {
        'value': metrics['chi_square'],
        'conforms': metrics['chi_square'] < 15.507,
        'note': 'Lower values indicate better conformance'
    }

    # P-value interpretation
    interpretations['p_value'] = {
        'value': metrics['p_value'],
        'significant': metrics['p_value'] < 0.05,
        'note': 'p < 0.05 suggests significant deviation from Benford\'s Law'
    }

    # MAD interpretation (Nigrini, 2012)
    mad_value = metrics['MAD']
    if mad_value < 0.006:
        mad_level = 'Close conformity'
    elif mad_value < 0.012:
        mad_level = 'Acceptable conformity'
    elif mad_value < 0.015:
        mad_level = 'Marginally acceptable'
    else:
        mad_level = 'Nonconformity'

    interpretations['MAD'] = {
        'value': mad_value,
        'level': mad_level,
        'conforms': mad_value < 0.015,
        'note': 'Industry standard: MAD < 0.015'
    }

    # KS test interpretation
    interpretations['KS_test'] = {
        'value': metrics['KS_test'],
        'note': 'Lower values indicate better fit to Benford\'s distribution'
    }

    return interpretations


def calculate_digit_zscores(numbers_series):
    """
    Calculate Z-scores for each digit showing which digits deviate most.

    Z-score formula: Z = (observed - expected) / sqrt(expected * (1 - expected) / n)

    Args:
        numbers_series: pandas Series of numerical values

    Returns:
        dict: Dictionary containing:
            - z_scores: Dict of Z-score per digit (1-9)
            - significant_digits: List of digits with |Z| > 1.96 (p < 0.05)
            - most_deviant: Tuple of (digit, z_score) for highest absolute Z
            - digit_analysis: List of dicts with full analysis per digit
    """
    if numbers_series is None or len(numbers_series) == 0:
        return {
            'z_scores': {},
            'significant_digits': [],
            'most_deviant': (None, 0),
            'digit_analysis': []
        }

    # Extract first digits
    df = pd.DataFrame({'value': numbers_series})
    df['first_digit'] = df['value'].abs().apply(get_first_digit)
    df = df.dropna(subset=['first_digit'])

    if len(df) == 0:
        return {
            'z_scores': {},
            'significant_digits': [],
            'most_deviant': (None, 0),
            'digit_analysis': []
        }

    n = len(df)
    digit_counts = df['first_digit'].value_counts()

    z_scores = {}
    digit_analysis = []
    significant_digits = []

    for digit in range(1, 10):
        observed_count = digit_counts.get(digit, 0)
        observed_prop = observed_count / n
        expected_prop = BENFORD_EXPECTED[digit] / 100

        # Z-score calculation (with continuity correction for large samples)
        std_error = np.sqrt(expected_prop * (1 - expected_prop) / n)

        if std_error > 0:
            z = (observed_prop - expected_prop) / std_error
        else:
            z = 0

        z_scores[digit] = round(z, 4)

        # Determine significance and direction
        if abs(z) > 2.576:
            significance = 'highly_significant'  # p < 0.01
        elif abs(z) > 1.96:
            significance = 'significant'  # p < 0.05
        else:
            significance = 'not_significant'

        direction = 'over' if z > 0 else 'under' if z < 0 else 'normal'

        if abs(z) > 1.96:
            significant_digits.append(digit)

        digit_analysis.append({
            'digit': digit,
            'observed_pct': round(observed_prop * 100, 2),
            'expected_pct': BENFORD_EXPECTED[digit],
            'difference_pct': round((observed_prop - expected_prop) * 100, 2),
            'z_score': round(z, 4),
            'significance': significance,
            'direction': direction
        })

    # Find most deviant digit
    most_deviant_digit = max(z_scores.keys(), key=lambda d: abs(z_scores[d]))
    most_deviant = (most_deviant_digit, z_scores[most_deviant_digit])

    return {
        'z_scores': z_scores,
        'significant_digits': significant_digits,
        'most_deviant': most_deviant,
        'digit_analysis': digit_analysis,
        'n_samples': n
    }


def calculate_anomaly_score(numbers_series, weights=None):
    """
    Calculate a composite anomaly score combining multiple metrics.

    Score ranges from 0 (perfect conformance) to 100 (extreme deviation).
    Uses normalized and weighted combination of Chi-square, MAD, and KS test.

    Args:
        numbers_series: pandas Series of numerical values
        weights: Optional dict with keys 'chi_square', 'MAD', 'KS_test'
                 Default: {'chi_square': 0.4, 'MAD': 0.4, 'KS_test': 0.2}

    Returns:
        dict: Dictionary containing:
            - anomaly_score: Composite score 0-100
            - risk_level: 'low', 'medium', 'high', 'critical'
            - component_scores: Individual normalized scores
            - metrics: Original metric values
    """
    if weights is None:
        weights = {'chi_square': 0.4, 'MAD': 0.4, 'KS_test': 0.2}

    metrics = calculate_benford_metrics(numbers_series)

    if metrics['n_samples'] == 0:
        return {
            'anomaly_score': 0,
            'risk_level': 'insufficient_data',
            'component_scores': {},
            'metrics': metrics
        }

    # Normalize each metric to 0-100 scale
    # Chi-square: 0 is perfect, ~30+ is very bad (8 df)
    chi_norm = min(100, (metrics['chi_square'] / 30) * 100)

    # MAD: 0 is perfect, 3+ is very bad
    mad_norm = min(100, (metrics['MAD'] / 3) * 100)

    # KS test: 0 is perfect, 0.3+ is very bad
    ks_norm = min(100, (metrics['KS_test'] / 0.3) * 100)

    component_scores = {
        'chi_square': round(chi_norm, 2),
        'MAD': round(mad_norm, 2),
        'KS_test': round(ks_norm, 2)
    }

    # Calculate weighted composite score
    anomaly_score = (
        weights['chi_square'] * chi_norm +
        weights['MAD'] * mad_norm +
        weights['KS_test'] * ks_norm
    )
    anomaly_score = round(min(100, anomaly_score), 2)

    # Determine risk level
    if anomaly_score < 25:
        risk_level = 'low'
    elif anomaly_score < 50:
        risk_level = 'medium'
    elif anomaly_score < 75:
        risk_level = 'high'
    else:
        risk_level = 'critical'

    return {
        'anomaly_score': anomaly_score,
        'risk_level': risk_level,
        'component_scores': component_scores,
        'metrics': metrics
    }


def get_digit_distribution(numbers_series):
    """
    Get observed vs expected digit distribution for visualization.

    Args:
        numbers_series: pandas Series of numerical values

    Returns:
        dict: Dictionary containing:
            - digits: List [1, 2, ..., 9]
            - observed: List of observed percentages
            - expected: List of expected percentages (Benford)
            - counts: List of raw counts per digit
            - n_samples: Total sample size
    """
    if numbers_series is None or len(numbers_series) == 0:
        return {
            'digits': list(range(1, 10)),
            'observed': [0] * 9,
            'expected': [BENFORD_EXPECTED[d] for d in range(1, 10)],
            'counts': [0] * 9,
            'n_samples': 0
        }

    df = pd.DataFrame({'value': numbers_series})
    df['first_digit'] = df['value'].abs().apply(get_first_digit)
    df = df.dropna(subset=['first_digit'])

    if len(df) == 0:
        return {
            'digits': list(range(1, 10)),
            'observed': [0] * 9,
            'expected': [BENFORD_EXPECTED[d] for d in range(1, 10)],
            'counts': [0] * 9,
            'n_samples': 0
        }

    n = len(df)
    digit_counts = df['first_digit'].value_counts()

    observed = []
    counts = []
    expected = []

    for digit in range(1, 10):
        count = digit_counts.get(digit, 0)
        counts.append(count)
        observed.append(round((count / n) * 100, 2))
        expected.append(BENFORD_EXPECTED[digit])

    return {
        'digits': list(range(1, 10)),
        'observed': observed,
        'expected': expected,
        'counts': counts,
        'n_samples': n
    }


def interpret_second_digit_results(metrics):
    """
    Interpret second-digit Benford's Law metrics.

    Args:
        metrics: Dictionary from calculate_second_digit_benford_metrics

    Returns:
        dict: Interpretation of each metric
    """
    interpretations = {}

    # Chi-square interpretation (9 df, α=0.05: critical value = 16.919)
    interpretations['chi_square'] = {
        'value': metrics['chi_square'],
        'conforms': metrics['chi_square'] < 16.919,
        'note': 'Lower values indicate better conformance (9 df for second digit)'
    }

    # P-value interpretation
    interpretations['p_value'] = {
        'value': metrics['p_value'],
        'significant': metrics['p_value'] < 0.05,
        'note': 'p < 0.05 suggests significant deviation from Benford\'s Law'
    }

    # MAD interpretation for second digit (Nigrini, 2012)
    # Thresholds are slightly different for second digit
    mad_value = metrics['MAD']
    if mad_value < 0.008:
        mad_level = 'Close conformity'
    elif mad_value < 0.010:
        mad_level = 'Acceptable conformity'
    elif mad_value < 0.012:
        mad_level = 'Marginally acceptable'
    else:
        mad_level = 'Nonconformity'

    interpretations['MAD'] = {
        'value': mad_value,
        'level': mad_level,
        'conforms': mad_value < 0.012,
        'note': 'Second-digit standard: MAD < 0.012'
    }

    # KS test interpretation
    interpretations['KS_test'] = {
        'value': metrics['KS_test'],
        'note': 'Lower values indicate better fit to second-digit Benford distribution'
    }

    return interpretations


if __name__ == '__main__':
    # Example usage
    print("Benford's Law Metrics Calculator")
    print("="*60)

    # Test with random data
    np.random.seed(42)
    test_data = pd.Series(np.random.lognormal(mean=5, sigma=2, size=1000))

    # First-digit analysis
    print("\n--- FIRST-DIGIT ANALYSIS ---")
    metrics = calculate_benford_metrics(test_data)

    print(f"\nTest Data (n={metrics['n_samples']:,}):")
    print(f"Chi-Square: {metrics['chi_square']:.4f}")
    print(f"P-Value: {metrics['p_value']:.4f}")
    print(f"MAD: {metrics['MAD']:.4f}")
    print(f"KS Test: {metrics['KS_test']:.4f}")

    print("\nInterpretation:")
    interp = interpret_results(metrics)
    for test_name, result in interp.items():
        print(f"\n{test_name.upper()}:")
        for key, value in result.items():
            print(f"  {key}: {value}")

    # Second-digit analysis
    print("\n" + "="*60)
    print("\n--- SECOND-DIGIT ANALYSIS ---")
    metrics_d2 = calculate_second_digit_benford_metrics(test_data)

    print(f"\nTest Data (n={metrics_d2['n_samples']:,}):")
    print(f"Chi-Square: {metrics_d2['chi_square']:.4f}")
    print(f"P-Value: {metrics_d2['p_value']:.4f}")
    print(f"MAD: {metrics_d2['MAD']:.4f}")
    print(f"KS Test: {metrics_d2['KS_test']:.4f}")

    print("\nInterpretation:")
    interp_d2 = interpret_second_digit_results(metrics_d2)
    for test_name, result in interp_d2.items():
        print(f"\n{test_name.upper()}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
