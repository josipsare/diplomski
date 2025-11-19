"""
Benford's Law Statistical Metrics Calculator
Calculates multiple statistical tests for conformance to Benford's Law
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest


# Benford's Law expected distribution (percentages)
BENFORD_EXPECTED = {
    1: 30.1, 2: 17.6, 3: 12.5, 4: 9.7, 5: 7.9,
    6: 6.7, 7: 5.8, 8: 5.1, 9: 4.5
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


def interpret_results(metrics):
    """
    Interpret Benford's Law metrics

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


if __name__ == '__main__':
    # Example usage
    print("Benford's Law Metrics Calculator")
    print("="*60)

    # Test with random data
    np.random.seed(42)
    test_data = pd.Series(np.random.lognormal(mean=5, sigma=2, size=1000))

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
