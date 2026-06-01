"""Feature engineering.

    financial_ratios — derive standard ratios (ROA, ROE, current ratio, accruals, ...)
    normalization    — cross-sectional standardization per quarter / per sector
    sequence_builder — produce per-firm tensors of shape (n_quarters, n_features)
                       with masking for missing quarters
"""
