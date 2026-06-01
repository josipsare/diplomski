"""Impact of detected anomalies on business outcomes (strict NN-only).

    forward_outcome   — NN MLP regressor that predicts next-year stock outcomes
                        (return, volatility, max drawdown, volume growth) from
                        the anomaly score plus controls. Headline metric:
                        out-of-sample delta-R² over a controls-only baseline.

Classical-statistics impact methods (event study with CAR/BHAR, propensity
matching, abnormal returns under CAPM) are in `legacy/classical_impact/` for
reference but are NOT part of the primary master's-thesis pipeline.
"""
