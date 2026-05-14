"""Cross-sectional signal transforms for the Study pipeline.

All functions share the signature:
    (signal: pd.DataFrame, ...) -> pd.DataFrame

They operate row-wise (cross-sectionally per date) and preserve the input shape.
NaN values are excluded from all cross-sectional calculations and propagated
in the output unless the transform explicitly handles them (e.g. truncate).

Intended use — via Study convenience methods::

    study = (
        Study(universe=universe, benchmark=benchmark)
        .base_signal(my_signal_fn)
        .winsorize(lower=0.05, upper=0.95)
        .zscore_signal()
        .build_long_short(n_long=25, n_short=25)
        .run()
    )

Or applied directly to a signal DataFrame::

    import qstudy as qs
    transformed = qs.winsorize(signal_df, lower=0.05, upper=0.95)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(signal: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
    """Clip cross-sectional outliers to percentile bounds on each date.

    Values above the upper percentile are clipped down; values below the lower
    percentile are clipped up. Outliers are retained but their influence is capped.
    Use when you want to consider outliers but not overweight them.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.
        lower:  Lower percentile bound (e.g. 0.05 = 5th percentile).
        upper:  Upper percentile bound (e.g. 0.95 = 95th percentile).

    Returns:
        Winsorized signal, same shape.
    """
    lo = signal.quantile(lower, axis=1)
    hi = signal.quantile(upper, axis=1)
    return signal.clip(lower=lo, upper=hi, axis=0)


def truncate(signal: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
    """Remove cross-sectional outliers by setting them to NaN on each date.

    Values outside [lower, upper] percentile bounds become NaN (ineligible).
    Use when outliers are likely data errors rather than genuine extreme signals.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.
        lower:  Lower percentile bound — values below become NaN.
        upper:  Upper percentile bound — values above become NaN.

    Returns:
        Truncated signal, same shape. Out-of-bounds values replaced with NaN.
    """
    lo = signal.quantile(lower, axis=1)
    hi = signal.quantile(upper, axis=1)
    mask = signal.ge(lo, axis=0) & signal.le(hi, axis=0)
    return signal.where(mask)


def rank_transform(signal: pd.DataFrame) -> pd.DataFrame:
    """Rank signal cross-sectionally and normalize to [0, 1] on each date.

    Produces a uniform distribution. NaN values are excluded from ranking and
    remain NaN in the output. Use to remove distributional shape, fix skew,
    eliminate outliers, or when the precise signal value has no meaning beyond order.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.

    Returns:
        Ranked signal in [0, 1], same shape. NaN where input is NaN.
    """
    ranked = signal.rank(axis=1, na_option="keep")
    counts = signal.notna().sum(axis=1)
    return ranked.div(counts, axis=0)


def rank_threshold(signal: pd.DataFrame, tail: float = 0.20) -> pd.DataFrame:
    """Rank cross-sectionally then zero out the middle, keeping only the tails.

    First applies :func:`rank_transform`, then sets values in the middle
    ``(1 - 2 * tail)`` fraction to zero (NaN). Concentrates signal fully in the
    top and bottom ``tail`` fraction. Use when you believe the signal is only
    meaningful at extremes.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.
        tail:   Fraction to keep on each end. Default 0.20 keeps the top 20%
                and bottom 20%, zeros out the middle 60%.

    Returns:
        Thresholded signal, same shape. Middle values set to NaN.
    """
    r = rank_transform(signal)
    return r.where((r <= tail) | (r >= 1.0 - tail))


def inverse_cdf(signal: pd.DataFrame) -> pd.DataFrame:
    """Map signal to standard normal quantiles via the inverse CDF on each date.

    First ranks cross-sectionally (percentile), then applies the standard normal
    inverse CDF (probit). Produces a normal distribution. Tails receive more
    weight than the center compared to a rank transform. Use for the same reasons
    as :func:`rank_transform` but when you want modest weight in the center and
    heavier tails.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.

    Returns:
        Normal-mapped signal, same shape. NaN where input is NaN.
    """
    from scipy.stats import norm

    percentiles = rank_transform(signal)
    # Clip away exact 0/1 to avoid ±inf from norm.ppf
    clipped = percentiles.clip(lower=1e-6, upper=1 - 1e-6)
    result = clipped.apply(lambda row: pd.Series(norm.ppf(row.values), index=row.index), axis=1)
    return result.where(signal.notna())


def tanh_scale(signal: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Soft-clip the signal to (-1, 1) using tanh on each date.

    Applies ``tanh(signal / scale)`` element-wise. Analogous to winsorization
    but smooth and differentiable. Large values are compressed; the sign and
    rough ordering are preserved. NaN values remain NaN.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.
        scale:  Controls the inflection point. Values near ±scale are compressed
                toward ±1. Smaller scale = more aggressive compression.

    Returns:
        Tanh-scaled signal in (-1, 1), same shape.
    """
    return np.tanh(signal / scale)


def zscore(signal: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: subtract mean and divide by std on each date.

    NaN values are excluded from the mean and std calculation and remain NaN
    in the output. Produces a roughly standard-normal distribution per row.
    Use to normalize signal scale before position building.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.

    Returns:
        Z-scored signal, same shape. NaN where input is NaN.
    """
    mean = signal.mean(axis=1)
    std = signal.std(axis=1).replace(0.0, float("nan"))
    return signal.sub(mean, axis=0).div(std, axis=0)


def demean(signal: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional mean on each date.

    Shifts signal values so each row sums to approximately zero. NaN values
    are excluded from the mean calculation and remain NaN in the output.
    First step toward dollar neutrality when used before a proportional
    position builder.

    Args:
        signal: Signal DataFrame (dates x tickers). NaN = ineligible.

    Returns:
        Demeaned signal, same shape. NaN where input is NaN.
    """
    mean = signal.mean(axis=1)
    return signal.sub(mean, axis=0)
