import pandas as pd


def vol_filter(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    vol_window: int = 40,
    quantile: float = 0.75,
    keep: str = "low",
) -> pd.DataFrame:
    """Zero out signal where realized vol is above (keep='low') or below (keep='high') the
    cross-sectional quantile across tickers on each date.

    The threshold is computed cross-sectionally: on each date, keep only assets whose realized vol
    is below (or above) the Nth percentile of all assets that day.

    Args:
        signal:     Raw signal (dates x tickers). NaN = no signal.
        returns:    Daily returns (dates x tickers).
        vol_window: Lookback for realized vol calculation.
        quantile:   Cross-sectional threshold percentile (e.g. 0.75 keeps assets below 75th pct).
        keep:       'low' keeps assets below the quantile; 'high' keeps assets above.

    Returns:
        Filtered signal with NaN where the condition is not met.
    """
    realized_vol = returns.rolling(vol_window).std()
    vol_thresh = realized_vol.quantile(quantile, axis=1)
    if keep == "low":
        mask = realized_vol.lt(vol_thresh, axis=0)
    else:
        mask = realized_vol.gt(vol_thresh, axis=0)
    return signal.where(mask)


def volume_zscore_filter(
    signal: pd.DataFrame,
    volume: pd.DataFrame,
    window: int = 10,
    min_zscore_quantile: float = 0.65,
) -> pd.DataFrame:
    """Zero out signal where the volume z-score is below the cross-sectional quantile threshold.

    Keeps assets experiencing above-average volume activity.

    Args:
        signal:              Raw signal (dates x tickers).
        volume:              Daily volume (dates x tickers).
        window:              Lookback for rolling mean/std of volume.
        min_zscore_quantile: Cross-sectional quantile threshold (keep assets above this).

    Returns:
        Filtered signal.
    """
    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std()
    vol_z = (volume - vol_mean) / vol_std
    thresh = vol_z.quantile(min_zscore_quantile, axis=1)
    mask = vol_z.ge(thresh, axis=0)
    return signal.where(mask)


def momentum_context_filter(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 15,
    max_abs_quantile: float = 0.75,
) -> pd.DataFrame:
    """Zero out signal where the medium-term momentum magnitude is above the cross-sectional
    quantile.

    Filters out strongly trending assets when trading mean reversion.

    Args:
        signal:           Raw signal (dates x tickers).
        returns:          Daily returns (dates x tickers).
        window:           Lookback for medium-term momentum.
        max_abs_quantile: Cross-sectional quantile cap on absolute momentum magnitude.

    Returns:
        Filtered signal.
    """
    long_mom = returns.rolling(window).mean()
    thresh = long_mom.abs().quantile(max_abs_quantile, axis=1)
    mask = long_mom.abs().le(thresh, axis=0)
    return signal.where(mask)
