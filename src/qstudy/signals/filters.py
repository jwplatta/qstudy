import pandas as pd


def vix_contango_filter(
    signal: pd.DataFrame,
    vix_close: pd.DataFrame,
    window: int = 1,
) -> pd.DataFrame:
    """Zero out all signals on dates where the VIX term structure is not in full contango.

    Checks adjacent pairs among whichever of ["^VIX1D", "^VIX9D", "^VIX"] are present.
    Any inversion zeroes out the entire cross-section that day.

    Args:
        signal:    Raw signal (dates x tickers).
        vix_close: Close prices for VIX indexes aligned to signal index.
                   Download with: qs.download(VOL_INDEXES, ...)["close"]
        window:    Require contango for N consecutive days before allowing trades.
                   Default 1 (no rolling — react on the same day).

    Returns:
        Filtered signal with NaN on all backwardation dates.
    """
    vix = vix_close.reindex(signal.index)
    cols = [c for c in ["^VIX1D", "^VIX9D", "^VIX"] if c in vix.columns]
    in_contango = pd.Series(True, index=signal.index)
    for a, b in zip(cols[:-1], cols[1:]):
        in_contango &= vix[a] < vix[b]
    if window > 1:
        in_contango = in_contango.rolling(window, min_periods=window).min().astype(bool)
    return signal.where(in_contango, other=float("nan"))


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

    The threshold is the Nth cross-sectional quantile of the raw momentum values (not their
    absolute values), consistent with: signal.where(mom.abs() < mom.quantile(q, axis=1)).

    Args:
        signal:           Raw signal (dates x tickers).
        returns:          Daily returns (dates x tickers).
        window:           Lookback for medium-term momentum.
        max_abs_quantile: Cross-sectional quantile of raw momentum used as abs-magnitude cap.

    Returns:
        Filtered signal.
    """
    long_mom = returns.rolling(window).mean()
    thresh = long_mom.quantile(max_abs_quantile, axis=1)
    mask = long_mom.abs().lt(thresh, axis=0)
    return signal.where(mask)
