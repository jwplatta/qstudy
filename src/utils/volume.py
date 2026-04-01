"""
Volume data loading and processing utilities for /ES futures data.

This module provides functions to load and process /ES 5-minute candle data
for calculating dollar volume metrics used in gamma influence calculations.
"""

from pathlib import Path
from datetime import datetime, time
import pandas as pd
import numpy as np


def load_es_volume(date, data_dir="data"):
    """
    Load /ES 5-minute candle data for a specific date.

    Parameters
    ----------
    date : str or datetime
        Date in YYYY-MM-DD format or datetime object
    data_dir : str or Path, optional
        Directory containing /ES CSV files, default="data"

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns: open, high, low, close, volume
        Returns None if file not found

    Examples
    --------
    >>> df = load_es_volume("2025-12-22")
    >>> df = load_es_volume("2025-12-22", data_dir="data/es_volume")
    """
    if isinstance(date, datetime):
        date_str = date.strftime("%Y-%m-%d")
    else:
        date_str = str(date)

    data_path = Path(data_dir)
    filename = f"ES_5_min_{date_str}.csv"
    filepath = data_path / filename

    if not filepath.exists():
        return None

    df = pd.read_csv(filepath)

    # Parse datetime and set as index
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_trading_hours(df, start_time="09:30", end_time="16:00"):
    """
    Filter DataFrame to trading hours only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index
    start_time : str, optional
        Start time in HH:MM format, default="09:30" (9:30 AM ET)
    end_time : str, optional
        End time in HH:MM format, default="16:00" (4:00 PM ET)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows within trading hours
    """
    if df is None or df.empty:
        return df

    # Parse time strings
    start_hour, start_min = map(int, start_time.split(":"))
    end_hour, end_min = map(int, end_time.split(":"))

    start = time(start_hour, start_min)
    end = time(end_hour, end_min)

    # Filter by time
    mask = (df.index.time >= start) & (df.index.time <= end)
    return df[mask]


def calculate_dollar_volume(volume_df, lookback_minutes=60, trading_hours_only=True):
    """
    Calculate rolling dollar volume from /ES 5-minute candles.

    Parameters
    ----------
    volume_df : pd.DataFrame
        DataFrame with datetime index and 'close' and 'volume' columns
    lookback_minutes : int, optional
        Rolling window size in minutes, default=60
    trading_hours_only : bool, optional
        Filter to trading hours (9:30 AM - 4:00 PM ET), default=True

    Returns
    -------
    pd.Series
        Rolling dollar volume per 1% move, indexed by datetime
        Returns None if input is None or missing required columns

    Notes
    -----
    Dollar volume is calculated as:
    1. Dollar volume = sum(close * volume) over lookback window
    2. Normalize to per-1% move: dollar_volume / 100

    Examples
    --------
    >>> df = load_es_volume("2025-12-22")
    >>> dv = calculate_dollar_volume(df, lookback_minutes=60)
    """
    if volume_df is None or volume_df.empty:
        return None

    if "close" not in volume_df.columns or "volume" not in volume_df.columns:
        return None

    # Filter to trading hours if requested
    if trading_hours_only:
        volume_df = filter_trading_hours(volume_df)

    if volume_df.empty:
        return None

    # Calculate dollar volume per bar
    dollar_vol_per_bar = volume_df["close"] * volume_df["volume"]

    # Calculate rolling sum over lookback window
    # lookback_minutes / 5 = number of 5-minute bars
    window_size = lookback_minutes // 5

    rolling_dollar_vol = dollar_vol_per_bar.rolling(window=window_size, min_periods=1).sum()

    # Normalize to per-1% move
    dollar_vol_per_1pct = rolling_dollar_vol / 100

    return dollar_vol_per_1pct


def get_dollar_volume_at_time(volume_df, target_time, lookback_minutes=60):
    """
    Get dollar volume at a specific timestamp.

    Parameters
    ----------
    volume_df : pd.DataFrame
        DataFrame with datetime index from load_es_volume
    target_time : str or datetime
        Target timestamp to query
    lookback_minutes : int, optional
        Rolling window for dollar volume calculation, default=60

    Returns
    -------
    float
        Dollar volume per 1% move at target time, or None if not found

    Examples
    --------
    >>> df = load_es_volume("2025-12-22")
    >>> dv = get_dollar_volume_at_time(df, "2025-12-22 14:30:00")
    >>> dv = get_dollar_volume_at_time(df, datetime(2025, 12, 22, 14, 30))
    """
    if volume_df is None or volume_df.empty:
        return None

    # Calculate dollar volume series
    dv_series = calculate_dollar_volume(volume_df, lookback_minutes=lookback_minutes)

    if dv_series is None or dv_series.empty:
        return None

    # Parse target time
    if isinstance(target_time, str):
        target_time = pd.to_datetime(target_time)

    # Find nearest timestamp
    if target_time in dv_series.index:
        return dv_series.loc[target_time]

    # Find closest timestamp within 10 minutes
    time_diff = np.abs((dv_series.index - target_time).total_seconds())
    min_diff_idx = time_diff.argmin()

    if time_diff.iloc[min_diff_idx] <= 600:  # 10 minutes = 600 seconds
        return dv_series.iloc[min_diff_idx]

    return None
