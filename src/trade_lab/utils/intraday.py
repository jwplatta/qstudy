"""Shared utility functions for intraday balance vs pause analysis."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .black_scholes import bs_gamma


def load_intraday_option_samples(symbol, sample_date, data_dir, days_out=10):
    """
    Load ALL option chain snapshots for a sample_date across expirations.

    Args:
        symbol: The ticker symbol (e.g., 'SPXW', 'SPX')
        sample_date: The date to load samples for (YYYY-MM-DD string)
        data_dir: Directory containing option chain CSV files
        days_out: Number of calendar days to include expirations (default: 10)

    Returns:
        List of (datetime, DataFrame) tuples sorted by time, where datetime is
        the sample timestamp and DataFrame contains the option chain data.
    """
    data_dir = Path(data_dir)
    sample_dt = datetime.strptime(sample_date, "%Y-%m-%d")
    end_dt = sample_dt + timedelta(days=days_out)

    pattern = f"{symbol}_exp*_{sample_date}_*.csv"
    csv_files = sorted(data_dir.glob(pattern))

    if not csv_files:
        raise ValueError(
            f"No option chain CSV files found for {symbol} sampled on {sample_date} in {data_dir}"
        )

    # Group files by sample time (truncated to minute level)
    # Files for different expirations are pulled sequentially, so they have
    # slightly different timestamps but should be treated as the same sample
    files_by_time = {}
    for csv_file in csv_files:
        try:
            parts = csv_file.stem.split("_")
            if len(parts) >= 4:
                exp_date_str = parts[1].replace("exp", "")
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")

                # Only include expirations within days_out window
                if not (sample_dt <= exp_date <= end_dt):
                    continue

                sample_time = parts[3]
                fetch_dt = datetime.strptime(f"{sample_date}_{sample_time}", "%Y-%m-%d_%H-%M-%S")

                # Truncate to minute level for grouping
                fetch_dt_minute = fetch_dt.replace(second=0, microsecond=0)

                if fetch_dt_minute not in files_by_time:
                    files_by_time[fetch_dt_minute] = []
                files_by_time[fetch_dt_minute].append(csv_file)
        except Exception:
            continue

    if not files_by_time:
        raise ValueError(
            f"No option chain files found for {symbol} on {sample_date} "
            f"with expirations within {days_out} days"
        )

    # Load and concatenate files for each sample time
    samples = []
    for fetch_dt in sorted(files_by_time.keys()):
        dfs = []
        for csv_file in files_by_time[fetch_dt]:
            df_temp = pd.read_csv(csv_file)
            if not df_temp.empty:
                dfs.append(df_temp)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            samples.append((fetch_dt, combined_df))

    return samples


def find_closest_expiration(sample_date, target_dte, data_dir, symbol):
    """
    Find expiration closest to target_dte days from sample_date.

    Args:
        sample_date: The date to search from (YYYY-MM-DD string)
        target_dte: Target days to expiration
        data_dir: Directory containing option chain CSV files
        symbol: The ticker symbol (e.g., 'SPXW', 'SPX')

    Returns:
        Expiration date string (YYYY-MM-DD) closest to target_dte
    """
    data_dir = Path(data_dir)
    sample_dt = datetime.strptime(sample_date, "%Y-%m-%d")
    target_exp = sample_dt + timedelta(days=target_dte)

    # Find all files for this symbol sampled on sample_date
    pattern = f"{symbol}_exp*_{sample_date}_*.csv"
    csv_files = sorted(data_dir.glob(pattern))

    if not csv_files:
        raise ValueError(
            f"No option chain CSV files found for {symbol} on {sample_date} in {data_dir}"
        )

    # Extract unique expirations
    expirations = set()
    for csv_file in csv_files:
        try:
            parts = csv_file.stem.split("_")
            if len(parts) >= 4:
                exp_date_str = parts[1].replace("exp", "")
                expirations.add(exp_date_str)
        except Exception:
            continue

    if not expirations:
        raise ValueError(f"No valid expirations found for {symbol} on {sample_date}")

    # Find closest to target_dte
    closest_exp = None
    closest_diff = float("inf")

    for exp_str in expirations:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
        diff = abs((exp_dt - target_exp).days)
        if diff < closest_diff:
            closest_diff = diff
            closest_exp = exp_str

    return closest_exp


def get_atm_iv(df):
    """
    Get ATM IV by averaging call/put IV at strike closest to underlying_price.

    Args:
        df: DataFrame with columns 'strike', 'underlying_price', 'contract_type',
            and 'theoretical_volatility'

    Returns:
        ATM IV as a decimal (e.g., 0.20 for 20%)
    """
    spot = pd.to_numeric(df["underlying_price"], errors="coerce").dropna().iloc[0]
    strikes = pd.to_numeric(df["strike"], errors="coerce")

    # Find strike closest to spot
    atm_strike = strikes.iloc[(strikes - spot).abs().argsort()[:1]].iloc[0]

    # Get call and put IV at ATM strike
    atm_rows = df[df["strike"] == atm_strike]
    ivs = pd.to_numeric(atm_rows["theoretical_volatility"], errors="coerce").dropna()

    if ivs.empty:
        return np.nan

    # Average call/put IV, convert from percent to decimal
    return ivs.mean() / 100.0


def calculate_net_gex_window(df, strike_window, spot):
    """
    Sum net GEX (calls - puts) within +/- strike_window of spot.

    Args:
        df: DataFrame with option chain data including 'strike', 'gamma',
            'open_interest', 'contract_type', and 'underlying_price'
        strike_window: Window in points around spot (e.g., 50 for +/- 50 pts)
        spot: Current spot price

    Returns:
        Net GEX value (calls - puts) within the window
    """
    df = df.copy()

    # Ensure numeric columns
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")

    # Filter to strike window
    mask = (df["strike"] >= spot - strike_window) & (df["strike"] <= spot + strike_window)
    window_df = df[mask].copy()

    if window_df.empty:
        return 0.0

    # Calculate GEX: gamma * OI * spot^2
    window_df["gex"] = window_df["gamma"] * window_df["open_interest"] * (spot**2)

    # Net GEX = calls - puts
    calls = window_df[window_df["contract_type"] == "CALL"]["gex"].sum()
    puts = window_df[window_df["contract_type"] == "PUT"]["gex"].sum()

    return calls - puts


def calculate_zero_gamma_line(df, spot, days_out=10):
    """
    Find zero-gamma crossing using GEXPrice algorithm.

    Uses Black-Scholes gamma calculation across a price grid to find
    where net GEX crosses zero.

    Args:
        df: DataFrame with option chain data
        spot: Current spot price
        days_out: Not used, kept for API compatibility

    Returns:
        Zero-gamma line strike price, or None if no crossing found
    """
    df = df.copy()

    # Parse expiration datetime (3 PM CT expiry)
    df["expiration_dt"] = pd.to_datetime(df["expiration_date"]) + pd.Timedelta(hours=15, minutes=15)

    # Use current time as reference
    now = datetime.now()

    # Time to expiry in years, floored at ~1 minute
    df["T"] = (df["expiration_dt"] - now).dt.total_seconds() / (365.0 * 24 * 3600)
    df["T"] = df["T"].clip(lower=(5.0 / (365.0 * 24 * 60)))

    # IV: convert percent -> decimal
    df["iv"] = pd.to_numeric(df["theoretical_volatility"], errors="coerce") / 100.0
    df["K"] = pd.to_numeric(df["strike"], errors="coerce")
    df["OI"] = pd.to_numeric(df["open_interest"], errors="coerce")

    df = df.dropna(subset=["iv", "K", "OI", "T"])
    df = df[(df["iv"] > 0) & (df["OI"] > 0)].copy()

    if df.empty:
        return None

    is_call = (df["contract_type"] == "CALL").to_numpy()
    k = df["K"].to_numpy(dtype=float)
    t = df["T"].to_numpy(dtype=float)
    iv = df["iv"].to_numpy(dtype=float)
    oi = df["OI"].to_numpy(dtype=float)

    # Price grid around spot
    prices_grid = np.arange(round(spot) - 300, round(spot) + 301, 1)

    net_gex_by_price = {}
    for p in prices_grid:
        s = np.full_like(k, float(p), dtype=float)
        gam = bs_gamma(s=s, k=k, t=t, sigma=iv, r=0.0, q=0.0)

        # GEX scaling: gamma * OI * price^2
        gex_each = gam * oi * (float(p) ** 2)

        # Net GEX = calls - puts
        net_gex = gex_each[is_call].sum() - gex_each[~is_call].sum()
        net_gex_by_price[float(p)] = float(net_gex)

    prices = np.array(sorted(net_gex_by_price.keys()), dtype=float)
    gex = np.array([net_gex_by_price[p] for p in prices], dtype=float)

    # Find zero-gamma crossing
    sign = np.sign(gex)

    # Handle zeros by forward-filling
    nonzero_mask = sign != 0
    if nonzero_mask.any():
        sign_filled = sign.copy()
        last_nonzero = None
        for i in range(len(sign)):
            if sign[i] != 0:
                last_nonzero = sign[i]
            elif last_nonzero is not None:
                sign_filled[i] = last_nonzero

        idx = np.where(np.diff(sign_filled) != 0)[0]
    else:
        idx = np.array([])

    if len(idx) == 0:
        return None

    # Use first crossing and interpolate
    i = idx[0]
    x1, x2 = prices[i], prices[i + 1]
    y1, y2 = gex[i], gex[i + 1]

    if y2 != y1:
        zgl = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
    else:
        zgl = (x1 + x2) / 2

    return zgl
