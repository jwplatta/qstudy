"""Shared gamma exposure calculation utilities."""

import numpy as np
import pandas as pd
from .intraday import calculate_zero_gamma_line


def row_gross_gex(df, spot, multiplier, gamma_scale):
    """
    Dealer-agnostic (gross) gamma exposure per row.

    Args:
        df: DataFrame with option chain data
        spot: Current underlying price
        multiplier: Contract multiplier (e.g., 100 for SPX)
        gamma_scale: Scaling factor for gamma units (e.g., 0.01)

    Returns:
        Series of gross gamma exposure values per row
    """
    return df["gamma"] * df["open_interest"] * (spot**2) * multiplier * gamma_scale


def apply_dealer_sign(value, dealer_short: bool):
    """
    Convert a gross metric into a signed metric under an assumed dealer position.

    Args:
        value: Gross metric value
        dealer_short: If True, assume dealers are short (sign=-1)

    Returns:
        Signed metric value
    """
    sign = -1.0 if dealer_short else 1.0
    return sign * value


def calculate_flip_distance(df, spot, days_out=5, deadband=0.002):
    """
    Calculate flip distance: how far spot is from the zero-gamma level.

    Flip distance measures proximity to the gamma regime boundary.
    Positive = inside positive-gamma territory (mean reversion expected)
    Negative = inside negative-gamma territory (trend risk expected)

    Args:
        df: DataFrame with option chain data
        spot: Current underlying price
        days_out: Max days to include in calculation, default=5
        deadband: Deadband threshold (±0.2% default) for neutral zone

    Returns:
        float: Flip distance as percentage (S - S0) / S
               Returns None if zero-gamma line cannot be found

    References:
        docs/GAMMA_REGIME_DIAGNOSTICS.md - Section 1: Flip Distance
    """
    # Find zero-gamma level using existing utility
    zero_gamma_level = calculate_zero_gamma_line(df, spot, days_out=days_out)

    if zero_gamma_level is None:
        return None

    # Calculate flip distance: (S - S0) / S
    flip_dist = (spot - zero_gamma_level) / spot

    # Apply deadband: treat small distances as neutral (zero)
    if abs(flip_dist) < deadband:
        return 0.0

    return flip_dist


def calculate_gamma_influence(gross_gex, dollar_volume):
    """
    Calculate gamma influence: mechanical impact of gamma hedging on price.

    Gamma influence estimates how much dealer hedging can affect price,
    conditional on movement. High values indicate gamma dominates microstructure.

    Args:
        gross_gex: Gross (unsigned) gamma exposure
        dollar_volume: Dollar volume per 1% move

    Returns:
        float: Gamma influence score (dimensionless ratio)
               Returns None if dollar_volume is invalid

    Formula:
        Gamma Influence = (0.01 * |GrossGEX|) / DollarVolume

    References:
        docs/GAMMA_REGIME_DIAGNOSTICS.md - Section 2: Gamma Influence
    """
    if dollar_volume is None or dollar_volume <= 0:
        return None

    hedge_flow_1pct = 0.01 * abs(gross_gex)
    gamma_influence = hedge_flow_1pct / dollar_volume

    return gamma_influence


def classify_regime(
    net_gex,
    flip_distance=None,
    gamma_influence=None,
    strong_threshold=50_000_000,
    neutral_threshold=5_000_000,
):
    """
    Classify gamma regime based on Net GEX and supporting metrics.

    Returns a classification dict with regime state, dealer position,
    and expected market behavior based on gamma exposure analysis.

    Args:
        net_gex: Net gamma exposure (calls - puts)
        flip_distance: Optional flip distance metric
        gamma_influence: Optional gamma influence metric
        strong_threshold: Threshold for "strongly positive/negative" ($50M default)
        neutral_threshold: Threshold for "near zero" ($5M default)

    Returns:
        dict: Classification with keys:
            - regime: "Strongly Positive" | "Near Zero" | "Strongly Negative"
            - dealer_state: "Long Gamma" | "Neutral" | "Short Gamma"
            - hedging_behavior: Description of dealer hedging
            - market_behavior: Expected market behavior
            - color: Suggested color code (green/yellow/red)

    References:
        docs/GAMMA_REGIME_DIAGNOSTICS.md - Section 3: Combined Interpretation
    """
    # Determine regime based on Net GEX
    if net_gex > strong_threshold:
        regime = "Strongly Positive"
        dealer_state = "Long Gamma"
        hedging_behavior = "Sell into strength, buy into weakness"
        market_behavior = "Volatility suppressed, mean reversion, pinning, slow/choppy price action"  # noqa: E501
        color = "green"

    elif net_gex < -strong_threshold:
        regime = "Strongly Negative"
        dealer_state = "Short Gamma"
        hedging_behavior = "Buy into strength, sell into weakness"
        market_behavior = "Trend risk, acceleration, squeezes, late-day rips or cascades"
        color = "red"

    elif abs(net_gex) <= neutral_threshold:
        regime = "Near Zero"
        dealer_state = "Neutral"
        hedging_behavior = "Minimal mechanical hedging impact"
        market_behavior = "Price responds more to flows/news, mixed behavior"
        color = "yellow"

    else:
        # Moderate positive or negative
        if net_gex > 0:
            regime = "Moderately Positive"
            dealer_state = "Long Gamma"
            hedging_behavior = "Sell into strength, buy into weakness"
            market_behavior = "Moderate mean reversion, some volatility suppression"
            color = "green"
        else:
            regime = "Moderately Negative"
            dealer_state = "Short Gamma"
            hedging_behavior = "Buy into strength, sell into weakness"
            market_behavior = "Moderate trend risk, some acceleration potential"
            color = "red"

    result = {
        "regime": regime,
        "dealer_state": dealer_state,
        "hedging_behavior": hedging_behavior,
        "market_behavior": market_behavior,
        "color": color,
        "net_gex": net_gex,
    }

    # Add optional metrics if provided
    if flip_distance is not None:
        result["flip_distance"] = flip_distance

    if gamma_influence is not None:
        result["gamma_influence"] = gamma_influence

    return result
