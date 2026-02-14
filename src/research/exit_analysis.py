"""Categorize and analyze trades by exit type."""

import pandas as pd


def classify_exit_reasons(trade_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Classify trades by exit reason based on exit time and P&L.

    Uses hardcoded rules specific to the 1DTE iron condor strategy:
    - 'prof_target_reached': exited before noon with profit
    - 'max_loss': exited before 1pm with loss
    - 'afternoon_prof': exited between noon-1pm with profit
    - 'forced_exit': exited at or after 1pm (forced close before expiration)

    Args:
        trade_totals: DataFrame with 'exit_time' and 'value' columns

    Returns:
        Copy of DataFrame with 'exit_reason' column added

    Example:
        >>> trades_with_exits = classify_exit_reasons(trade_totals)
        >>> print(trades_with_exits['exit_reason'].value_counts())
    """
    df = trade_totals.copy()

    # Extract hour from exit_time
    exit_hour = df["exit_time"].dt.hour

    # Initialize exit_reason column
    df["exit_reason"] = "not_set"

    # Apply classification rules
    # Priority order: forced_exit > afternoon_prof > max_loss > prof_target_reached
    prof_target_mask = (exit_hour < 12) & (df["value"] > 0.0)
    loss_target_mask = (exit_hour < 13) & (df["value"] <= 0.0)
    afternoon_prof_mask = (exit_hour >= 12) & (exit_hour < 13) & (df["value"] > 0.0)
    forced_exit_mask = (exit_hour >= 13)

    df.loc[prof_target_mask, "exit_reason"] = "prof_target_reached"
    df.loc[loss_target_mask, "exit_reason"] = "max_loss"
    df.loc[afternoon_prof_mask, "exit_reason"] = "afternoon_prof"
    df.loc[forced_exit_mask, "exit_reason"] = "forced_exit"

    return df


def compute_exit_type_stats(trade_totals_with_exits: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics grouped by exit type.

    Args:
        trade_totals_with_exits: DataFrame with 'exit_reason' and 'value' columns

    Returns:
        DataFrame with exit_reason as index and columns:
            - count: number of trades
            - win_rate: percentage of profitable trades
            - avg_pnl: average P&L
            - std_pnl: standard deviation of P&L
            - total_pnl: sum of P&L

    Example:
        >>> stats = compute_exit_type_stats(trade_totals_with_exits)
        >>> print(stats)
    """
    stats = trade_totals_with_exits.groupby("exit_reason")["value"].agg([
        ("count", "count"),
        ("win_rate", lambda x: (x > 0).mean()),
        ("avg_pnl", "mean"),
        ("std_pnl", "std"),
        ("total_pnl", "sum"),
    ])

    return stats


def get_exit_hour_distribution(trade_totals: pd.DataFrame) -> pd.Series:
    """
    Get distribution of exit hours.

    Useful for understanding when trades are being closed.

    Args:
        trade_totals: DataFrame with 'exit_time' column

    Returns:
        Series with hour as index and count as values, sorted by hour

    Example:
        >>> dist = get_exit_hour_distribution(trade_totals)
        >>> print(dist)
    """
    exit_hours = trade_totals["exit_time"].dt.hour
    return exit_hours.value_counts().sort_index()
