"""Compute core statistical metrics on backtest trade results."""

import pandas as pd


def compute_basic_stats(trade_totals: pd.DataFrame) -> dict:
    """
    Compute basic statistical metrics for trade returns.

    Args:
        trade_totals: DataFrame with 'value' column containing trade P&L

    Returns:
        Dict with keys:
            - mean_return: average P&L per trade
            - median_return: median P&L per trade
            - std_return: standard deviation of P&L
            - min_return: worst trade
            - max_return: best trade
            - total_pnl: sum of all trades

    Example:
        >>> stats = compute_basic_stats(trade_totals)
        >>> print(f"Mean: {stats['mean_return']:.2f}")
    """
    values = trade_totals["value"]

    return {
        "mean_return": values.mean(),
        "median_return": values.median(),
        "std_return": values.std(),
        "min_return": values.min(),
        "max_return": values.max(),
        "total_pnl": values.sum(),
    }


def compute_win_loss_stats(trade_totals: pd.DataFrame, win_threshold: float = 0.0) -> dict:
    """
    Compute win/loss statistics.

    Args:
        trade_totals: DataFrame with 'value' column containing trade P&L
        win_threshold: Threshold above which a trade is considered a win (default 0.0)

    Returns:
        Dict with keys:
            - wins: count of winning trades
            - losses: count of losing trades
            - win_rate: wins / (wins + losses)
            - avg_win: mean of winning trades
            - avg_loss: mean of losing trades
            - largest_win: best winning trade
            - largest_loss: worst losing trade

    Example:
        >>> stats = compute_win_loss_stats(trade_totals)
        >>> print(f"Win rate: {stats['win_rate']:.1%}")
    """
    values = trade_totals["value"]
    wins_mask = values > win_threshold
    losses_mask = values <= win_threshold

    wins_count = wins_mask.sum()
    losses_count = losses_mask.sum()
    total_trades = wins_count + losses_count

    winning_trades = values[wins_mask]
    losing_trades = values[losses_mask]

    return {
        "wins": wins_count,
        "losses": losses_count,
        "win_rate": wins_count / total_trades if total_trades > 0 else 0.0,
        "avg_win": winning_trades.mean() if len(winning_trades) > 0 else 0.0,
        "avg_loss": losing_trades.mean() if len(losing_trades) > 0 else 0.0,
        "largest_win": winning_trades.max() if len(winning_trades) > 0 else 0.0,
        "largest_loss": losing_trades.min() if len(losing_trades) > 0 else 0.0,
    }


def compute_expectancy(trade_totals: pd.DataFrame, win_threshold: float = 0.0) -> dict:
    """
    Compute trade expectancy (expected value per trade).

    Expectancy = (win_rate × avg_win) + ((1 - win_rate) × avg_loss)

    Args:
        trade_totals: DataFrame with 'value' column containing trade P&L
        win_threshold: Threshold above which a trade is considered a win (default 0.0)

    Returns:
        Dict with keys:
            - expectancy: expected value per trade
            - win_rate: included for context
            - avg_win: included for context
            - avg_loss: included for context

    Example:
        >>> exp = compute_expectancy(trade_totals)
        >>> print(f"Expectancy: ${exp['expectancy']:.2f} per trade")
    """
    win_loss = compute_win_loss_stats(trade_totals, win_threshold)

    win_rate = win_loss["win_rate"]
    avg_win = win_loss["avg_win"]
    avg_loss = win_loss["avg_loss"]

    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

    return {
        "expectancy": expectancy,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def compute_distribution_stats(trade_totals: pd.DataFrame) -> dict:
    """
    Compute distribution statistics (skewness, kurtosis, percentiles).

    Args:
        trade_totals: DataFrame with 'value' column containing trade P&L

    Returns:
        Dict with keys:
            - skew: skewness of return distribution
            - kurtosis: kurtosis of return distribution
            - percentiles: dict with percentile values [1, 5, 10, 25, 50, 75, 90, 95, 99]

    Example:
        >>> dist = compute_distribution_stats(trade_totals)
        >>> print(f"Skew: {dist['skew']:.2f}, Kurtosis: {dist['kurtosis']:.2f}")
    """
    values = trade_totals["value"]

    percentile_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentiles = {
        f"p{p}": values.quantile(p / 100.0)
        for p in percentile_levels
    }

    return {
        "skew": values.skew(),
        "kurtosis": values.kurt(),
        "percentiles": percentiles,
    }


def get_monthly_stats(trade_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly performance statistics.

    Args:
        trade_totals: DataFrame with 'value' and 'month' columns

    Returns:
        DataFrame with month as index and columns:
            - total_pnl: sum of trade P&L for the month
            - mean_trade: average P&L per trade
            - count: number of trades

    Example:
        >>> monthly = get_monthly_stats(trade_totals)
        >>> print(monthly.head())
    """
    monthly = trade_totals.groupby("month")["value"].agg(
        total_pnl="sum",
        mean_trade="mean",
        count="count"
    )

    return monthly
