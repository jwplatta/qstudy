from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StudyMetrics:
    """Performance metrics for a Study or PortfolioStudy.

    Accessible via ``study.metrics.sharpe_ratio`` etc.
    """

    sharpe_ratio: float
    ann_return: float
    ann_vol: float
    max_drawdown: float
    drawdown_duration: int
    avg_daily_turnover: float | None
    benchmark_sharpe: float | None
    benchmark_corr: float | None
    information_ratio: float | None
    gross_ann_return: float | None = None
    cost_drag_ann: float | None = None
    cost_bps: float | None = None


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annualized return."""
    n = len(returns)
    total = (1 + returns).prod()
    return total ** (periods_per_year / n) - 1


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized realized volatility."""
    return returns.std() * np.sqrt(periods_per_year)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown from peak.

    Returns a Series with values <= 0 representing the fractional drawdown at each point.
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return cum / running_max - 1


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    return drawdown_series(returns).min()


def max_drawdown_duration(
    returns: pd.Series,
) -> tuple[int, tuple[pd.Timestamp, pd.Timestamp] | None]:
    """Longest number of days spent below a prior equity peak.

    Returns:
        (duration, (start_date, end_date)) where duration is in trading days and
        the dates are the inclusive bounds of the longest drawdown window.
        Returns (0, None) if there is no drawdown.
    """
    dd = drawdown_series(returns)
    in_drawdown = dd < 0
    max_dur = 0
    current_dur = 0
    best_start_idx: int | None = None
    current_start_idx: int | None = None

    for i, val in enumerate(in_drawdown):
        if val:
            if current_dur == 0:
                current_start_idx = i
            current_dur += 1
            if current_dur > max_dur:
                max_dur = current_dur
                best_start_idx = current_start_idx
        else:
            current_dur = 0

    if max_dur == 0 or best_start_idx is None:
        return (0, None)

    start_date = returns.index[best_start_idx]
    end_date = returns.index[best_start_idx + max_dur - 1]
    return (max_dur, (start_date, end_date))


def rolling_sharpe(
    returns: pd.Series,
    window: int = 90,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling annualized Sharpe over a lookback window."""
    return (
        returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(periods_per_year)
    )


def turnover(positions: pd.DataFrame) -> pd.Series:
    """Daily one-way turnover: sum of absolute position changes across tickers.

    Missing positions are treated as 0.0 so that first-day entries and gaps in
    coverage don't produce artificially inflated turnover.

    Useful for estimating transaction cost drag.
    """
    return positions.fillna(0.0).diff().abs().sum(axis=1)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualized information ratio: excess return over benchmark / tracking error."""
    excess = returns - benchmark.reindex(returns.index).fillna(0)
    return excess.mean() / excess.std() * np.sqrt(periods_per_year)


def summary(
    returns: pd.Series,
    positions: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    periods_per_year: int = 252,
    gross_returns: pd.Series | None = None,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Compute standard performance metrics.

    Returns a named Series with keys:
      sharpe, ann_return, ann_vol, max_drawdown, max_drawdown_duration,
      [max_drawdown_start, max_drawdown_end], [avg_daily_turnover],
      [benchmark_ann_return, benchmark_corr, information_ratio],
      [gross_sharpe, gross_ann_return, net_sharpe, cost_drag_ann, cost_bps]
        (last group only present when gross_returns is provided)

    When ``gross_returns`` is provided, ``returns`` is treated as the net (post-cost)
    series and ``sharpe`` / ``ann_return`` reflect net performance.
    ``gross_sharpe`` and ``gross_ann_return`` expose pre-cost figures.
    """
    dur, date_range = max_drawdown_duration(returns)
    result: dict = {
        "sharpe": sharpe(returns, periods_per_year),
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_vol(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_duration": dur,
    }
    if date_range is not None:
        result["max_drawdown_start"] = date_range[0]
        result["max_drawdown_end"] = date_range[1]
    if positions is not None:
        result["avg_daily_turnover"] = turnover(positions).mean()

    if benchmark is not None:
        bm = benchmark.squeeze().reindex(returns.index).fillna(0)
        result["benchmark_ann_return"] = annualized_return(bm, periods_per_year)
        result["benchmark_sharpe"] = sharpe(bm, periods_per_year)
        result["benchmark_corr"] = returns.corr(bm)
        result["information_ratio"] = information_ratio(returns, bm, periods_per_year)

    if gross_returns is not None:
        avg_to = result.get("avg_daily_turnover", float("nan"))
        result["gross_sharpe"] = sharpe(gross_returns, periods_per_year)
        result["gross_ann_return"] = annualized_return(gross_returns, periods_per_year)
        result["net_sharpe"] = result["sharpe"]
        result["cost_drag_ann"] = avg_to * (cost_bps / 10_000) * periods_per_year
        result["cost_bps"] = cost_bps

    return pd.Series(result)
