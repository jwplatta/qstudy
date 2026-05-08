import numpy as np
import pandas as pd


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


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest number of days spent below a prior equity peak (in trading days)."""
    dd = drawdown_series(returns)
    in_drawdown = dd < 0
    max_dur = 0
    current_dur = 0
    for val in in_drawdown:
        if val:
            current_dur += 1
            max_dur = max(max_dur, current_dur)
        else:
            current_dur = 0
    return max_dur


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

    Useful for estimating transaction cost drag.
    """
    return positions.diff().abs().sum(axis=1)


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
) -> pd.Series:
    """Compute standard performance metrics.

    Returns a named Series with keys:
      sharpe, ann_return, ann_vol, max_drawdown, max_drawdown_duration,
      [avg_daily_turnover], [benchmark_ann_return, benchmark_corr, information_ratio]
    """
    metrics: dict = {
        "sharpe": sharpe(returns, periods_per_year),
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_vol": annualized_vol(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "max_drawdown_duration": max_drawdown_duration(returns),
    }
    if positions is not None:
        metrics["avg_daily_turnover"] = turnover(positions).mean()

    if benchmark is not None:
        bm = benchmark.squeeze().reindex(returns.index).fillna(0)
        metrics["benchmark_ann_return"] = annualized_return(bm, periods_per_year)
        metrics["benchmark_sharpe"] = sharpe(bm, periods_per_year)
        metrics["benchmark_corr"] = returns.corr(bm)
        metrics["information_ratio"] = information_ratio(returns, bm, periods_per_year)
    return pd.Series(metrics)
