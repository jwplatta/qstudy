from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from qstudy.study.metrics import rolling_sharpe as _rolling_sharpe


def rolling_sharpe_plot(
    returns: pd.Series,
    window: int = 90,
    title: str = "Rolling Sharpe",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot rolling annualized Sharpe over a lookback window.

    Args:
        returns: Daily portfolio return Series.
        window:  Rolling window in trading days.
        title:   Chart title.
        ax:      Optional existing Axes to plot on.

    Returns:
        The Axes object for further customization.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    rs = _rolling_sharpe(returns, window=window)
    ax.plot(rs.index, rs.values)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{title} ({window}d window)")
    ax.grid(True)

    return ax
