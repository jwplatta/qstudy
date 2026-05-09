from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from qstudy.study.metrics import sharpe


def equity_curve(
    returns: pd.Series,
    title: str = "Equity Curve",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot cumulative (1 + r).cumprod() with Sharpe annotated in the lower-left.

    Args:
        returns: Daily portfolio return Series.
        title:   Chart title.
        ax:      Optional existing Axes to plot on.

    Returns:
        The Axes object for further customization.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    cum = (1 + returns).cumprod()
    ax.plot(cum.index, cum.values)
    ax.set_title(title)
    ax.grid(True)

    sr = sharpe(returns)
    ax.text(0.02, 0.05, f"Sharpe: {sr:.2f}", transform=ax.transAxes, fontsize=10)

    return ax
