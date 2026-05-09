from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from qstudy.study.metrics import drawdown_series, max_drawdown


def drawdown_plot(
    returns: pd.Series,
    title: str = "Drawdown",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the drawdown series with max drawdown annotated.

    Args:
        returns: Daily portfolio return Series.
        title:   Chart title.
        ax:      Optional existing Axes to plot on.

    Returns:
        The Axes object for further customization.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    dd = drawdown_series(returns)
    ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="red")
    ax.plot(dd.index, dd.values, color="red", linewidth=0.8)
    ax.set_title(title)
    ax.grid(True)

    mdd = max_drawdown(returns)
    ax.text(0.02, 0.05, f"Max DD: {mdd:.1%}", transform=ax.transAxes, fontsize=10)

    return ax
