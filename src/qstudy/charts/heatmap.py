from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def corr_heatmap(
    returns_matrix: pd.DataFrame,
    title: str = "Correlation",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Axes:
    """Plot a correlation heatmap from a returns matrix.

    Args:
        returns_matrix: DataFrame of returns (dates x series) to correlate.
        title:          Chart title.
        figsize:        Figure size.

    Returns:
        The Axes object for further customization.
    """
    _, ax = plt.subplots(figsize=figsize)
    corr = returns_matrix.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title)
    return ax


def param_heatmap(
    results_df: pd.DataFrame,
    row_param: str,
    col_param: str,
    metric: str = "sharpe",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Axes:
    """Pivot a param_grid() results DataFrame into a 2D heatmap.

    Args:
        results_df: Output of backtest.grid.param_grid().
        row_param:  Column name to use as heatmap rows.
        col_param:  Column name to use as heatmap columns.
        metric:     Metric column to visualize (default 'sharpe').
        title:      Chart title (defaults to metric name).
        figsize:    Figure size.

    Returns:
        The Axes object for further customization.
    """
    pivot = results_df.pivot(index=row_param, columns=col_param, values=metric).astype(float)
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(title or metric)
    return ax
