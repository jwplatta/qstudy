from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from qstudy.study.metrics import sharpe


def equity_curve(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Equity Curve",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot cumulative (1 + r).cumprod() with Sharpe annotated in the lower-left.

    Args:
        returns:   Daily portfolio return Series.
        benchmark: Optional benchmark return Series to overlay.
        title:     Chart title.
        ax:        Optional existing Axes to plot on.

    Returns:
        The Axes object for further customization.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    cum = (1 + returns).cumprod()
    label = "Strategy" if benchmark is not None else None
    ax.plot(cum.index, cum.values, label=label)

    sr = sharpe(returns)
    annotation = f"Sharpe: {sr:.2f}"

    if benchmark is not None:
        bm = benchmark.reindex(returns.index).fillna(0)
        cum_bm = (1 + bm).cumprod()
        ax.plot(cum_bm.index, cum_bm.values, linestyle="--", color="gray", label="Benchmark")
        ax.legend(loc="upper left")
        bm_sr = sharpe(bm)
        annotation += f"  |  Benchmark Sharpe: {bm_sr:.2f}"

    ax.set_title(title)
    ax.grid(True)
    ax.text(0.02, 0.05, annotation, transform=ax.transAxes, fontsize=10)

    return ax
