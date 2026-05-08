from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from qstudy.charts.drawdown import drawdown_plot
from qstudy.charts.equity_curve import equity_curve
from qstudy.charts.rolling_sharpe import rolling_sharpe_plot


def summary_plot(
    returns: pd.Series,
    rolling_window: int = 90,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """3-panel composite: equity curve (top), drawdown (middle), rolling Sharpe (bottom).

    Args:
        returns:        Daily portfolio return Series.
        rolling_window: Window for the rolling Sharpe panel.
        figsize:        Figure size.

    Returns:
        The Figure object.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.tight_layout(pad=3.0)

    equity_curve(returns, title="Equity Curve", ax=axes[0])
    drawdown_plot(returns, title="Drawdown", ax=axes[1])
    rolling_sharpe_plot(returns, window=rolling_window, title="Rolling Sharpe", ax=axes[2])

    plt.show()
    return fig
