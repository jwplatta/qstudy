import pandas as pd
import pytest
from src.qc_utils.tail_analysis import (
    compute_drawdown_stats,
    compute_tail_contribution,
    get_best_trades,
    get_worst_trades,
)


def test_compute_tail_contribution():
    trades = pd.DataFrame({"value": [1.0, -5.0, 0.5, -3.0, 2.0]})
    result = compute_tail_contribution(trades, n_worst=[1, 2])

    assert len(result) == 2
    assert result.iloc[0]["n_trades"] == 1
    assert result.iloc[0]["sum"] == -5.0


def test_get_worst_trades():
    trades = pd.DataFrame({"value": [1.0, -5.0, 0.5, -3.0, 2.0]})
    worst = get_worst_trades(trades, n=2)

    assert len(worst) == 2
    assert worst.iloc[0]["value"] == -5.0


def test_get_best_trades():
    trades = pd.DataFrame({"value": [1.0, -5.0, 0.5, -3.0, 2.0]})
    best = get_best_trades(trades, n=2)

    assert len(best) == 2
    assert best.iloc[0]["value"] == 2.0


def test_compute_drawdown_stats():
    trades = pd.DataFrame({"value": [1.0, -3.0, 0.5, 1.0]})
    dd = compute_drawdown_stats(trades)

    assert "max_drawdown" in dd
    assert "equity_curve" in dd
    assert dd["max_drawdown"] < 0
