import pandas as pd
import pytest

from src.qc_utils.market_context import (
    compare_win_loss_market_context,
    compute_range_group_stats,
)


def test_compare_win_loss_market_context():
    trades = pd.DataFrame({
        "value": [1.0, -2.0, 0.5, -1.0],
        "range": [1.5, 3.0, 1.8, 2.5],
        "close": [100, 95, 102, 98],
    })

    result = compare_win_loss_market_context(trades, context_columns=["range", "close"])

    assert len(result) == 2
    assert "range" in result["column"].values
    assert "close" in result["column"].values
    assert "win_mean" in result.columns
    assert "loss_mean" in result.columns


def test_compute_range_group_stats_threshold_split():
    trades = pd.DataFrame(
        {
            "value": [10.0, -5.0, 5.0, -10.0],
            "range": [1.0, 2.0, 3.0, 4.0],
            "exit_reason": ["forced_exit", "max_loss", "prof_target_reached", "max_loss"],
        }
    )

    result = compute_range_group_stats(trades, range_threshold=2.0)

    assert result["trade_count"].sum() == 4
    assert result["trade_count"].tolist() == [1, 3]
    assert result["win_rate"].tolist() == [1.0, pytest.approx(1.0 / 3.0)]
    assert result["mean_pnl"].tolist() == [10.0, pytest.approx(-10.0 / 3.0)]
    assert result["avg_win_pnl"].tolist() == [10.0, 5.0]
    assert result["avg_loss"].tolist() == [0.0, -7.5]
    assert result["pct_max_loss_exits"].tolist() == [0.0, pytest.approx(2.0 / 3.0)]
    assert result["group_total_pnl"].tolist() == [10.0, -10.0]
    assert result["total_pnl_contribution"].tolist() == [10.0, -10.0]


def test_compute_range_group_stats_threshold():
    trades = pd.DataFrame(
        {
            "value": [4.0, 3.0, -1.0, -2.0],
            "range": [1.0, 2.0, 3.0, 4.0],
            "exit_reason": ["forced_exit", "prof_target_reached", "max_loss", "max_loss"],
        }
    )

    result = compute_range_group_stats(trades, range_threshold=2.5)

    first_bucket = result.iloc[0]
    second_bucket = result.iloc[1]

    assert first_bucket["trade_count"] == 2
    assert first_bucket["win_rate"] == 1.0
    assert first_bucket["mean_pnl"] == 3.5
    assert first_bucket["avg_win_pnl"] == 3.5
    assert first_bucket["avg_loss"] == 0.0
    assert first_bucket["pct_max_loss_exits"] == 0.0
    assert first_bucket["group_total_pnl"] == pytest.approx(7.0)
    assert first_bucket["total_pnl_contribution"] == pytest.approx(7.0)

    assert second_bucket["trade_count"] == 2
    assert second_bucket["win_rate"] == 0.0
    assert second_bucket["mean_pnl"] == -1.5
    assert second_bucket["avg_win_pnl"] == 0.0
    assert second_bucket["avg_loss"] == -1.5
    assert second_bucket["pct_max_loss_exits"] == 1.0
    assert second_bucket["group_total_pnl"] == pytest.approx(-3.0)
    assert second_bucket["total_pnl_contribution"] == pytest.approx(-3.0)


def test_compute_range_group_stats_negative_portfolio_pnl_contribution_sign():
    trades = pd.DataFrame(
        {
            "value": [3.0, -1.0, -10.0],
            "range": [1.0, 2.0, 3.0],
            "exit_reason": ["forced_exit", "max_loss", "max_loss"],
        }
    )

    result = compute_range_group_stats(trades, range_threshold=2.0)

    first_bucket = result.iloc[0]   # < 2.0 => +3.0
    second_bucket = result.iloc[1]  # >= 2.0 => -11.0

    assert first_bucket["group_total_pnl"] == pytest.approx(3.0)
    assert first_bucket["total_pnl_contribution"] == pytest.approx(3.0)
    assert second_bucket["group_total_pnl"] == pytest.approx(-11.0)
    assert second_bucket["total_pnl_contribution"] == pytest.approx(-11.0)
