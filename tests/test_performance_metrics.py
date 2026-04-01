import pandas as pd
import pytest
from src.qc_utils.performance_metrics import (
    compute_basic_stats,
    compute_distribution_stats,
    compute_expectancy,
    compute_win_loss_stats,
    get_monthly_stats,
)


@pytest.fixture
def sample_trades():
    """Create sample trade data for testing."""
    return pd.DataFrame({
        "exit_time": pd.to_datetime([
            "2022-04-06 09:45:00",
            "2022-04-08 10:35:00",
            "2022-04-13 09:35:00",
            "2022-05-05 11:00:00",
        ], utc=True).tz_convert("America/New_York"),
        "value": [1.50, -2.00, 0.75, 1.00],
        "month": pd.Period("2022-04", freq="M"),
    })


def test_compute_basic_stats(sample_trades):
    stats = compute_basic_stats(sample_trades)

    assert "mean_return" in stats
    assert "median_return" in stats
    assert "std_return" in stats
    assert "min_return" in stats
    assert "max_return" in stats
    assert "total_pnl" in stats

    assert stats["mean_return"] == pytest.approx(0.3125, abs=0.01)
    assert stats["min_return"] == -2.00
    assert stats["max_return"] == 1.50
    assert stats["total_pnl"] == pytest.approx(1.25, abs=0.01)


def test_compute_win_loss_stats(sample_trades):
    stats = compute_win_loss_stats(sample_trades)

    assert "wins" in stats
    assert "losses" in stats
    assert "win_rate" in stats
    assert "avg_win" in stats
    assert "avg_loss" in stats

    # 3 wins (1.50, 0.75, 1.00), 1 loss (-2.00)
    assert stats["wins"] == 3
    assert stats["losses"] == 1
    assert stats["win_rate"] == pytest.approx(0.75, abs=0.01)
    assert stats["avg_win"] == pytest.approx(1.0833, abs=0.01)
    assert stats["avg_loss"] == -2.00


def test_compute_expectancy(sample_trades):
    exp = compute_expectancy(sample_trades)

    assert "expectancy" in exp
    assert "win_rate" in exp
    assert "avg_win" in exp
    assert "avg_loss" in exp

    # Expectancy = (0.75 * 1.0833) + (0.25 * -2.00) = 0.8125 - 0.50 = 0.3125
    assert exp["expectancy"] == pytest.approx(0.3125, abs=0.01)


def test_compute_distribution_stats(sample_trades):
    dist = compute_distribution_stats(sample_trades)

    assert "skew" in dist
    assert "kurtosis" in dist
    assert "percentiles" in dist

    percentiles = dist["percentiles"]
    assert "p1" in percentiles
    assert "p50" in percentiles  # median
    assert "p99" in percentiles


def test_get_monthly_stats():
    trades = pd.DataFrame({
        "value": [1.0, -0.5, 2.0, 1.5],
        "month": [
            pd.Period("2022-04", freq="M"),
            pd.Period("2022-04", freq="M"),
            pd.Period("2022-05", freq="M"),
            pd.Period("2022-05", freq="M"),
        ],
    })

    monthly = get_monthly_stats(trades)

    assert len(monthly) == 2
    assert monthly.loc[pd.Period("2022-04", freq="M"), "total_pnl"] == pytest.approx(0.5, abs=0.01)
    assert monthly.loc[pd.Period("2022-05", freq="M"), "total_pnl"] == pytest.approx(3.5, abs=0.01)
    assert monthly.loc[pd.Period("2022-04", freq="M"), "count"] == 2


def test_compute_win_loss_stats_all_wins():
    trades = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    stats = compute_win_loss_stats(trades)

    assert stats["wins"] == 3
    assert stats["losses"] == 0
    assert stats["win_rate"] == 1.0
    assert stats["avg_loss"] == 0.0


def test_compute_win_loss_stats_all_losses():
    trades = pd.DataFrame({"value": [-1.0, -2.0, -3.0]})
    stats = compute_win_loss_stats(trades)

    assert stats["wins"] == 0
    assert stats["losses"] == 3
    assert stats["win_rate"] == 0.0
    assert stats["avg_win"] == 0.0
