import pandas as pd
import pytest
from src.qc_utils.backtest_loader import build_trade_totals, load_orders


def test_build_trade_totals():
    # Create sample order data - 2 timestamps = 1 trade (index // 2 = both 0)
    orders = pd.DataFrame({
        "Time": pd.to_datetime([
            "2022-04-05 15:55:00",
            "2022-04-05 15:55:00",
            "2022-04-06 09:45:00",
            "2022-04-06 09:45:00",
        ], utc=True).tz_convert("America/New_York"),
        "Value": [0.95, -1.55, -3.60, 8.00],
    })

    result = build_trade_totals(orders)

    # Check columns exist
    assert "exit_time" in result.columns
    assert "value" in result.columns
    assert "month" in result.columns
    assert "short_call_strike" in result.columns
    assert "short_put_strike" in result.columns

    # 2 timestamps -> 1 trade (both rows have index // 2 = 0)
    assert len(result) == 1
    assert result.iloc[0]["value"] == pytest.approx(-3.80, abs=0.01)


def test_build_trade_totals_multiple_trades():
    # Create sample data for 2 complete trades
    orders = pd.DataFrame({
        "Time": pd.to_datetime([
            "2022-04-05 15:55:00",  # Entry 1
            "2022-04-06 09:45:00",  # Exit 1
            "2022-04-07 15:55:00",  # Entry 2
            "2022-04-08 10:35:00",  # Exit 2
        ], utc=True).tz_convert("America/New_York"),
        "Value": [1.10, -4.20, 1.40, -0.50],
    })

    result = build_trade_totals(orders)

    # Should have 2 trades
    assert len(result) == 2

    # Trade 0: (-1.10) + (4.20) = 3.10... wait let me recalculate
    # Group by Time: row 0: 1.10 * -1 = -1.10
    #                 row 1: -4.20 * -1 = 4.20
    #                 row 2: 1.40 * -1 = -1.40
    #                 row 3: -0.50 * -1 = 0.50
    # After grouping, index // 2:
    # row 0 (value=-1.10) -> trade 0
    # row 1 (value=4.20) -> trade 0
    # row 2 (value=-1.40) -> trade 1
    # row 3 (value=0.50) -> trade 1

    # Trade 0 sum: -1.10 + 4.20 = 3.10
    # Trade 1 sum: -1.40 + 0.50 = -0.90

    assert result.iloc[0]["value"] == pytest.approx(3.10, abs=0.01)
    assert result.iloc[1]["value"] == pytest.approx(-0.90, abs=0.01)


def test_build_trade_totals_empty():
    # Test with empty DataFrame
    orders = pd.DataFrame({
        "Time": pd.Series([], dtype="datetime64[ns, UTC]").dt.tz_convert("America/New_York"),
        "Value": pd.Series([], dtype=float),
    })

    result = build_trade_totals(orders)

    # Should return empty DataFrame with correct structure
    assert len(result) == 0
    assert "exit_time" in result.columns
    assert "value" in result.columns
    assert "month" in result.columns
    assert "short_call_strike" in result.columns
    assert "short_put_strike" in result.columns


def test_build_trade_totals_extracts_short_strikes():
    orders = pd.DataFrame({
        "Time": pd.to_datetime([
            "2022-04-05 15:55:00",
            "2022-04-05 15:55:00",
            "2022-04-05 15:55:00",
            "2022-04-05 15:55:00",
            "2022-04-06 09:45:00",
            "2022-04-06 09:45:00",
            "2022-04-06 09:45:00",
            "2022-04-06 09:45:00",
        ], utc=True).tz_convert("America/New_York"),
        "Symbol": [
            "SPXW 220404P04430000",  # long put (entry buy)
            "SPXW 220404P04440000",  # short put (entry sell)
            "SPXW 220404C04500000",  # short call (entry sell)
            "SPXW 220404C04510000",  # long call (entry buy)
            "SPXW 220404P04430000",
            "SPXW 220404P04440000",
            "SPXW 220404C04500000",
            "SPXW 220404C04510000",
        ],
        "Quantity": [1, -1, -1, 1, -1, 1, 1, -1],
        "entry": [True, True, True, True, False, False, False, False],
        "Value": [1.0, -2.0, -2.0, 1.0, -0.5, 0.5, 0.5, -0.5],
    })

    result = build_trade_totals(orders)

    assert result.iloc[0]["short_put_strike"] == pytest.approx(4440.0)
    assert result.iloc[0]["short_call_strike"] == pytest.approx(4500.0)
