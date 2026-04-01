import pandas as pd
from src.qc_utils.exit_analysis import (
    classify_exit_reasons,
    compute_exit_type_stats,
)


def test_classify_exit_reasons():
    # Create times in ET timezone directly to avoid confusion
    trades = pd.DataFrame({
        "exit_time": pd.to_datetime([
            "2022-04-06 09:45:00-04:00",  # hour=9, value>0 -> prof_target_reached
            "2022-04-08 10:35:00-04:00",  # hour=10, value<0 -> max_loss
            "2022-04-13 12:30:00-04:00",  # hour=12, value>0 -> afternoon_prof
            "2022-04-14 13:00:00-04:00",  # hour=13 -> forced_exit
        ]),
        "value": [1.0, -2.0, 0.5, -1.0],
    })

    result = classify_exit_reasons(trades)

    assert "exit_reason" in result.columns
    assert result.iloc[0]["exit_reason"] == "prof_target_reached"
    assert result.iloc[1]["exit_reason"] == "max_loss"
    assert result.iloc[2]["exit_reason"] == "afternoon_prof"
    assert result.iloc[3]["exit_reason"] == "forced_exit"


def test_compute_exit_type_stats():
    trades = pd.DataFrame({
        "exit_reason": ["prof_target_reached", "prof_target_reached", "max_loss"],
        "value": [1.0, 0.8, -2.0],
    })

    stats = compute_exit_type_stats(trades)

    assert len(stats) == 2
    assert stats.loc["prof_target_reached", "count"] == 2
    assert stats.loc["max_loss", "count"] == 1
