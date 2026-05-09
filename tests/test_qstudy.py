"""
Unit tests for qstudy backtesting library.
Each test verifies against hand-calculated expected values, not against the implementation itself.
"""

import numpy as np
import pandas as pd
import pytest

from qstudy.study.engine import run
from qstudy.study.metrics import (
    annualized_return,
    annualized_vol,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    sharpe,
    turnover,
)
from qstudy.study.portfolio import build_long_short_positions, liquidity_filter, rebalance
from qstudy.signals.filters import momentum_context_filter, vol_filter, volume_zscore_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


# ---------------------------------------------------------------------------
# build_long_short_positions
# ---------------------------------------------------------------------------


class TestBuildPositions:
    def test_top_signal_is_long_bottom_is_short(self):
        """Highest signal value → long, lowest → short."""
        dates = make_dates(3)
        # A=5 (highest), B=3, C=2, D=1, E=0 (lowest)
        signal = pd.DataFrame(
            [[5, 3, 2, 1, 0]] * 3,
            index=dates,
            columns=list("ABCDE"),
            dtype=float,
        )
        pos = build_long_short_positions(signal, n_long=1, n_short=1)
        assert (pos["A"] > 0).all(), "highest signal should be long"
        assert (pos["E"] < 0).all(), "lowest signal should be short"
        assert (pos[["B", "C", "D"]] == 0).all().all(), "middle signals should be zero"

    def test_dollar_neutral(self):
        """abs(weights).sum(axis=1) == 1.0 on every row."""
        dates = make_dates(5)
        rng = np.random.default_rng(0)
        signal = pd.DataFrame(
            rng.normal(0, 1, (5, 10)), index=dates, columns=[f"T{i}" for i in range(10)]
        )
        pos = build_long_short_positions(signal, n_long=3, n_short=3)
        abs_sum = pos.abs().sum(axis=1)
        np.testing.assert_allclose(abs_sum.values, 1.0, atol=1e-10)

    def test_correct_counts(self):
        """Exactly n_long longs and n_short shorts per row."""
        dates = make_dates(4)
        rng = np.random.default_rng(1)
        signal = pd.DataFrame(
            rng.normal(0, 1, (4, 20)), index=dates, columns=[f"T{i}" for i in range(20)]
        )
        pos = build_long_short_positions(signal, n_long=5, n_short=5)
        assert (pos > 0).sum(axis=1).eq(5).all(), "should have exactly 5 longs"
        assert (pos < 0).sum(axis=1).eq(5).all(), "should have exactly 5 shorts"

    def test_nan_signal_ranked_last_becomes_short(self):
        """na_option='bottom' ranks NaN signals last, so they land in the short bucket.
        This matches the original notebook behavior: NaN = no view, ranked to the bottom,
        and the short cutoff selects the bottom n_short ranks regardless."""
        dates = make_dates(2)
        # 5 tickers, C is NaN — with n_long=1, n_short=1:
        # ranks: A=1, B=2, D=3, E=4, C=5 (NaN pushed to bottom)
        # short cutoff = 5 - (1-1) = 5, so rank >= 5 → C is short
        signal = pd.DataFrame(
            [[5.0, 3.0, np.nan, 1.0, 0.0]] * 2,
            index=dates,
            columns=list("ABCDE"),
        )
        pos = build_long_short_positions(signal, n_long=1, n_short=1)
        assert (pos["A"] > 0).all(), "A has highest signal, should be long"
        assert (pos["C"] < 0).all(), "C is NaN → ranked last → becomes the short"
        assert (pos["E"] == 0).all(), "E is rank 4, above short cutoff of 5"

    def test_short_cutoff_exact_match_original(self):
        """Verify short_cutoff = rank.count(axis=1) - (n_short - 1), matching original notebook.

        With 5 tickers and n_short=2: cutoff = 5 - 1 = 4, so ranks 4 and 5 are short.
        """
        dates = make_dates(1)
        # Explicit signal values so ranks are deterministic: A=5>B=4>C=3>D=2>E=1
        signal = pd.DataFrame([[5, 4, 3, 2, 1]], index=dates, columns=list("ABCDE"), dtype=float)
        pos = build_long_short_positions(signal, n_long=2, n_short=2)
        row = pos.iloc[0]
        # A, B → long (ranks 1, 2); D, E → short (ranks 4, 5); C → zero (rank 3)
        assert row["A"] > 0
        assert row["B"] > 0
        assert row["C"] == 0
        assert row["D"] < 0
        assert row["E"] < 0


# ---------------------------------------------------------------------------
# rebalance
# ---------------------------------------------------------------------------


class TestRebalance:
    def test_every_1_is_noop(self):
        """every=1 should return positions unchanged."""
        dates = make_dates(10)
        pos = pd.DataFrame(
            np.ones((10, 3)) * 0.25,
            index=dates,
            columns=list("ABC"),
        )
        result = rebalance(pos, every=1)
        pd.testing.assert_frame_equal(result, pos)

    def test_every_5_only_updates_on_stride(self):
        """Rows 0, 5, 10... should reflect new positions; rows 1-4 should be ffilled from row 0."""
        dates = make_dates(10)
        rng = np.random.default_rng(2)
        pos = pd.DataFrame(rng.normal(0, 1, (10, 3)), index=dates, columns=list("ABC"))

        result = rebalance(pos, every=5)

        # Row 0: rebalance date — should match original
        pd.testing.assert_series_equal(result.iloc[0], pos.iloc[0])
        # Rows 1-4: should match row 0 (ffilled)
        for i in range(1, 5):
            pd.testing.assert_series_equal(result.iloc[i], result.iloc[0], check_names=False)
        # Row 5: rebalance date — should match original row 5
        pd.testing.assert_series_equal(result.iloc[5], pos.iloc[5])
        # Rows 6-9: should match row 5 (ffilled)
        for i in range(6, 10):
            pd.testing.assert_series_equal(result.iloc[i], result.iloc[5], check_names=False)

    def test_fills_zeros_before_first_signal(self):
        """If first rows are NaN after masking, they should become 0."""
        dates = make_dates(6)
        pos = pd.DataFrame(
            {"A": [0.5] * 6, "B": [-0.5] * 6},
            index=dates,
        )
        result = rebalance(pos, every=5)
        assert not result.isna().any().any(), "no NaNs should remain"


# ---------------------------------------------------------------------------
# engine.run
# ---------------------------------------------------------------------------


class TestEngine:
    def test_execution_lag(self):
        """Position on day T should generate PnL using return on day T+1."""
        dates = make_dates(3)
        # Day 0: position A=1.0, B=0
        # Day 1: A returns 10% → PnL = position_from_day0 * return_day1 = 1.0 * 0.10 = 0.10
        positions = pd.DataFrame({"A": [1.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]}, index=dates)
        returns = pd.DataFrame({"A": [0.05, 0.10, 0.20], "B": [0.0, 0.0, 0.0]}, index=dates)

        port_ret = run(positions, returns)

        assert port_ret.iloc[0] == 0.0, "day 0 has no prior position, PnL must be 0"
        assert port_ret.iloc[1] == pytest.approx(0.10), "day 1 PnL = pos[0] * ret[1] = 1.0 * 0.10"
        assert port_ret.iloc[2] == pytest.approx(0.0), "day 2 position was 0"

    def test_long_short_pnl(self):
        """Dollar-neutral long/short: long A +0.5, short B -0.5."""
        dates = make_dates(2)
        positions = pd.DataFrame({"A": [0.5, 0.5], "B": [-0.5, -0.5]}, index=dates)
        # A up 10%, B up 4% → PnL = 0.5*0.10 + (-0.5)*0.04 = 0.05 - 0.02 = 0.03
        returns = pd.DataFrame({"A": [0.0, 0.10], "B": [0.0, 0.04]}, index=dates)

        port_ret = run(positions, returns)

        assert port_ret.iloc[1] == pytest.approx(0.03)

    def test_sum_across_tickers(self):
        """PnL is summed across all tickers each day."""
        dates = make_dates(2)
        positions = pd.DataFrame(
            {"A": [0.25, 0.25], "B": [0.25, 0.25], "C": [-0.25, -0.25], "D": [-0.25, -0.25]},
            index=dates,
        )
        returns = pd.DataFrame(
            {"A": [0.0, 0.08], "B": [0.0, 0.04], "C": [0.0, -0.02], "D": [0.0, 0.06]}, index=dates
        )
        # PnL day 1 = 0.25*0.08 + 0.25*0.04 + (-0.25)*(-0.02) + (-0.25)*0.06
        #           = 0.02 + 0.01 + 0.005 - 0.015 = 0.02
        port_ret = run(positions, returns)
        assert port_ret.iloc[1] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_sharpe_flat_returns(self):
        """Constant positive returns → infinite Sharpe (std=0), handle gracefully or return inf."""
        ret = pd.Series([0.001] * 100)
        sr = sharpe(ret)
        assert sr > 0 or np.isinf(sr)

    def test_sharpe_sign(self):
        """Positive mean returns → positive Sharpe."""
        rng = np.random.default_rng(3)
        ret = pd.Series(rng.normal(0.002, 0.01, 252))
        assert sharpe(ret) > 0

    def test_sharpe_known_value(self):
        """Sharpe = mean/std * sqrt(252). Verify with known inputs."""
        ret = pd.Series([0.01, -0.01] * 50)  # mean=0, std>0
        assert sharpe(ret) == pytest.approx(0.0, abs=1e-10)

    def test_annualized_return_known(self):
        """1% daily return for 252 days → (1.01^252 - 1) annualized."""
        ret = pd.Series([0.01] * 252)
        expected = 1.01**252 - 1
        assert annualized_return(ret) == pytest.approx(expected, rel=1e-6)

    def test_annualized_vol_known(self):
        """annualized_vol = std * sqrt(252)."""
        ret = pd.Series([0.01, -0.01] * 50)
        expected = ret.std() * np.sqrt(252)
        assert annualized_vol(ret) == pytest.approx(expected, rel=1e-10)

    def test_drawdown_series_starts_at_zero(self):
        """First value of drawdown series is always 0 (at peak)."""
        ret = pd.Series([0.05, -0.03, 0.02, -0.01])
        dd = drawdown_series(ret)
        assert dd.iloc[0] == pytest.approx(0.0)

    def test_drawdown_series_values(self):
        """Manually verify drawdown after a known loss sequence."""
        # Start at 1.0 → up 10% → 1.10 (peak) → down 10% → 0.99
        # drawdown at day 2 = 0.99/1.10 - 1 = -0.10
        ret = pd.Series([0.10, -0.10])
        dd = drawdown_series(ret)
        assert dd.iloc[0] == pytest.approx(0.0)
        assert dd.iloc[1] == pytest.approx(0.99 / 1.10 - 1, rel=1e-6)

    def test_max_drawdown_is_negative(self):
        """max_drawdown should always be <= 0."""
        rng = np.random.default_rng(4)
        ret = pd.Series(rng.normal(0, 0.01, 200))
        assert max_drawdown(ret) <= 0

    def test_max_drawdown_known(self):
        """100% loss → max drawdown = -1.0."""
        ret = pd.Series([0.5, -1.0])
        assert max_drawdown(ret) == pytest.approx(-1.0, rel=1e-6)

    def test_max_drawdown_duration_no_drawdown(self):
        """Monotonically rising equity → duration = 0, no date range."""
        ret = pd.Series([0.01] * 20)
        assert max_drawdown_duration(ret) == (0, None)

    def test_max_drawdown_duration_known(self):
        """One drawdown lasting exactly 3 days with correct date range."""
        # up → down → down → down → recover
        dates = pd.bdate_range("2020-01-01", periods=5)
        ret = pd.Series([0.10, -0.05, -0.02, -0.01, 0.20], index=dates)
        dur, date_range = max_drawdown_duration(ret)
        assert dur == 3
        assert date_range is not None
        assert date_range[0] == dates[1]  # drawdown starts on day 1
        assert date_range[1] == dates[3]  # drawdown ends on day 3

    def test_turnover_zero_when_positions_unchanged(self):
        """No position changes → turnover = 0 every day."""
        dates = make_dates(5)
        pos = pd.DataFrame({"A": [0.5] * 5, "B": [-0.5] * 5}, index=dates)
        tv = turnover(pos)
        assert (tv.iloc[1:] == 0).all()

    def test_turnover_known(self):
        """Full position flip A: 0.5→-0.5 (change=1.0), B: -0.5→0.5 (change=1.0) → turnover=2."""
        dates = make_dates(2)
        pos = pd.DataFrame({"A": [0.5, -0.5], "B": [-0.5, 0.5]}, index=dates)
        tv = turnover(pos)
        assert tv.iloc[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# liquidity_filter
# ---------------------------------------------------------------------------


class TestLiquidityFilter:
    def test_correct_count_per_row(self):
        """Exactly top_n True values per row after warmup period."""
        dates = make_dates(100)
        rng = np.random.default_rng(5)
        close = pd.DataFrame(
            rng.uniform(10, 200, (100, 20)), index=dates, columns=[f"T{i}" for i in range(20)]
        )
        volume = pd.DataFrame(
            rng.integers(1_000, 100_000, (100, 20)),
            index=dates,
            columns=[f"T{i}" for i in range(20)],
        )
        mask = liquidity_filter(close, volume, top_n=10, window=20)
        # After warmup, every row should have exactly 10 True values
        assert mask.iloc[20:].sum(axis=1).eq(10).all()

    def test_highest_dollar_volume_is_eligible(self):
        """The ticker with the highest dollar volume should always be in the eligible set."""
        dates = make_dates(80)
        close = pd.DataFrame({"cheap": [1.0] * 80, "expensive": [1000.0] * 80}, index=dates)
        volume = pd.DataFrame({"cheap": [1] * 80, "expensive": [1_000_000] * 80}, index=dates)
        mask = liquidity_filter(close, volume, top_n=1, window=10)
        assert mask["expensive"].iloc[10:].all()
        assert not mask["cheap"].iloc[10:].any()


# ---------------------------------------------------------------------------
# vol_filter
# ---------------------------------------------------------------------------


class TestVolFilter:
    def test_keep_low_removes_high_vol(self):
        """keep='low': assets with vol above the cross-sectional quantile should be NaN."""
        dates = make_dates(60)
        rng = np.random.default_rng(6)
        # Give ticker HV consistently high vol, LV consistently low vol
        hv = pd.Series(rng.normal(0, 0.05, 60), index=dates)  # high vol
        lv = pd.Series(rng.normal(0, 0.001, 60), index=dates)  # low vol
        returns = pd.DataFrame({"HV": hv, "LV": lv})
        signal = pd.DataFrame({"HV": [1.0] * 60, "LV": [1.0] * 60}, index=dates)

        filtered = vol_filter(signal, returns, vol_window=20, quantile=0.5, keep="low")

        # After warmup, LV should mostly pass and HV should mostly fail
        lv_pass_rate = filtered["LV"].iloc[20:].notna().mean()
        hv_pass_rate = filtered["HV"].iloc[20:].notna().mean()
        assert lv_pass_rate > hv_pass_rate

    def test_keep_high_is_inverse(self):
        """keep='high' and keep='low' should produce complementary masks."""
        dates = make_dates(60)
        rng = np.random.default_rng(7)
        returns = pd.DataFrame(rng.normal(0, 0.01, (60, 5)), index=dates, columns=list("ABCDE"))
        signal = pd.DataFrame(np.ones((60, 5)), index=dates, columns=list("ABCDE"))

        low = vol_filter(signal, returns, vol_window=20, quantile=0.6, keep="low")
        high = vol_filter(signal, returns, vol_window=20, quantile=0.6, keep="high")

        # Where low passes, high should fail and vice versa (after warmup)
        low_mask = low.iloc[20:].notna()
        high_mask = high.iloc[20:].notna()
        overlap = (low_mask & high_mask).any().any()
        assert not overlap, "low and high vol filters should not overlap"

    def test_cross_sectional_not_time_series(self):
        """Threshold is per-date across tickers, not per-ticker across time."""
        dates = make_dates(60)
        rng = np.random.default_rng(8)
        returns = pd.DataFrame(
            rng.normal(0, 0.01, (60, 10)), index=dates, columns=[f"T{i}" for i in range(10)]
        )
        signal = pd.DataFrame(np.ones((60, 10)), index=dates, columns=[f"T{i}" for i in range(10)])

        filtered = vol_filter(signal, returns, vol_window=20, quantile=0.5, keep="low")

        # With quantile=0.5 cross-sectionally, ~50% of tickers should pass each day
        pass_rate_per_day = filtered.iloc[20:].notna().mean(axis=1)
        # Should be consistently ~50%, not drifting over time
        assert pass_rate_per_day.mean() == pytest.approx(0.5, abs=0.15)


# ---------------------------------------------------------------------------
# volume_zscore_filter
# ---------------------------------------------------------------------------


class TestVolumeZscoreFilter:
    def test_high_volume_spike_passes(self):
        """A ticker with a sudden volume spike should pass the filter."""
        dates = make_dates(30)
        # Normal volume for all tickers, then spike on ticker A on last day
        volume = pd.DataFrame(
            {"A": [1_000_000] * 30, "B": [1_000_000] * 30, "C": [1_000_000] * 30},
            index=dates,
            dtype=float,
        )
        volume.loc[dates[-1], "A"] = 10_000_000  # big spike

        signal = pd.DataFrame({"A": [1.0] * 30, "B": [1.0] * 30, "C": [1.0] * 30}, index=dates)
        filtered = volume_zscore_filter(signal, volume, window=10, min_zscore_quantile=0.5)

        assert filtered.loc[dates[-1], "A"] == pytest.approx(1.0), "volume spike should pass"

    def test_low_volume_blocked(self):
        """A ticker with below-average volume should be filtered out."""
        dates = make_dates(30)
        volume = pd.DataFrame(
            {
                "A": [100] * 30,  # very low
                "B": [1_000_000] * 30,
                "C": [1_000_000] * 30,
            },
            index=dates,
            dtype=float,
        )
        signal = pd.DataFrame({"A": [1.0] * 30, "B": [1.0] * 30, "C": [1.0] * 30}, index=dates)

        filtered = volume_zscore_filter(signal, volume, window=10, min_zscore_quantile=0.5)

        # A has the lowest z-score every day after warmup → should be NaN
        a_pass_rate = filtered["A"].iloc[10:].notna().mean()
        assert a_pass_rate < 0.2, "low-volume ticker should mostly be filtered"


# ---------------------------------------------------------------------------
# momentum_context_filter
# ---------------------------------------------------------------------------


class TestMomentumContextFilter:
    def test_strongly_trending_asset_filtered(self):
        """Asset with strong consistent momentum should be filtered out (for MR strategies)."""
        dates = make_dates(40)
        # Ticker TREND goes up 2% every day, others are flat
        returns = pd.DataFrame(
            {
                "TREND": [0.02] * 40,
                "FLAT1": [0.0] * 40,
                "FLAT2": [0.0] * 40,
                "FLAT3": [0.0] * 40,
            },
            index=dates,
        )
        signal = pd.DataFrame(
            np.ones((40, 4)), index=dates, columns=["TREND", "FLAT1", "FLAT2", "FLAT3"]
        )

        filtered = momentum_context_filter(signal, returns, window=15, max_abs_quantile=0.5)

        # TREND should be filtered most of the time after warmup
        trend_pass_rate = filtered["TREND"].iloc[15:].notna().mean()
        assert trend_pass_rate < 0.2, "strongly trending asset should mostly be filtered"

    def test_flat_asset_passes(self):
        """Asset with near-zero momentum should pass the filter."""
        dates = make_dates(40)
        returns = pd.DataFrame(
            {
                "FLAT": [0.0001] * 40,
                "TREND": [0.05] * 40,
                "TREND2": [0.04] * 40,
            },
            index=dates,
        )
        signal = pd.DataFrame(np.ones((40, 3)), index=dates, columns=["FLAT", "TREND", "TREND2"])

        filtered = momentum_context_filter(signal, returns, window=15, max_abs_quantile=0.5)

        flat_pass_rate = filtered["FLAT"].iloc[15:].notna().mean()
        assert flat_pass_rate > 0.8, "near-flat asset should mostly pass"
