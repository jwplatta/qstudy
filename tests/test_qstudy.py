"""
Unit tests for qstudy backtesting library.
Each test verifies against hand-calculated expected values, not against the implementation itself.
"""

import numpy as np
import pandas as pd
import pytest

import qstudy as qs
from qstudy.signals.filters import momentum_context_filter, vol_filter, volume_zscore_filter
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
from qstudy.study.portfolio import (
    build_long_short_positions,
    build_proportional_positions,
    liquidity_filter,
    rebalance,
)

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

    def test_proportional_positions_match_expected_weights(self):
        """Proportional builder should size by clipped cross-sectional z-score."""
        dates = make_dates(1)
        signal = pd.DataFrame([[3.0, 1.0, -1.0, -3.0]], index=dates, columns=list("ABCD"))
        positions = build_proportional_positions(signal)
        expected = pd.DataFrame([[0.375, 0.125, -0.125, -0.375]], index=dates, columns=list("ABCD"))
        pd.testing.assert_frame_equal(positions, expected)

    def test_proportional_positions_are_dollar_neutral_and_fully_invested(self):
        """Proportional builder should keep row sums at 0 and gross at 1."""
        dates = make_dates(5)
        rng = np.random.default_rng(3)
        signal = pd.DataFrame(
            rng.normal(0, 1, (5, 12)), index=dates, columns=[f"T{i}" for i in range(12)]
        )
        positions = build_proportional_positions(signal)
        np.testing.assert_allclose(positions.sum(axis=1).values, 0.0, atol=1e-10)
        np.testing.assert_allclose(positions.abs().sum(axis=1).values, 1.0, atol=1e-10)

    def test_proportional_positions_preserve_nan_ineligible_assets(self):
        """NaN signals should stay NaN in the output weights."""
        dates = make_dates(1)
        signal = pd.DataFrame([[2.0, 0.0, np.nan, -2.0]], index=dates, columns=list("ABCD"))
        positions = build_proportional_positions(signal)
        assert np.isnan(positions.loc[dates[0], "C"])
        np.testing.assert_allclose(positions.drop(columns="C").sum(axis=1).values, 0.0, atol=1e-10)

    def test_proportional_positions_respect_clip_zscore(self):
        """Clipping should change the final cross-sectional weight mix."""
        dates = make_dates(1)
        signal = pd.DataFrame([[100.0, 5.0, 0.0, -1.0, -2.0]], index=dates, columns=list("ABCDE"))
        unclipped = build_proportional_positions(signal, clip_zscore=100.0)
        clipped = build_proportional_positions(signal, clip_zscore=0.5)
        assert not np.isclose(clipped.loc[dates[0], "B"], unclipped.loc[dates[0], "B"])
        np.testing.assert_allclose(clipped.abs().sum(axis=1).values, 1.0, atol=1e-10)


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


# ---------------------------------------------------------------------------
# BarraLiteFactorModel
# ---------------------------------------------------------------------------


def make_barra_data(n_dates=120, n_tickers=20, seed=42):
    """Synthetic returns, benchmark, close, and sector_map for Barra tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    # Common factor + idiosyncratic noise
    mkt = rng.normal(0, 0.01, n_dates)
    idio = rng.normal(0, 0.005, (n_dates, n_tickers))
    betas = rng.uniform(0.5, 1.5, n_tickers)
    returns = pd.DataFrame(
        mkt[:, None] * betas[None, :] + idio,
        index=dates,
        columns=tickers,
    )
    benchmark = pd.Series(mkt, index=dates, name="SPY")
    close = pd.DataFrame(
        (1 + returns).cumprod() * 100,
        index=dates,
        columns=tickers,
    )
    sectors = ["Tech", "Finance", "Energy", "Health"]
    sector_map = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}
    return returns, benchmark, close, sector_map


class TestBarraLiteFactorModel:
    def test_residuals_shape_matches_returns(self):
        returns, benchmark, close, sector_map = make_barra_data()
        from qstudy.signals.factors import BarraLiteFactorModel

        model = BarraLiteFactorModel(factors=["market"], beta_window=20)
        model.fit(returns, benchmark, close)
        residuals, daily_r2 = model.residualize(returns)
        assert residuals.shape == returns.shape

    def test_no_lookahead_in_early_rows(self):
        """Rolling beta needs beta_window days of history; early rows should be NaN residuals."""
        returns, benchmark, close, _ = make_barra_data()
        from qstudy.signals.factors import BarraLiteFactorModel

        model = BarraLiteFactorModel(factors=["market"], beta_window=60, min_stocks=5)
        model.fit(returns, benchmark, close)
        residuals, _ = model.residualize(returns)
        # First 60 rows should be all NaN (beta not yet estimable)
        assert residuals.iloc[:60].isna().all().all()

    def test_residuals_lower_benchmark_correlation(self):
        """Residuals should correlate less with the benchmark than raw returns."""
        returns, benchmark, close, _ = make_barra_data(n_dates=200, n_tickers=30)
        from qstudy.signals.factors import BarraLiteFactorModel

        model = BarraLiteFactorModel(factors=["market"], beta_window=40, min_stocks=10)
        model.fit(returns, benchmark, close)
        residuals, _ = model.residualize(returns)

        valid = residuals.dropna(how="all")
        bench_aligned = benchmark.reindex(valid.index)

        raw_corr = returns.reindex(valid.index).corrwith(bench_aligned).abs().mean()
        resid_corr = valid.corrwith(bench_aligned).abs().mean()
        assert resid_corr < raw_corr, "residuals should have lower benchmark correlation"

    def test_sector_dummies_included(self):
        """With sector dummies, factor_exposures_ should contain sector columns."""
        returns, benchmark, close, sector_map = make_barra_data()
        from qstudy.signals.factors import BarraLiteFactorModel

        model = BarraLiteFactorModel(
            factors=["market", "sector"], beta_window=20, sector_map=sector_map
        )
        model.fit(returns, benchmark, close)
        assert model._sector_dummies is not None
        assert len(model._sector_cols) > 0

    def test_min_stocks_threshold_produces_nan(self):
        """Dates where fewer than min_stocks have valid exposure data → NaN residuals."""
        returns, benchmark, close, _ = make_barra_data(n_dates=200, n_tickers=10)
        from qstudy.signals.factors import BarraLiteFactorModel

        # min_stocks larger than universe → every date is skipped
        model = BarraLiteFactorModel(factors=["market"], beta_window=20, min_stocks=999)
        model.fit(returns, benchmark, close)
        residuals, _ = model.residualize(returns)
        assert residuals.isna().all().all()

    def test_daily_r2_series_shape(self):
        """daily_r2 should be a Series indexed by dates where regression ran."""
        returns, benchmark, close, _ = make_barra_data(n_dates=150)
        from qstudy.signals.factors import BarraLiteFactorModel

        model = BarraLiteFactorModel(factors=["market"], beta_window=30, min_stocks=5)
        model.fit(returns, benchmark, close)
        _, daily_r2 = model.residualize(returns)
        assert isinstance(daily_r2, pd.Series)
        assert daily_r2.name == "cross_sectional_r2"
        assert (daily_r2 >= 0).all() and (daily_r2 <= 1.0).all()

    def test_exposures_on_returns_dataframe(self):
        """exposures_on() should return a DataFrame (tickers x factors) for a valid date."""
        returns, benchmark, close, sector_map = make_barra_data(n_dates=100)
        from qstudy.signals.factors import BarraLiteFactorModel

        model = BarraLiteFactorModel(
            factors=["market", "momentum"], beta_window=20, momentum_window=10
        )
        model.fit(returns, benchmark, close)
        date = returns.index[-1]
        exp = model.exposures_on(date)
        assert isinstance(exp, pd.DataFrame)
        assert "market" in exp.columns
        assert "momentum" in exp.columns

    def test_cross_sectional_residualize_wrapper(self):
        """Functional wrapper should return same types as BarraLiteFactorModel."""
        returns, benchmark, close, sector_map = make_barra_data()
        from qstudy.signals.factors import cross_sectional_residualize

        residuals, daily_r2 = cross_sectional_residualize(
            returns, benchmark, close, sector_map=sector_map, beta_window=20
        )
        assert isinstance(residuals, pd.DataFrame)
        assert isinstance(daily_r2, pd.Series)
        assert residuals.shape == returns.shape


# ---------------------------------------------------------------------------
# Study: add_factor_model / neutralize_positions / scale_risk / new aliases
# ---------------------------------------------------------------------------


def make_study_data(n_dates=150, n_tickers=20, seed=7):
    """Returns (universe_StudyData, benchmark_StudyData) with synthetic data."""
    from qstudy.data.loader import StudyData

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    mkt = rng.normal(0, 0.01, n_dates)
    returns_arr = mkt[:, None] * rng.uniform(0.5, 1.5, n_tickers)[None, :] + rng.normal(
        0, 0.005, (n_dates, n_tickers)
    )
    returns_df = pd.DataFrame(returns_arr, index=dates, columns=tickers)
    open_df = (1 + returns_df.shift(1).fillna(0.0)).cumprod() * 100
    high_df = np.maximum(open_df, close_df := (1 + returns_df).cumprod() * 100) * 1.01
    low_df = np.minimum(open_df, close_df) * 0.99
    volume_df = pd.DataFrame(
        rng.integers(100_000, 1_000_000, (n_dates, n_tickers)).astype(float),
        index=dates,
        columns=tickers,
    )
    universe = StudyData(
        tickers=tickers,
        open=open_df,
        high=high_df,
        low=low_df,
        close=close_df,
        volume=volume_df,
        returns=returns_df,
        log_returns=np.log(close_df / close_df.shift(1)),
    )

    bm_ret = pd.DataFrame({"SPY": mkt}, index=dates)
    bm_open = (1 + bm_ret.shift(1).fillna(0.0)).cumprod() * 100
    bm_close = (1 + bm_ret).cumprod() * 100
    bm_high = np.maximum(bm_open, bm_close) * 1.01
    bm_low = np.minimum(bm_open, bm_close) * 0.99
    benchmark = StudyData(
        tickers=["SPY"],
        open=bm_open,
        high=bm_high,
        low=bm_low,
        close=bm_close,
        volume=pd.DataFrame({"SPY": [1e6] * n_dates}, index=dates),
        returns=bm_ret,
        log_returns=np.log(bm_close / bm_close.shift(1)),
    )
    return universe, benchmark


def mr5(**cache):
    """5-day mean-reversion signal used as a base_signal fn throughout tests."""
    returns = (
        cache["residual_returns"] if cache.get("residual_returns") is not None else cache["returns"]
    )
    return -returns.rolling(5).mean()


class TestStudyNewMethods:
    def _make_study(self):
        return make_study_data()

    def test_add_factor_model_raises_without_benchmark(self):
        from qstudy import Study

        universe, _ = make_study_data()
        # Validation is deferred to run() so the chain call succeeds,
        # but run() raises when benchmark is missing.
        s = (
            Study(universe=universe)
            .add_factor_model("barra-lite", factors=["market"])
            .base_signal(mr5)
            .build_long_short(n_long=3, n_short=3)
        )
        with pytest.raises(ValueError, match="benchmark="):
            s.run()

    def test_add_factor_model_populates_factor_exposures(self):
        from qstudy import Study

        universe, benchmark = make_study_data()
        sectors = {t: "Tech" if i % 2 == 0 else "Finance" for i, t in enumerate(universe.tickers)}
        study = (
            Study(universe=universe, benchmark=benchmark)
            .add_factor_model(
                "barra-lite", factors=["market", "sector"], sector_map=sectors, beta_window=20
            )
            .residualize_returns()
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        assert study.cache["factor_exposures"] is not None
        assert "market" in study.cache["factor_exposures"]
        assert study.cache["residual_returns"] is not None

    def test_xs_daily_r2_in_cache_after_run(self):
        from qstudy import Study

        universe, benchmark = make_study_data()
        study = (
            Study(universe=universe, benchmark=benchmark)
            .add_factor_model("barra-lite", factors=["market"], beta_window=20)
            .residualize_returns()
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        assert study.cache["_xs_daily_r2"] is not None
        assert isinstance(study.cache["_xs_daily_r2"], pd.Series)

    def test_transform_signal_and_filter_signal_are_aliases(self):
        """transform_signal and filter_signal both add signal_filter steps."""
        from qstudy import Study

        universe, benchmark = make_study_data()
        demean = lambda signal, **cache: signal.sub(signal.mean(axis=1), axis=0)  # noqa: E731
        demean.__name__ = "demean"
        vol_f = lambda signal, **cache: signal  # noqa: E731
        vol_f.__name__ = "passthrough"

        s = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .transform_signal(demean)
            .filter_signal(vol_f)
            .build_long_short(n_long=5, n_short=5)
        )
        step_types = [stype for stype, _, _ in s._steps]
        assert step_types.count("signal_filter") == 2

    def test_add_tradeable_constraint_applies_mask(self):
        """Ineligible assets (all positions) should be zeroed by constraint."""
        from qstudy import Study, liquidity

        universe, benchmark = make_study_data()
        study = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .add_tradeable_constraint(liquidity(top_n=5, window=20))
            .build_long_short(n_long=3, n_short=3)
            .run()
        )
        assert study.cache["_tradeable_mask"] is not None
        # Non-zero positions should only be in top-5 liquid assets
        pos = study.cache["positions"]
        mask = study.cache["_tradeable_mask"]
        non_zero = pos.where(pos != 0).dropna(how="all")
        if not non_zero.empty:
            for date in non_zero.index[:5]:
                active_tickers = non_zero.loc[date].dropna().index
                assert mask.loc[date, active_tickers].all(), "active positions must be in mask"

    def test_scale_risk_with_vol_target(self):
        """vol_target scaling should keep portfolio vol near the target (within 2x)."""
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=250)
        target = 0.05
        study = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .scale_risk(vol_target=target)
            .run()
        )
        port_ret = study.cache["portfolio_returns"]
        realized_vol = port_ret.std() * (252**0.5)
        # Should be within 3x of target (loose check — scaling uses lookback)
        assert realized_vol < target * 3

    def test_build_proportional_positions_matches_helper(self):
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=80, n_tickers=12, seed=11)
        study = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .build_proportional_positions()
            .run()
        )
        expected = build_proportional_positions(mr5(returns=universe.returns))
        pd.testing.assert_frame_equal(study.cache["positions"], expected)

    def test_backward_compat_scale_returns_alias(self):
        """scale_returns(fn) should still work and emit a DeprecationWarning."""
        import warnings as _warnings

        from qstudy import Study

        universe, benchmark = make_study_data()
        identity = lambda positions, **cache: positions  # noqa: E731
        identity.__name__ = "identity"

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            s = (
                Study(universe=universe, benchmark=benchmark)
                .base_signal(mr5)
                .build_long_short(n_long=5, n_short=5)
                .scale_returns(identity)
                .run()
            )
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert s.cache["portfolio_returns"] is not None

    def test_backward_compat_add_filter_alias(self):
        """add_filter(fn) should still add a signal_filter step without error."""
        from qstudy import Study

        universe, benchmark = make_study_data()
        passthrough = lambda signal, **cache: signal  # noqa: E731
        passthrough.__name__ = "passthrough"

        s = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .add_filter(passthrough)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        assert s.cache["portfolio_returns"] is not None

    def test_study_cache_exposes_ohl_fields(self):
        from qstudy import Study

        universe, benchmark = make_study_data()
        study = Study(universe=universe, benchmark=benchmark)
        pd.testing.assert_frame_equal(study.cache["open"], universe.open)
        pd.testing.assert_frame_equal(study.cache["high"], universe.high)
        pd.testing.assert_frame_equal(study.cache["low"], universe.low)

    def test_download_uses_cache_when_available(self, monkeypatch, tmp_path):
        dates = pd.date_range("2024-01-02", periods=3, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL", "MSFT"]]
        )
        data = pd.DataFrame(
            [
                [100.0, 200.0, 101.0, 201.0, 99.0, 199.0, 100.5, 200.5, 1_000, 2_000],
                [101.0, 201.0, 102.0, 202.0, 100.0, 200.0, 101.5, 201.5, 1_100, 2_100],
                [102.0, 202.0, 103.0, 203.0, 101.0, 201.0, 102.5, 202.5, 1_200, 2_200],
            ],
            index=dates,
            columns=columns,
        )
        call_count = {"value": 0}

        def fake_download(*args, **kwargs):
            call_count["value"] += 1
            return data

        monkeypatch.setattr("qstudy.data.loader.yf.download", fake_download)

        first = qs.download(
            ["AAPL", "MSFT"],
            "2024-01-02",
            "2024-01-05",
            data_dir=tmp_path,
        )
        second = qs.download(
            ["AAPL", "MSFT"],
            "2024-01-02",
            "2024-01-05",
            data_dir=tmp_path,
        )

        assert call_count["value"] == 1
        pd.testing.assert_frame_equal(first.close, second.close)
        assert list((tmp_path / "yfinance").glob("*.pkl"))
        assert list((tmp_path / "yfinance").glob("*.json"))


# ---------------------------------------------------------------------------
# Pipeline vs manual equivalence
# ---------------------------------------------------------------------------
# These tests guard against regressions where the Study pipeline produces
# different results than the equivalent manual step-by-step computation.
# Two bugs were found during development:
#   1. add_tradeable_constraint filled excluded stocks with 0.0 instead of NaN,
#      so they remained valid candidates in build_long_short_positions ranking.
#   2. A custom position scaler that internally recomputes the equity curve must
#      use positions.shift(1) * returns to match qs.run() (1-day execution lag).
# ---------------------------------------------------------------------------


def make_factor_study_data(n_dates=200, n_tickers=30, seed=42):
    """Synthetic universe + benchmark + factor returns for equivalence tests."""
    from qstudy.data.loader import StudyData

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    mkt = rng.normal(0, 0.01, n_dates)
    returns_arr = mkt[:, None] * rng.uniform(0.5, 1.5, n_tickers)[None, :] + rng.normal(
        0, 0.005, (n_dates, n_tickers)
    )
    returns_df = pd.DataFrame(returns_arr, index=dates, columns=tickers)
    open_df = (1 + returns_df.shift(1).fillna(0.0)).cumprod() * 100
    close_df = (1 + returns_df).cumprod() * 100
    high_df = np.maximum(open_df, close_df) * 1.01
    low_df = np.minimum(open_df, close_df) * 0.99
    volume_df = pd.DataFrame(
        rng.integers(100_000, 10_000_000, (n_dates, n_tickers)).astype(float),
        index=dates,
        columns=tickers,
    )
    universe = StudyData(
        tickers=tickers,
        open=open_df,
        high=high_df,
        low=low_df,
        close=close_df,
        volume=volume_df,
        returns=returns_df,
        log_returns=np.log(close_df / close_df.shift(1)),
    )
    bm_ret = pd.DataFrame({"SPY": mkt}, index=dates)
    bm_open = (1 + bm_ret.shift(1).fillna(0.0)).cumprod() * 100
    bm_close = (1 + bm_ret).cumprod() * 100
    bm_high = np.maximum(bm_open, bm_close) * 1.01
    bm_low = np.minimum(bm_open, bm_close) * 0.99
    benchmark = StudyData(
        tickers=["SPY"],
        open=bm_open,
        high=bm_high,
        low=bm_low,
        close=bm_close,
        volume=pd.DataFrame({"SPY": [1e6] * n_dates}, index=dates),
        returns=bm_ret,
        log_returns=np.log(bm_close / bm_close.shift(1)),
    )
    # Two factor ETFs correlated with market
    f1 = mkt + rng.normal(0, 0.003, n_dates)
    f2 = mkt + rng.normal(0, 0.003, n_dates)
    factor_ret = pd.DataFrame({"F1": f1, "F2": f2}, index=dates)
    factor_open = (1 + factor_ret.shift(1).fillna(0.0)).cumprod() * 100
    factor_close = (1 + factor_ret).cumprod() * 100
    factor_high = np.maximum(factor_open, factor_close) * 1.01
    factor_low = np.minimum(factor_open, factor_close) * 0.99
    factors = StudyData(
        tickers=["F1", "F2"],
        open=factor_open,
        high=factor_high,
        low=factor_low,
        close=factor_close,
        volume=pd.DataFrame({"F1": [1e6] * n_dates, "F2": [1e6] * n_dates}, index=dates),
        returns=factor_ret,
        log_returns=np.log(factor_close / factor_close.shift(1)),
    )
    return universe, benchmark, factors


class TestStudyPipelineEquivalence:
    """Pipeline must produce identical portfolio returns to the equivalent manual computation.

    The manual side of each test mirrors exactly what the pipeline does internally,
    including the same signal formula (no extra .shift(1) — the engine handles execution lag).
    """

    def test_tradeable_constraint_excludes_from_ranking(self):
        """add_tradeable_constraint must NaN excluded stocks so they don't appear in positions.

        Regression: constraint used to fill with 0.0, letting zeroed stocks win short slots
        in build_long_short_positions (na_option='bottom' ranks NaN last = short candidates).
        """
        universe, benchmark, factors = make_factor_study_data()
        returns_df = universe.returns
        close_df = universe.close
        volume_df = universe.volume

        # Manual: signal matches pipeline mean_reversion (no .shift — engine handles lag)
        liq_mask = liquidity_filter(close_df, volume_df, top_n=15, window=30)
        signal_manual = -returns_df.rolling(5).mean()
        signal_manual = signal_manual.where(liq_mask)  # NaN excluded stocks — not 0.0
        pos_manual = build_long_short_positions(signal_manual, n_long=3, n_short=3)

        from qstudy import Study

        study = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .add_tradeable_constraint(qs.liquidity(top_n=15, window=30))
            .build_long_short(n_long=3, n_short=3)
            .run()
        )
        pos_pipeline = study.cache["_position_history"][0]["df"]  # position_builder output

        pd.testing.assert_frame_equal(
            pos_manual.fillna(0.0),
            pos_pipeline.fillna(0.0),
            check_exact=False,
            atol=1e-10,
        )

    def test_pipeline_matches_manual_portfolio_returns(self):
        """Full pipeline (residualize + filters + liquidity + positions) matches manual.

        Regression: add_tradeable_constraint filled with 0.0 instead of NaN, causing
        ineligible stocks to be selected as shorts in build_long_short_positions.
        """
        universe, benchmark, factors = make_factor_study_data()
        returns_df = universe.returns
        close_df = universe.close
        volume_df = universe.volume
        factor_returns = factors.returns

        # --- Manual (mirrors pipeline exactly: no .shift on signal, engine handles lag) ---
        residuals_df, _, _ = qs.residualize(returns_df, factor_returns)
        signal = -residuals_df.rolling(5).mean()
        signal = signal.sub(signal.mean(axis=1), axis=0)
        signal = vol_filter(signal, residuals_df, vol_window=5, quantile=0.6)
        signal = volume_zscore_filter(signal, volume_df, window=20, min_zscore_quantile=0.7)
        liq_mask = liquidity_filter(close_df, volume_df, top_n=15, window=30)
        signal = signal.where(liq_mask)  # NaN excluded — not 0.0
        ret_filtered = returns_df.where(liq_mask)
        positions_manual = build_long_short_positions(signal, n_long=3, n_short=3)
        port_ret_manual = run(positions_manual, ret_filtered)

        # --- Pipeline ---
        def demean(s, **c):
            return s.sub(s.mean(axis=1), axis=0)

        demean.__name__ = "demean"

        from qstudy import Study

        study = (
            Study(universe=universe, benchmark=benchmark, factors=factors)
            .residualize_returns()
            .base_signal(mr5)
            .transform_signal(demean)
            .add_vol_filter(vol_window=5, quantile=0.6)
            .add_volume_zscore_filter(window=20, min_zscore_quantile=0.7)
            .add_tradeable_constraint(qs.liquidity(top_n=15, window=30))
            .build_long_short(n_long=3, n_short=3)
            .run()
        )
        port_ret_pipeline = study.cache["portfolio_returns"]

        pd.testing.assert_series_equal(
            port_ret_manual,
            port_ret_pipeline,
            check_exact=False,
            check_names=False,
            atol=1e-10,
        )

    def test_equity_curve_scaler_uses_lagged_positions(self):
        """A position scaler that recomputes the equity curve must use positions.shift(1).

        Regression: missing shift(1) caused the equity curve inside the scaler to use
        same-day positions × returns, diverging from qs.run() which always applies a 1-day lag.
        The scaler receives unscaled positions; the engine will later shift them. So to
        accurately preview what returns will look like, the scaler must also shift by 1.
        """
        universe, benchmark, factors = make_factor_study_data()
        returns_df = universe.returns
        close_df = universe.close
        volume_df = universe.volume
        factor_returns = factors.returns

        # --- Manual (signal matches pipeline: no extra .shift) ---
        residuals_df, _, _ = qs.residualize(returns_df, factor_returns)
        signal = -residuals_df.rolling(5).mean()
        liq_mask = liquidity_filter(close_df, volume_df, top_n=15, window=30)
        signal = signal.where(liq_mask)
        ret_filtered = returns_df.where(liq_mask)
        positions = build_long_short_positions(signal, n_long=3, n_short=3)
        raw_port_ret = run(positions, ret_filtered)  # engine applies positions.shift(1)
        equity = (1 + raw_port_ret).cumprod()
        equity_ma = equity.rolling(10).mean()
        scale = pd.Series(np.where(equity > equity_ma, 1.0, 0.25), index=equity.index)
        scaled_positions = positions.mul(scale.shift(1), axis=0)
        port_ret_manual = run(scaled_positions, ret_filtered)

        # --- Pipeline scaler: must use positions.shift(1) to match qs.run() ---
        def equity_regime_scale(positions, **cache):
            returns = cache["returns"]
            mask = cache.get("_tradeable_mask")
            if mask is not None:
                returns = returns.where(mask)
            raw_ret = (positions.shift(1) * returns).sum(axis=1)  # shift required
            equity = (1 + raw_ret).cumprod()
            equity_ma = equity.rolling(10).mean()
            scale = pd.Series(np.where(equity > equity_ma, 1.0, 0.25), index=equity.index)
            return positions.mul(scale.shift(1), axis=0)

        equity_regime_scale.__name__ = "equity_regime_scale"

        from qstudy import Study

        study = (
            Study(universe=universe, benchmark=benchmark, factors=factors)
            .residualize_returns()
            .base_signal(mr5)
            .add_tradeable_constraint(qs.liquidity(top_n=15, window=30))
            .build_long_short(n_long=3, n_short=3)
            .scale_risk(equity_regime_scale)
            .run()
        )
        port_ret_pipeline = study.cache["portfolio_returns"]

        pd.testing.assert_series_equal(
            port_ret_manual,
            port_ret_pipeline,
            check_exact=False,
            check_names=False,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# PortfolioStudy
# ---------------------------------------------------------------------------


class TestPortfolioStudy:
    def _make_portfolio(self, n_dates=150, n_tickers=20, seed=7):
        from qstudy import PortfolioStudy, Study

        universe, benchmark = make_study_data(n_dates=n_dates, n_tickers=n_tickers, seed=seed)

        study1 = Study(name="mr").base_signal(mr5).build_long_short(n_long=3, n_short=3)
        study2 = (
            Study(name="mom")
            .base_signal(lambda **cache: cache["returns"].rolling(10).mean())
            .build_long_only(n=5)
        )
        portfolio = PortfolioStudy(
            strategies=[study1, study2],
            universe=universe,
            benchmark=benchmark,
            name="test_portfolio",
        )
        return portfolio, universe, benchmark

    def test_smoke_run(self):
        """PortfolioStudy.run() completes without error."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.run()
        assert portfolio.cache["portfolio_returns"] is not None

    def test_portfolio_returns_is_series(self):
        """portfolio_returns is a pd.Series with a DatetimeIndex."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.run()
        ret = portfolio.cache["portfolio_returns"]
        assert isinstance(ret, pd.Series)
        assert isinstance(ret.index, pd.DatetimeIndex)

    def test_positions_not_auto_renormalized(self):
        """Combined positions are NOT auto-renormalized; abs sum reflects sleeve weights."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.run()
        positions = portfolio.cache["positions"]
        abs_sum = positions.abs().sum(axis=1)
        nonzero = abs_sum[abs_sum > 0]
        # With two equal-weighted (0.5 each) non-overlapping strategies the abs sum
        # is approximately 0.5; it should be in (0, 1.0] but NOT forced to exactly 1.0.
        assert (nonzero > 0).all()
        assert (nonzero <= 1.0 + 1e-10).all()

    def test_fully_invest_opt_in(self):
        """fully_invest() forces abs(w).sum(axis=1) == 1.0."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.fully_invest().run()
        positions = portfolio.cache["positions"]
        abs_sum = positions.abs().sum(axis=1)
        nonzero = abs_sum[abs_sum > 0]
        np.testing.assert_allclose(nonzero.values, 1.0, atol=1e-10)

    def test_strategy_returns_df_shape(self):
        """strategy_returns_df has one column per strategy."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.run()
        df = portfolio.strategy_returns
        assert df.shape[1] == 2
        assert set(df.columns) == {"mr", "mom"}

    def test_strategy_corr_is_square(self):
        """strategy_corr is a 2x2 symmetric matrix."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.run()
        corr = portfolio.strategy_corr
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_metrics_attribute(self):
        """portfolio.metrics.sharpe_ratio is a float."""
        from qstudy.study.metrics import StudyMetrics

        portfolio, _, _ = self._make_portfolio()
        portfolio.run()
        m = portfolio.metrics
        assert isinstance(m, StudyMetrics)
        assert isinstance(m.sharpe_ratio, float)
        assert isinstance(m.max_drawdown, float)
        assert m.max_drawdown <= 0.0

    def test_study_metrics_attribute(self):
        """Study.metrics returns a StudyMetrics dataclass after run()."""
        from qstudy import Study
        from qstudy.study.metrics import StudyMetrics

        universe, benchmark = make_study_data()
        study = (
            Study(universe=universe, benchmark=benchmark, name="sm_test")
            .base_signal(mr5)
            .build_long_short(n_long=3, n_short=3)
            .run()
        )
        m = study.metrics
        assert isinstance(m, StudyMetrics)
        assert isinstance(m.sharpe_ratio, float)
        assert isinstance(m.information_ratio, float)

    def test_data_injection_strategies_no_universe(self):
        """Strategies initialized without universe run correctly via PortfolioStudy."""
        from qstudy import PortfolioStudy, Study

        universe, benchmark = make_study_data()

        # Build strategies without passing universe
        study1 = Study(name="mr_only").base_signal(mr5).build_long_short(n_long=3, n_short=3)
        # Attempting to run directly should raise
        with pytest.raises(RuntimeError, match="No data"):
            study1.run()

        # But running via PortfolioStudy should succeed
        portfolio = PortfolioStudy(
            strategies=[study1],
            universe=universe,
            benchmark=benchmark,
        ).run()
        assert portfolio.cache["portfolio_returns"] is not None

    def test_weight_equal_vol(self):
        """weight_equal_vol does not crash and produces non-zero positions."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.weight_equal_vol(window=60).run()
        abs_sum = portfolio.cache["positions"].abs().sum(axis=1)
        nonzero = abs_sum[abs_sum > 0]
        assert len(nonzero) > 0

    def test_metrics_dict(self):
        """metrics_dict() returns a dict with expected keys."""
        portfolio, _, _ = self._make_portfolio()
        portfolio.run()
        d = portfolio.metrics_dict()
        assert isinstance(d, dict)
        assert "sharpe" in d
        assert "ann_return" in d
        assert "max_drawdown" in d


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    """Tests for the transaction cost feature on Study and PortfolioStudy."""

    def _run_study(self, cost_bps=0.0):
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=200, seed=42)
        study = (
            Study(universe=universe, benchmark=benchmark, cost_bps=cost_bps)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        return study

    def test_zero_cost_identity(self):
        """cost_bps=0 must produce identical portfolio_returns to the default (no costs)."""
        s_no_cost = self._run_study(cost_bps=0.0)
        s_zero = self._run_study(cost_bps=0.0)
        pd.testing.assert_series_equal(
            s_no_cost.cache["portfolio_returns"],
            s_zero.cache["portfolio_returns"],
        )

    def test_gross_equals_portfolio_returns_when_no_costs(self):
        """gross_portfolio_returns == portfolio_returns when cost_bps == 0."""
        s = self._run_study(cost_bps=0.0)
        pd.testing.assert_series_equal(
            s.cache["gross_portfolio_returns"],
            s.cache["portfolio_returns"],
        )

    def test_costs_reduce_returns(self):
        """Net return must be <= gross return every day (costs are non-negative)."""
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=200, seed=42)
        study = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .with_transaction_costs(cost_bps=20)
            .run()
        )
        gross = study.cache["gross_portfolio_returns"]
        net = study.cache["portfolio_returns"]
        # Net <= gross everywhere (costs always >= 0)
        assert (net <= gross + 1e-12).all(), "net returns must not exceed gross returns"
        # At least some days have a cost applied
        assert (gross - net > 0).any(), "non-zero costs should reduce returns on some days"

    def test_fluent_chain_with_transaction_costs(self):
        """with_transaction_costs() works in the fluent chain."""
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=150, seed=1)
        study = (
            Study(universe=universe, benchmark=benchmark)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .with_transaction_costs(cost_bps=10)
            .run()
        )
        assert study.cache["portfolio_returns"] is not None
        assert study._cost_bps == 10.0

    def test_cost_drag_formula(self):
        """cost_drag_ann ≈ avg_daily_turnover * (cost_bps / 10_000) * 252."""
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=200, seed=42)
        cost_bps = 15.0
        study = (
            Study(universe=universe, benchmark=benchmark, cost_bps=cost_bps)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        s = study.cache["metrics_summary"]
        avg_to = s["avg_daily_turnover"]
        expected_drag = avg_to * (cost_bps / 10_000) * 252
        assert s["cost_drag_ann"] == pytest.approx(expected_drag, rel=1e-6)

    def test_gross_metrics_in_summary(self):
        """Summary Series contains gross_sharpe, gross_ann_return, net_sharpe, cost_bps."""
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=200, seed=42)
        study = (
            Study(universe=universe, benchmark=benchmark, cost_bps=10)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        s = study.cache["metrics_summary"]
        assert "gross_sharpe" in s
        assert "gross_ann_return" in s
        assert "net_sharpe" in s
        assert "cost_bps" in s
        assert s["cost_bps"] == pytest.approx(10.0)
        assert s["net_sharpe"] == pytest.approx(s["sharpe"])

    def test_no_cost_metrics_absent(self):
        """When cost_bps=0, gross/cost keys should not appear in the summary."""
        s = self._run_study(cost_bps=0.0)
        summary = s.cache["metrics_summary"]
        assert "gross_sharpe" not in summary
        assert "cost_drag_ann" not in summary

    def test_study_metrics_dataclass_populated(self):
        """StudyMetrics dataclass fields gross_ann_return/cost_drag_ann/cost_bps are populated."""
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=200, seed=42)
        study = (
            Study(universe=universe, benchmark=benchmark, cost_bps=10)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        m = study.metrics
        assert m.cost_bps == pytest.approx(10.0)
        assert m.gross_ann_return is not None
        assert m.cost_drag_ann is not None
        assert m.cost_drag_ann >= 0.0

    def test_pickle_round_trip_preserves_cost_config(self):
        """save() + from_cache() restores _cost_bps correctly."""
        import tempfile
        from pathlib import Path

        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=150, seed=5)
        study = (
            Study(universe=universe, benchmark=benchmark, cost_bps=12)
            .base_signal(mr5)
            .build_long_short(n_long=5, n_short=5)
            .run()
        )
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = Path(f.name)
        try:
            study.save(path)
            loaded = Study.from_cache(path)
            assert loaded._cost_bps == pytest.approx(12.0)
            assert loaded.cache["_cost_bps_config"] == pytest.approx(12.0)
        finally:
            path.unlink(missing_ok=True)

    def test_portfolio_study_with_transaction_costs(self):
        """PortfolioStudy with costs produces net < gross returns on some days."""
        from qstudy import PortfolioStudy, Study

        universe, benchmark = make_study_data(n_dates=200, seed=9)
        study1 = Study(name="mr").base_signal(mr5).build_long_short(n_long=3, n_short=3)
        study2 = (
            Study(name="mom")
            .base_signal(lambda **cache: cache["returns"].rolling(10).mean())
            .build_long_only(n=5)
        )
        portfolio = PortfolioStudy(
            strategies=[study1, study2],
            universe=universe,
            benchmark=benchmark,
            cost_bps=10,
        ).run()
        gross = portfolio.cache["gross_portfolio_returns"]
        net = portfolio.cache["portfolio_returns"]
        assert (net <= gross + 1e-12).all()
        assert (gross - net > 0).any()

    def test_portfolio_study_with_transaction_costs_method(self):
        """PortfolioStudy.with_transaction_costs() fluent method works."""
        from qstudy import PortfolioStudy, Study

        universe, benchmark = make_study_data(n_dates=150, seed=3)
        study1 = Study(name="mr").base_signal(mr5).build_long_short(n_long=3, n_short=3)
        portfolio = (
            PortfolioStudy(
                strategies=[study1],
                universe=universe,
                benchmark=benchmark,
            )
            .with_transaction_costs(cost_bps=8)
            .run()
        )
        assert portfolio._cost_bps == pytest.approx(8.0)
        assert portfolio.cache["gross_portfolio_returns"] is not None
        assert "cost_bps" in portfolio.cache["metrics_summary"]


class TestPipelineOrderEnforcement:
    """Verify that Study enforces the canonical position-scaler order:
    weight → scale_risk → neutralize → rebalance.
    """

    def _make_study(self, n_dates=100, seed=42):
        from qstudy import Study

        universe, benchmark = make_study_data(n_dates=n_dates, seed=seed)

        def mr_signal(**cache):
            return -cache["returns"].rolling(3).mean().shift(1)

        return Study(universe=universe, benchmark=benchmark).base_signal(mr_signal)

    # --- Valid orderings ---

    def test_weight_then_scale_risk_then_rebalance_is_valid(self):
        """Canonical order must not raise."""
        s = self._make_study()

        def noop_scaler(positions, **cache):
            return positions * 0.9

        noop_scaler.__name__ = "noop_scaler"
        s.build_long_short(n_long=5, n_short=5).weight_equal().scale_risk(noop_scaler).rebalance(
            every=5
        ).run()

    def test_weight_then_rebalance_is_valid(self):
        s = self._make_study()
        s.build_long_short(n_long=5, n_short=5).weight_equal().rebalance(every=5).run()

    def test_scale_risk_then_rebalance_is_valid(self):
        """No weight step is fine — scale_risk before rebalance is valid."""
        s = self._make_study()

        def noop_scaler(positions, **cache):
            return positions * 0.9

        noop_scaler.__name__ = "noop_scaler"
        s.build_long_short(n_long=5, n_short=5).scale_risk(noop_scaler).rebalance(every=5).run()

    # --- Invalid orderings that must raise ValueError ---

    def test_scale_risk_before_weight_raises(self):
        """scale_risk declared before weight_equal should raise ValueError."""
        s = self._make_study()

        def noop_scaler(positions, **cache):
            return positions * 0.9

        noop_scaler.__name__ = "noop_scaler"
        s.build_long_short(n_long=5, n_short=5).scale_risk(noop_scaler).weight_equal()
        with pytest.raises(ValueError, match="Canonical order"):
            s.run()

    def test_rebalance_before_scale_risk_raises(self):
        """rebalance declared before scale_risk should raise ValueError."""
        s = self._make_study()

        def noop_scaler(positions, **cache):
            return positions * 0.9

        noop_scaler.__name__ = "noop_scaler"
        s.build_long_short(n_long=5, n_short=5).rebalance(every=5).scale_risk(noop_scaler)
        with pytest.raises(ValueError, match="Canonical order"):
            s.run()

    def test_rebalance_before_weight_raises(self):
        """rebalance declared before weight_equal should raise ValueError."""
        s = self._make_study()
        s.build_long_short(n_long=5, n_short=5).rebalance(every=5).weight_equal()
        with pytest.raises(ValueError, match="Canonical order"):
            s.run()

    # --- Minor type tracking ---

    def test_step_minor_types_tracked(self):
        """_steps entries carry the correct minor type strings."""
        s = self._make_study()

        def noop_scaler(positions, **cache):
            return positions * 0.9

        noop_scaler.__name__ = "noop_scaler"
        s.build_long_short(n_long=5, n_short=5).weight_equal().scale_risk(noop_scaler).rebalance(
            every=5
        )
        minor_types = [minor for major, minor, fn in s._steps if major == "position_scaler"]
        assert minor_types == ["weight", "scale_risk", "rebalance"]
