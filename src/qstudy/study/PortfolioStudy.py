"""PortfolioStudy: combine multiple Study pipelines into a single portfolio backtest."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

import qstudy.study.engine as engine
import qstudy.study.metrics as metrics
from qstudy.data.loader import StudyData
from qstudy.signals.factors import BarraLiteFactorModel
from qstudy.study.metrics import StudyMetrics
from qstudy.study.Study import Study


class PortfolioStudy:
    """Combine multiple strategy Studies into a single portfolio backtest.

    Each strategy Study is run independently; their positions are aggregated with
    per-strategy weights into a single set of portfolio positions.  Individual
    strategy return streams are retained for correlation analysis and weighting.

    Usage::

        study1 = (
            Study(name="mean_reversion")
            .base_signal(mr_signal)
            .build_long_short(n_long=20, n_short=20)
            .weight_equal_vol(vol_window=60)
        )
        study2 = (
            Study(name="momentum")
            .base_signal(mom_signal)
            .build_long_only(n=15)
            .weight_equal_sharpe(window=126)
        )

        portfolio = (
            PortfolioStudy(
                strategies=[study1, study2],
                universe=load_sp500_universe(),
                benchmark=load_benchmark(),
                name="combined",
            )
            .weight_equal()
            .run()
        )
        portfolio.report()
        portfolio.metrics.sharpe_ratio

    Pipeline execution order::

        for each strategy:
            inject shared data → run strategy pipeline → compute strategy returns
        aggregate positions with portfolio weights
        run portfolio backtest engine
        compute portfolio metrics
        [optional] fit factor model and run factor regression on portfolio returns
    """

    def __init__(
        self,
        strategies: list[Study],
        universe: StudyData,
        benchmark: StudyData | None = None,
        name: str | None = None,
        cost_bps: float = 0.0,
    ) -> None:
        """
        Args:
            strategies: List of configured Study instances (not yet run).  Each
                        strategy's data will be overridden by the shared universe
                        and benchmark passed here.
            universe:   :class:`~qstudy.data.loader.StudyData` for the trading universe.
            benchmark:  Optional benchmark (e.g. SPY) used for metrics and factor regression.
            name:       Label shown in the tqdm progress bar.
        """
        if not isinstance(universe, StudyData):
            raise TypeError(
                "universe must be a StudyData object returned by qs.download(). "
                f"Got {type(universe).__name__}."
            )
        if not strategies:
            raise ValueError("strategies must be a non-empty list of Study instances.")

        self._strategies = list(strategies)
        self._universe = universe
        self._benchmark = benchmark
        self._name = name
        self._cost_bps: float = float(cost_bps)

        # Portfolio-level weighting (across strategies); default = equal
        self._portfolio_weighting_fn: Callable | None = None
        self._renormalize_combined: bool = False

        # Position scalers applied to combined positions before engine.run()
        self._position_scalers: list[Callable] = []

        # Factor model for portfolio-level neutralization (fitted at run time)
        self._neutralization_model: BarraLiteFactorModel | None = None
        self._neutralization_constraints: dict | None = None

        # Factor model for portfolio-level factor regression (optional)
        self._factor_model: BarraLiteFactorModel | None = None

        # Result cache — populated after run()
        self._cache: dict = {
            "portfolio_returns": None,
            "gross_portfolio_returns": None,
            "positions": None,
            "strategy_returns_df": None,
            "strategy_corr": None,
            "metrics_summary": None,
            "factor_regression": None,
            "benchmark": None,
        }

        if benchmark is not None:
            # Pre-align benchmark to universe dates; we'll re-slice at run time
            idx = universe.returns.index.intersection(benchmark.returns.index)
            self._cache["benchmark"] = benchmark.returns.reindex(idx).iloc[:, 0].copy()

    # ------------------------------------------------------------------
    # Portfolio-level weighting (how to weight strategy sleeves)
    # ------------------------------------------------------------------

    def fully_invest(self) -> PortfolioStudy:
        """Rescale combined portfolio positions so abs(weights).sum() == 1.0 each day.

        Off by default. Call this when you want the combined book to be fully invested
        rather than preserving the dollar exposure implied by the sleeve weights.
        Has no effect when strategies are already normalized and weights sum to 1.
        """
        self._renormalize_combined = True
        return self

    def weight_equal(self) -> PortfolioStudy:
        """Equal weight across all strategy sleeves."""
        self._set_portfolio_weighting(apply_equal_strategies, "equal")
        return self

    def weight_equal_vol(self, window: int = 126) -> PortfolioStudy:
        """Weight strategy sleeves inversely proportional to their realized return volatility."""
        fn = partial(_apply_equal_vol_strategies, window=window)
        fn.__name__ = "equal_vol"
        self._set_portfolio_weighting(fn, "equal_vol")
        return self

    def weight_equal_sharpe(self, window: int = 126) -> PortfolioStudy:
        """Weight strategy sleeves proportional to their rolling absolute Sharpe ratio."""
        fn = partial(_apply_equal_sharpe_strategies, window=window)
        fn.__name__ = "equal_sharpe"
        self._set_portfolio_weighting(fn, "equal_sharpe")
        return self

    def weight_optimal(self, window: int = 126, gamma: float = 1.0) -> PortfolioStudy:
        """Weight strategy sleeves using mean-variance optimization (ridge-regularized)."""
        fn = partial(_apply_optimal_strategies, window=window, gamma=gamma)
        fn.__name__ = "optimal"
        self._set_portfolio_weighting(fn, "optimal")
        return self

    def with_transaction_costs(self, cost_bps: float) -> PortfolioStudy:
        """Set a per-trade transaction cost assumption applied at the portfolio level.

        Costs are applied after all strategy positions are combined and netted, so
        internally crossing trades (e.g. strategy A long AAPL, strategy B short AAPL)
        do not generate spurious turnover.

        Args:
            cost_bps: One-way cost in basis points per dollar traded
                      (e.g. ``10`` = 10 bps = 0.10%).  Default is ``0`` (no costs).
        """
        self._cost_bps = float(cost_bps)
        return self

    # ------------------------------------------------------------------
    # Combined-position scalers
    # ------------------------------------------------------------------

    def scale_risk(self, fn: Callable) -> PortfolioStudy:
        """Apply a position scaler to the combined portfolio positions before backtesting.

        Called after sleeve positions are weighted and combined, so the scaler sees the
        full portfolio position DataFrame (dates x all tickers).

        Args:
            fn: ``fn(positions, **cache) -> pd.DataFrame``.
                Cache keys available: ``"returns"``, ``"benchmark"``, ``"close"``,
                ``"volume"``, ``"factor_exposures"`` (if neutralize_positions was called).
        """
        self._position_scalers.append(fn)
        return self

    def neutralize_positions(
        self,
        constraints: dict[str, float | tuple[float, float]],
        sector_map: dict | None = None,
        beta_window: int = 60,
        momentum_window: int = 20,
        vol_window: int = 20,
    ) -> PortfolioStudy:
        """Enforce factor exposure constraints on the combined portfolio positions.

        Fits a portfolio-level Barra-lite factor model on ``universe.returns`` at run
        time, then subtracts the factor projection from positions each day so that net
        exposure to each requested factor is driven to the target.

        Supported constraint factors: ``"market"`` (beta), ``"sector"``, ``"momentum"``.

        Args:
            constraints:      Dict of factor name -> target or tolerance band.
                              Examples:
                                ``{"market": 0}``              — zero net beta
                                ``{"sector": 0}``              — zero net sector tilt
                                ``{"momentum": (-0.05, 0.05)}`` — tolerance band
            sector_map:       ``{ticker: sector_name}`` dict. Required for ``"sector"``.
                              Fetch with ``qs.get_sector_map(tickers)``.
            beta_window:      Rolling window for market beta estimation.
            momentum_window:  Rolling window for momentum factor.
            vol_window:       Rolling window for volatility factor.
        """
        if self._benchmark is None:
            raise ValueError("neutralize_positions() requires benchmark= to be set.")

        factors = list(constraints.keys())
        self._neutralization_model = BarraLiteFactorModel(
            factors=factors,
            beta_window=beta_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            sector_map=sector_map,
        )
        self._neutralization_constraints = constraints

        # The actual scaler is registered as a position scaler; factor_exposures
        # will be populated into cache by run() before scalers are applied.
        targets = {}
        for fname, val in constraints.items():
            if isinstance(val, (int, float)):
                targets[fname] = (float(val), float(val))
            else:
                targets[fname] = (float(val[0]), float(val[1]))

        def _neutralize(positions, **cache):
            factor_exposures = cache.get("factor_exposures")
            if factor_exposures is None:
                return positions

            adjusted = positions.copy()
            for date in positions.index:
                active = positions.loc[date]
                active = active[active != 0.0]
                if active.empty:
                    continue

                w = active.copy()
                for fname, (lo, hi) in targets.items():
                    if fname not in factor_exposures:
                        continue
                    exp = factor_exposures[fname].loc[date, active.index].dropna()
                    if exp.empty:
                        continue
                    w_valid = w.reindex(exp.index).fillna(0.0)
                    net_exp = float((w_valid * exp).sum())
                    if lo <= net_exp <= hi:
                        continue
                    target_val = (lo + hi) / 2.0
                    exp_norm = float((exp**2).sum())
                    if exp_norm == 0.0:
                        continue
                    adj = (net_exp - target_val) / exp_norm
                    w_valid = w_valid - adj * exp
                    w = w.copy()
                    w.loc[exp.index] = w_valid.values

                gross = w.abs().sum()
                if gross > 0:
                    w = w / gross
                adjusted.loc[date, w.index] = w.values

            return adjusted.fillna(0.0)

        _neutralize.__name__ = f"neutralize_positions({list(constraints.keys())})"
        self._position_scalers.append(_neutralize)
        return self

    # ------------------------------------------------------------------
    # Factor model (for portfolio-level factor regression)
    # ------------------------------------------------------------------

    def add_factor_model(
        self,
        model: str = "barra-lite",
        factors: list[str] = ("market", "sector"),
        sector_map: dict | None = None,
        beta_window: int = 60,
        momentum_window: int = 20,
        vol_window: int = 20,
    ) -> PortfolioStudy:
        """Attach a factor model for portfolio-level factor regression.

        After :meth:`run`, the portfolio return stream is regressed against fitted
        factor returns to expose hidden beta, sector tilts, and style loadings.

        Args:
            model:            Only ``"barra-lite"`` is supported.
            factors:          Factor names to fit. Subset of
                              ``["market", "sector", "momentum", "volatility", "size"]``.
            sector_map:       ``{ticker: sector_name}`` dict. Required for ``"sector"`` factor.
            beta_window:      Rolling window for market beta estimation.
            momentum_window:  Rolling window for momentum factor.
            vol_window:       Rolling window for volatility factor.
        """
        if model != "barra-lite":
            raise ValueError(
                f"Unsupported factor model: {model!r}. Only 'barra-lite' is supported."
            )
        if self._benchmark is None:
            raise ValueError("add_factor_model() requires benchmark= to be set.")
        self._factor_model = BarraLiteFactorModel(
            factors=list(factors),
            beta_window=beta_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            sector_map=sector_map,
        )
        return self

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> PortfolioStudy:
        """Execute all strategy pipelines, aggregate positions, and compute portfolio metrics.

        Steps:
            1. Inject shared universe/benchmark data into each strategy.
            2. Run each strategy's pipeline independently.
            3. Collect per-strategy return streams.
            4. Compute portfolio-level strategy weights.
            5. Combine positions using weighted sum, then renormalize.
            5a. (Optional) Fit neutralization factor model; populate factor_exposures cache.
            5b. (Optional) Apply position scalers (scale_risk / neutralize_positions).
            6. Run portfolio backtest engine.
            7. Compute portfolio metrics.
            8. (Optional) Fit factor model and run factor regression.
        """
        n_strategies = len(self._strategies)
        n_extra = (
            3
            + len(self._position_scalers)
            + (1 if self._neutralization_model else 0)
            + (1 if self._cost_bps > 0.0 else 0)
        )

        # Stage 1 + 2: inject data and run each strategy
        strategy_returns: dict[str, pd.Series] = {}
        strategy_positions: dict[str, pd.DataFrame] = {}

        with tqdm(
            total=n_strategies + n_extra,
            desc=self._name or "PortfolioStudy.run",
        ) as pbar:
            for i, study in enumerate(self._strategies):
                label = study._name or f"strategy_{i}"
                pbar.set_postfix({"stage": f"run:{label}"})
                study._inject_data(self._universe, self._benchmark)
                study.run()
                strategy_returns[label] = study._cache["portfolio_returns"]
                strategy_positions[label] = study._cache["positions"]
                pbar.update(1)

            # Stage 3: build strategy_returns_df (dates x strategies)
            pbar.set_postfix({"stage": "strategy_returns"})
            strategy_returns_df = pd.DataFrame(strategy_returns)
            self._cache["strategy_returns_df"] = strategy_returns_df
            self._cache["strategy_corr"] = strategy_returns_df.corr()

            # Stage 4: compute strategy weights
            pbar.set_postfix({"stage": "weighting"})
            if self._portfolio_weighting_fn is not None:
                weights_series = self._portfolio_weighting_fn(strategy_returns_df)
            else:
                n = len(self._strategies)
                weights_series = pd.Series(
                    {label: 1.0 / n for label in strategy_returns_df.columns}
                )

            # Stage 5: combine positions
            pbar.set_postfix({"stage": "combine_positions"})
            combined_positions = _combine_positions(
                strategy_positions, weights_series, renormalize=self._renormalize_combined
            )
            self._cache["positions"] = combined_positions
            pbar.update(1)

            # Stage 5a: fit neutralization factor model (if requested)
            if self._neutralization_model is not None:
                pbar.set_postfix({"stage": "fit_neutralization_model"})
                idx_neu = self._universe.returns.index
                if self._cache["benchmark"] is not None:
                    idx_neu = idx_neu.intersection(self._cache["benchmark"].index)
                self._neutralization_model.fit(
                    returns=self._universe.returns.reindex(idx_neu),
                    benchmark_returns=self._cache["benchmark"],
                    close=self._universe.close.reindex(idx_neu),
                )
                self._cache["factor_exposures"] = self._neutralization_model.factor_exposures_
                pbar.update(1)

            # Stage 5b: apply position scalers (scale_risk / neutralize_positions)
            if self._position_scalers:
                idx_ps = self._universe.returns.index
                if self._cache["benchmark"] is not None:
                    idx_ps = idx_ps.intersection(self._cache["benchmark"].index)
                scaler_cache = {
                    "returns": self._universe.returns.reindex(idx_ps),
                    "close": self._universe.close.reindex(idx_ps),
                    "volume": self._universe.volume.reindex(idx_ps),
                    "benchmark": self._cache["benchmark"],
                    "factor_exposures": self._cache.get("factor_exposures"),
                }
                for scaler_fn in self._position_scalers:
                    pbar.set_postfix({"stage": getattr(scaler_fn, "__name__", "scale_risk")})
                    combined_positions = scaler_fn(combined_positions, **scaler_cache)
                    pbar.update(1)
                self._cache["positions"] = combined_positions

            # Stage 6: run portfolio engine
            pbar.set_postfix({"stage": "backtest"})
            idx = self._universe.returns.index
            if self._cache["benchmark"] is not None:
                idx = idx.intersection(self._cache["benchmark"].index)
            universe_returns = self._universe.returns.reindex(idx)
            # Align combined positions to same date range
            combined_positions_aligned = combined_positions.reindex(idx).fillna(0.0)
            # Align universe returns to combined position columns
            all_tickers = combined_positions_aligned.columns
            returns_for_engine = universe_returns.reindex(columns=all_tickers)
            gross_returns = engine.run(combined_positions_aligned, returns_for_engine)
            self._cache["gross_portfolio_returns"] = gross_returns
            self._cache["portfolio_returns"] = gross_returns
            pbar.update(1)

            # Transaction costs (optional)
            if self._cost_bps > 0.0:
                pbar.set_postfix({"stage": "transaction_costs"})
                cost_per_dollar = self._cost_bps / 10_000.0
                cost_series = (
                    metrics.turnover(combined_positions_aligned)
                    .reindex(gross_returns.index)
                    .fillna(0.0)
                    * cost_per_dollar
                )
                self._cache["portfolio_returns"] = gross_returns - cost_series
                pbar.update(1)

            # Stage 7: metrics
            pbar.set_postfix({"stage": "metrics"})
            self._cache["metrics_summary"] = metrics.summary(
                self._cache["portfolio_returns"],
                positions=combined_positions_aligned,
                benchmark=self._cache["benchmark"],
                gross_returns=(
                    self._cache["gross_portfolio_returns"] if self._cost_bps > 0.0 else None
                ),
                cost_bps=self._cost_bps,
            )
            pbar.update(1)

        # Stage 8: factor model + regression (outside tqdm to avoid nesting)
        if self._factor_model is not None:
            self._run_factor_regression()

        return self

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, rolling_window: int = 90) -> PortfolioStudy:
        """Print metrics, strategy correlations, and show performance charts.

        Requires :meth:`run` to have been called first.
        """
        from qstudy.charts.summary import summary_plot

        self._require_run("report")

        print(f"\n{'=' * 60}")
        print(f"  Portfolio: {self._name or 'PortfolioStudy'}")
        print(f"{'=' * 60}\n")
        print(self._cache["metrics_summary"].to_string())

        print("\n--- Strategy Return Correlations ---")
        print(self._cache["strategy_corr"].to_string())

        if self._cache.get("factor_regression") is not None:
            fr = self._cache["factor_regression"]
            print("\n--- Factor Regression ---")
            reg_df = pd.DataFrame({"coefficient": fr["coefficients"], "t_stat": fr["t_stats"]})
            print(reg_df.to_string())
            print(f"R²: {fr['r_squared']:.4f}")

        summary_plot(self._cache["portfolio_returns"], rolling_window=rolling_window)
        return self

    # ------------------------------------------------------------------
    # Metrics property
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> StudyMetrics:
        """Performance metrics as a typed dataclass.

        Requires :meth:`run` to have been called first.

        Example::

            portfolio.metrics.sharpe_ratio
            portfolio.metrics.max_drawdown
        """
        self._require_run("metrics")
        s = self._cache["metrics_summary"]
        return StudyMetrics(
            sharpe_ratio=float(s.get("sharpe", float("nan"))),
            ann_return=float(s.get("ann_return", float("nan"))),
            ann_vol=float(s.get("ann_vol", float("nan"))),
            max_drawdown=float(s.get("max_drawdown", float("nan"))),
            drawdown_duration=int(s.get("max_drawdown_duration", 0)),
            avg_daily_turnover=(
                float(s["avg_daily_turnover"]) if "avg_daily_turnover" in s else None
            ),
            benchmark_sharpe=(float(s["benchmark_sharpe"]) if "benchmark_sharpe" in s else None),
            benchmark_corr=(float(s["benchmark_corr"]) if "benchmark_corr" in s else None),
            information_ratio=(float(s["information_ratio"]) if "information_ratio" in s else None),
            gross_ann_return=(float(s["gross_ann_return"]) if "gross_ann_return" in s else None),
            cost_drag_ann=(float(s["cost_drag_ann"]) if "cost_drag_ann" in s else None),
            cost_bps=(float(s["cost_bps"]) if "cost_bps" in s else None),
        )

    # ------------------------------------------------------------------
    # Cache / I/O accessors
    # ------------------------------------------------------------------

    @property
    def cache(self) -> dict:
        """The portfolio cache containing all intermediate and final DataFrames."""
        return self._cache

    @property
    def strategy_returns(self) -> pd.DataFrame:
        """Per-strategy return streams as a DataFrame (dates x strategies).

        Requires :meth:`run` to have been called first.
        """
        self._require_run("strategy_returns")
        return self._cache["strategy_returns_df"]

    @property
    def strategy_corr(self) -> pd.DataFrame:
        """Correlation matrix of per-strategy return streams.

        Requires :meth:`run` to have been called first.
        """
        self._require_run("strategy_corr")
        return self._cache["strategy_corr"]

    def metrics_dict(self) -> dict:
        """Return the portfolio metrics summary as a plain dictionary."""
        self._require_run("metrics_dict")
        return self._cache["metrics_summary"].to_dict()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_portfolio_weighting(self, fn: Callable, label: str) -> None:
        if self._portfolio_weighting_fn is not None:
            warnings.warn(
                f"Replacing existing portfolio weighting scheme with '{label}'.", stacklevel=3
            )
        self._portfolio_weighting_fn = fn

    def _require_run(self, method: str) -> None:
        if self._cache.get("portfolio_returns") is None:
            raise RuntimeError(f"Call .run() before .{method}().")

    def _run_factor_regression(self) -> None:
        """Fit the factor model and regress portfolio returns against factor returns."""
        portfolio_returns = self._cache["portfolio_returns"]
        benchmark_series = self._cache["benchmark"]

        # Fit the factor model on the universe
        universe_returns = self._universe.returns
        if benchmark_series is not None:
            universe_returns = universe_returns.reindex(benchmark_series.index)

        self._factor_model.fit(
            returns=universe_returns,
            benchmark_returns=benchmark_series,
            close=self._universe.close.reindex(universe_returns.index),
        )

        # We want the time series of factor returns; use benchmark as market proxy
        # and build factor return streams from exposures
        factor_return_streams: dict[str, pd.Series] = {}

        common_idx = portfolio_returns.index
        if benchmark_series is not None:
            factor_return_streams["market"] = benchmark_series.reindex(common_idx).fillna(0.0)

        # Build factor_returns matrix for regression
        factor_df = pd.DataFrame(factor_return_streams).reindex(common_idx).fillna(0.0)
        if factor_df.empty or factor_df.shape[1] == 0:
            return

        y = portfolio_returns.reindex(common_idx).fillna(0.0).values
        x_mat = np.column_stack([np.ones(len(y)), factor_df.values])
        factor_labels = ["alpha"] + list(factor_df.columns)

        try:
            coeffs, residuals_ss, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
            y_hat = x_mat @ coeffs
            ss_tot = np.sum((y - y.mean()) ** 2)
            ss_res = np.sum((y - y_hat) ** 2)
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            # T-stats: (coeff / se), se = sqrt(diag(sigma^2 * (X'X)^-1))
            n, p = x_mat.shape
            sigma2 = ss_res / max(n - p, 1)
            try:
                xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
                se = np.sqrt(np.diag(sigma2 * xtx_inv))
                t_stats = coeffs / np.where(se > 0, se, float("nan"))
            except np.linalg.LinAlgError:
                t_stats = np.full_like(coeffs, float("nan"))

            self._cache["factor_regression"] = {
                "coefficients": pd.Series(coeffs, index=factor_labels),
                "t_stats": pd.Series(t_stats, index=factor_labels),
                "r_squared": float(r_squared),
                "residual_returns": pd.Series(y - y_hat, index=common_idx),
            }
        except np.linalg.LinAlgError:
            pass

    def __repr__(self) -> str:
        n = len(self._strategies)
        ran = self._cache.get("portfolio_returns") is not None
        return f"PortfolioStudy(name={self._name!r}, n_strategies={n}, ran={ran})"


# ---------------------------------------------------------------------------
# Portfolio-level weighting functions (strategy sleeve weighting)
# Signature: (strategy_returns_df: pd.DataFrame) -> pd.Series[float]
# Returns a Series of weights indexed by strategy name (columns of strategy_returns_df).
# These are static weights applied uniformly across time.
# ---------------------------------------------------------------------------


def apply_equal_strategies(strategy_returns_df: pd.DataFrame) -> pd.Series:
    """Equal weight across all strategy sleeves."""
    n = len(strategy_returns_df.columns)
    return pd.Series({col: 1.0 / n for col in strategy_returns_df.columns})


def _apply_equal_vol_strategies(strategy_returns_df: pd.DataFrame, window: int = 126) -> pd.Series:
    """Weight strategies by inverse realized volatility over the full backtest period."""
    vol = strategy_returns_df.std()
    inv_vol = 1.0 / vol.clip(lower=1e-8)
    return inv_vol / inv_vol.sum()


def _apply_equal_sharpe_strategies(
    strategy_returns_df: pd.DataFrame, window: int = 126
) -> pd.Series:
    """Weight strategies by absolute Sharpe ratio over the full backtest period."""
    sharpe = (strategy_returns_df.mean() / strategy_returns_df.std().clip(lower=1e-8)) * np.sqrt(
        252
    )
    abs_sharpe = sharpe.abs().clip(lower=0.0)
    total = abs_sharpe.sum()
    if total == 0:
        return apply_equal_strategies(strategy_returns_df)
    return abs_sharpe / total


def _apply_optimal_strategies(
    strategy_returns_df: pd.DataFrame, window: int = 126, gamma: float = 1.0
) -> pd.Series:
    """Mean-variance optimal weights across strategy sleeves (ridge-regularized)."""
    r = strategy_returns_df.dropna()
    if len(r) < 10:
        return apply_equal_strategies(strategy_returns_df)
    mu = r.mean().values
    sigma = r.cov().values
    try:
        ridge = gamma * np.diag(sigma).mean() * np.eye(len(sigma))
        w = np.linalg.solve(sigma + ridge, mu)
        w = w / np.abs(w).sum()
        return pd.Series(w, index=strategy_returns_df.columns)
    except np.linalg.LinAlgError:
        return apply_equal_strategies(strategy_returns_df)


# ---------------------------------------------------------------------------
# Position combination
# ---------------------------------------------------------------------------


def _combine_positions(
    strategy_positions: dict[str, pd.DataFrame],
    weights: pd.Series,
    renormalize: bool = False,
) -> pd.DataFrame:
    """Combine per-strategy position DataFrames into a single portfolio position DataFrame.

    Steps:
        1. Union all ticker columns.
        2. Reindex each strategy to the common date intersection and full ticker set.
           Missing positions are treated as 0.0 (not held) rather than NaN.
        3. Weighted sum.
        4. Optionally renormalize per day so abs(w).sum() == 1.0 (off by default).

    Args:
        strategy_positions: Dict of strategy label -> positions DataFrame.
        weights:            Series of strategy weights indexed by label.
        renormalize:        If True, rescale combined rows so abs sum == 1.0 each day.
                            Default False — preserves the dollar exposure implied by the
                            individual strategy position sizes and the sleeve weights.
    """
    # Common date index: intersection across all strategies
    date_idx = None
    for positions in strategy_positions.values():
        if date_idx is None:
            date_idx = positions.index
        else:
            date_idx = date_idx.intersection(positions.index)

    # Union of all ticker columns
    all_tickers: pd.Index = pd.Index([])
    for positions in strategy_positions.values():
        all_tickers = all_tickers.union(positions.columns)

    combined = pd.DataFrame(0.0, index=date_idx, columns=all_tickers)
    for label, positions in strategy_positions.items():
        w = float(weights.get(label, 0.0))
        if w == 0.0:
            continue
        aligned = positions.reindex(index=date_idx, columns=all_tickers).fillna(0.0)
        combined += w * aligned

    if renormalize:
        abs_sum = combined.abs().sum(axis=1).replace(0.0, float("nan"))
        combined = combined.div(abs_sum, axis=0).fillna(0.0)

    return combined
