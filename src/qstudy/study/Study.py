"""Study: a chainable pipeline for running cross-sectional equity backtests."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from qstudy.data.loader import StudyData
from qstudy.signals.factors import residualize
from qstudy.signals.filters import (
    momentum_context_filter,
    vix_contango_filter,
    vol_filter,
    volume_zscore_filter,
)
import qstudy.study.engine as engine
import qstudy.study.metrics as metrics
from qstudy.study.portfolio import (
    build_long_only as _build_long_only,
    build_long_short_positions,
    liquidity_filter,
    rebalance,
)
from qstudy.study.weighting import (
    apply_equal,
    apply_equal_sharpe,
    apply_equal_vol,
    apply_optimal,
)


class Study:
    """Chainable pipeline for cross-sectional equity backtests.

    Usage::

        universe_data = qs.download(SP500, "2018-01-01", "2024-12-31")
        benchmark_data = qs.download(["SPY"], "2018-01-01", "2024-12-31")
        factors_data = qs.download(["SPY", "QQQ"], "2018-01-01", "2024-12-31")

        study = (
            Study(universe=universe_data, benchmark=benchmark_data, factors=factors_data)
            .mean_reversion(window=20)
            .add_liquidity_filter(top_n=300)
            .build_long_short(n_long=25, n_short=25)
            .run()
        )
        study.report()
        study.to_csv("returns.csv")
        study.save("my_study.pkl")
        study2 = Study.from_cache("my_study.pkl")

    Pipeline sequence enforced at :meth:`run` time:
        1. [optional] Residualize returns
        2. Base signal (required)
        3. Signal filters (zero or more, in declaration order)
        4. Position builder (required)
        5. Position scalers (zero or more)
        6. Weighting scheme (optional, default = equal dollar)
        7. :func:`engine.run` -> portfolio returns
        8. :func:`metrics.summary` -> summary metrics

    Custom function contracts:

    - ``base_signal(fn)`` -- ``fn(**cache) -> pd.DataFrame``
    - ``add_filter(fn)`` -- ``fn(signal, **cache) -> pd.DataFrame``
    - ``scale_returns(fn)`` -- ``fn(positions, **cache) -> pd.DataFrame``

    Cache is passed as a shallow copy; you can read cache values freely but must not
    mutate the DataFrames in place. Reassigning cache keys inside your function has no
    effect on the Study state.

    Note on serialization: :meth:`save` pickles only the cache (DataFrames and scalars),
    not the pipeline definition or custom callables. :meth:`from_cache` restores a Study
    with the cache pre-populated, allowing :meth:`report` and :meth:`to_csv` to work without
    re-running the pipeline.
    """

    def __init__(
        self,
        universe: StudyData,
        benchmark: StudyData | None = None,
        factors: StudyData | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            universe:  :class:`~qstudy.data.loader.StudyData` for the trading universe.
                       Returned by ``qs.download(tickers, start, end)``.
            benchmark: Optional :class:`~qstudy.data.loader.StudyData` for the benchmark
                       (e.g. ``qs.download(["SPY"], ...)``). Used for metrics and/or
                       residualization when no factors are provided.
            factors:   Optional :class:`~qstudy.data.loader.StudyData` for residualization
                       factors (takes priority over benchmark).
            name:      Optional label shown in the tqdm progress bar.
        """
        if not isinstance(universe, StudyData):
            raise TypeError(
                "universe must be a StudyData object returned by qs.download(). "
                f"Got {type(universe).__name__}."
            )

        self._name = name
        self._factors_data = factors

        # Pipeline state
        self._steps: list[tuple[str, Callable]] = []
        self._residualize: bool = False
        self._weighting_fn: Callable | None = None

        # Align all indexes to the universe date range
        idx = universe.returns.index
        if benchmark is not None:
            idx = idx.intersection(benchmark.returns.index)
        if factors is not None:
            idx = idx.intersection(factors.returns.index)

        self._cache: dict = {
            "close": universe.close.reindex(idx),
            "volume": universe.volume.reindex(idx),
            "returns": universe.returns.reindex(idx),
            "log_returns": universe.log_returns.reindex(idx),
            "benchmark": None,
            "factor_returns": None,
            "residual_returns": None,
            "_active_returns": None,
            "base_signal": None,
            "signal": None,
            "positions": None,
            "portfolio_returns": None,
            "metrics_summary": None,
            "_signal_history": [],
            "_position_history": [],
            "_liquidity_mask": None,
        }

        if benchmark is not None:
            bm_returns = benchmark.returns.reindex(idx)
            if bm_returns.shape[1] == 1:
                self._cache["benchmark"] = bm_returns.iloc[:, 0]
            else:
                self._cache["benchmark"] = bm_returns.iloc[:, 0]

        if factors is not None:
            self._cache["factor_returns"] = factors.returns.reindex(idx)

    # ------------------------------------------------------------------
    # Signal source methods (exactly one required)
    # ------------------------------------------------------------------

    def mean_reversion(self, window: int = 20) -> Study:
        """Use short-term mean reversion as the base signal.

        Signal = -rolling_mean(returns, window). Recent losers get positive signal.

        Args:
            window: Lookback in trading days.
        """
        def fn(**cache):
            return -cache["_active_returns"].rolling(window).mean()

        self._set_base_signal(fn, label=f"mean_reversion(window={window})")
        return self

    def momentum(self, window: int = 60) -> Study:
        """Use cross-sectional momentum as the base signal.

        Signal = rolling_mean(returns, window). Recent winners get positive signal.

        Args:
            window: Lookback in trading days.
        """
        def fn(**cache):
            return cache["_active_returns"].rolling(window).mean()

        self._set_base_signal(fn, label=f"momentum(window={window})")
        return self

    def base_signal(self, fn: Callable) -> Study:
        """Use a custom function as the base signal.

        Args:
            fn: ``fn(**cache) -> pd.DataFrame``. Receives a shallow copy of the study
                cache. Must return a signal DataFrame (dates x tickers).
        """
        self._set_base_signal(fn, label=getattr(fn, "__name__", "custom_signal"))
        return self

    # ------------------------------------------------------------------
    # Pre-signal: residualization
    # ------------------------------------------------------------------

    def residualize_returns(self) -> Study:
        """Residualize returns against factors or benchmark before computing the signal.

        Requires ``factors`` or ``benchmark`` to be set in the constructor.
        If both are set, factors take priority.

        The residualized returns are stored in ``cache["residual_returns"]`` and used
        as ``_active_returns`` by the built-in signal generators.
        """
        if self._cache["factor_returns"] is None and self._cache["benchmark"] is None:
            raise ValueError(
                "residualize_returns() requires either factors= or benchmark= "
                "to be specified in the Study constructor."
            )
        self._residualize = True
        return self

    # ------------------------------------------------------------------
    # Signal filters
    # ------------------------------------------------------------------

    def add_filter(self, fn: Callable) -> Study:
        """Add a custom signal filter.

        Args:
            fn: ``fn(signal, **cache) -> pd.DataFrame``. Receives the current signal
                DataFrame and a shallow copy of the cache. Must return a signal DataFrame
                of the same shape.
        """
        self._append_signal_filter(fn, label=getattr(fn, "__name__", "custom_filter"))
        return self

    def add_liquidity_filter(self, top_n: int = 250, window: int = 60) -> Study:
        """Zero out signals for tickers outside the top_n most liquid assets.

        Also stores the liquidity mask in the cache so that the engine uses
        masked returns (matching the manual ``ret_filtered = returns.where(liq_mask)`` pattern).

        Args:
            top_n:  Keep only the top N assets by rolling dollar volume.
            window: Lookback for rolling average dollar volume.
        """
        def fn(signal, **cache):
            mask = liquidity_filter(cache["close"], cache["volume"], top_n=top_n, window=window)
            # Store the mask so run() can apply it to returns before the engine
            self._cache["_liquidity_mask"] = mask
            return signal.where(mask.reindex(columns=signal.columns))

        self._append_signal_filter(fn, label=f"liquidity_filter(top_n={top_n})")
        return self

    def add_vol_filter(
        self,
        vol_window: int = 40,
        quantile: float = 0.75,
        keep: str = "low",
    ) -> Study:
        """Zero out signal where realized vol is above/below a cross-sectional quantile.

        Uses residual returns when residualize_returns() was called, otherwise raw returns.

        Args:
            vol_window: Lookback for realized vol.
            quantile:   Cross-sectional threshold percentile.
            keep:       'low' keeps assets below the quantile; 'high' keeps assets above.
        """
        def fn(signal, **cache):
            returns = cache["residual_returns"] if cache.get("residual_returns") is not None else cache["returns"]
            return vol_filter(
                signal, returns, vol_window=vol_window, quantile=quantile, keep=keep
            )

        self._append_signal_filter(fn, label=f"vol_filter(q={quantile}, keep={keep})")
        return self

    def add_volume_zscore_filter(
        self,
        window: int = 10,
        min_zscore_quantile: float = 0.65,
    ) -> Study:
        """Zero out signal where volume z-score is below the cross-sectional quantile.

        Args:
            window:              Lookback for rolling volume z-score.
            min_zscore_quantile: Keep assets above this cross-sectional quantile.
        """
        def fn(signal, **cache):
            return volume_zscore_filter(
                signal, cache["volume"], window=window, min_zscore_quantile=min_zscore_quantile
            )

        self._append_signal_filter(fn, label=f"volume_zscore_filter(q={min_zscore_quantile})")
        return self

    def add_momentum_context_filter(
        self,
        window: int = 15,
        max_abs_quantile: float = 0.75,
    ) -> Study:
        """Zero out signal where medium-term momentum magnitude exceeds the quantile.

        Useful for mean-reversion studies: removes strongly trending assets.
        Uses residual returns when residualize_returns() was called, otherwise raw returns.

        Args:
            window:           Lookback for medium-term momentum.
            max_abs_quantile: Filter out assets with abs-momentum above this quantile.
        """
        def fn(signal, **cache):
            returns = cache["residual_returns"] if cache.get("residual_returns") is not None else cache["returns"]
            return momentum_context_filter(
                signal, returns, window=window, max_abs_quantile=max_abs_quantile
            )

        self._append_signal_filter(fn, label=f"momentum_context_filter(q={max_abs_quantile})")
        return self

    def add_vix_contango_filter(
        self,
        vix_close: pd.DataFrame,
        window: int = 1,
    ) -> Study:
        """Zero out signals on dates where the VIX term structure is not in contango.

        Args:
            vix_close: Close prices for VIX indexes (dates x vix_tickers).
                       Download with ``qs.download(qs.VOL_INDEXES, ...)["close"]``.
            window:    Require contango for N consecutive days.
        """
        def fn(signal, **cache):
            return vix_contango_filter(signal, vix_close, window=window)

        self._append_signal_filter(fn, label="vix_contango_filter")
        return self

    # ------------------------------------------------------------------
    # Position builders (exactly one required)
    # ------------------------------------------------------------------

    def build_long_short(
        self,
        n_long: int = 25,
        n_short: int = 25,
        rebalance_every: int = 1,
    ) -> Study:
        """Build a dollar-neutral long/short portfolio.

        Args:
            n_long:          Number of long positions.
            n_short:         Number of short positions.
            rebalance_every: Rebalance every N trading days (1 = daily).
        """
        def fn(signal):
            pos = build_long_short_positions(signal, n_long=n_long, n_short=n_short)
            return rebalance(pos, every=rebalance_every)

        self._set_position_builder(fn, label=f"build_long_short(n_long={n_long}, n_short={n_short})")
        return self

    def build_long_only(self, n: int = 10, rebalance_every: int = 1) -> Study:
        """Build a long-only equal-weighted portfolio.

        Args:
            n:               Number of long positions.
            rebalance_every: Rebalance every N trading days (1 = daily).
        """
        def fn(signal):
            pos = _build_long_only(signal, n=n)
            return rebalance(pos, every=rebalance_every)

        self._set_position_builder(fn, label=f"build_long_only(n={n})")
        return self

    # ------------------------------------------------------------------
    # Position processing
    # ------------------------------------------------------------------

    def scale_returns(self, fn: Callable) -> Study:
        """Add a custom position scaler.

        Args:
            fn: ``fn(positions, **cache) -> pd.DataFrame``. Receives the current positions
                DataFrame and a shallow copy of the cache. Must return a positions DataFrame
                of the same shape.
        """
        self._steps.append(("position_scaler", fn))
        return self

    # ------------------------------------------------------------------
    # Weighting schemes
    # ------------------------------------------------------------------

    def weight_equal(self) -> Study:
        """Equal dollar weights (default -- no-op, positions already normalized)."""
        self._set_weighting(apply_equal, label="weight_equal")
        return self

    def weight_equal_vol(self, vol_window: int = 60) -> Study:
        """Scale positions inversely proportional to realized volatility.

        Args:
            vol_window: Lookback for realized vol calculation.
        """
        def fn(positions, **cache):
            return apply_equal_vol(positions, vol_window=vol_window, **cache)

        self._set_weighting(fn, label=f"weight_equal_vol(window={vol_window})")
        return self

    def weight_equal_sharpe(self, window: int = 126) -> Study:
        """Scale positions by rolling absolute Sharpe ratio.

        Args:
            window: Lookback for rolling Sharpe calculation.
        """
        def fn(positions, **cache):
            return apply_equal_sharpe(positions, window=window, **cache)

        self._set_weighting(fn, label=f"weight_equal_sharpe(window={window})")
        return self

    def weight_optimal(self, window: int = 126, gamma: float = 1.0) -> Study:
        """Rolling mean-variance optimal weights (ridge-regularized).

        Note: runs a per-date optimization loop and may be slow for large universes.

        Args:
            window: Rolling lookback in trading days.
            gamma:  Ridge regularization multiplier.
        """
        def fn(positions, **cache):
            return apply_optimal(positions, window=window, gamma=gamma, **cache)

        self._set_weighting(fn, label=f"weight_optimal(window={window}, gamma={gamma})")
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> Study:
        """Execute the pipeline and compute portfolio returns and metrics.

        Must be called after all pipeline methods have been chained. Returns self
        for further chaining (e.g. ``.run().report()``).

        Raises:
            RuntimeError: If no base signal or position builder has been defined.
        """
        self._validate_pipeline()

        stages = self._build_stage_list()
        with tqdm(total=len(stages), desc=self._name or "Study.run") as pbar:
            # Stage: residualize (optional)
            if self._residualize:
                pbar.set_postfix({"stage": "residualize"})
                self._run_residualize()
                pbar.update(1)

            # Stage: set _active_returns
            pbar.set_postfix({"stage": "setup"})
            self._cache["_active_returns"] = (
                self._cache["residual_returns"]
                if self._cache["residual_returns"] is not None
                else self._cache["returns"]
            )
            pbar.update(1)

            # Pipeline steps
            for step_type, fn in self._steps:
                pbar.set_postfix({"stage": step_type})
                self._execute_step(step_type, fn)
                pbar.update(1)

            # Weighting
            if self._weighting_fn is not None:
                pbar.set_postfix({"stage": "weighting"})
                cache_kw = {k: v for k, v in self._cache.items() if k != "positions"}
                self._cache["positions"] = self._weighting_fn(
                    self._cache["positions"], **cache_kw
                )
                self._cache["_position_history"].append(
                    ("weighting", self._cache["positions"].copy())
                )
                pbar.update(1)

            # Backtest engine
            pbar.set_postfix({"stage": "backtest"})
            returns_for_engine = self._cache["returns"]
            liq_mask = self._cache.get("_liquidity_mask")
            if liq_mask is not None:
                returns_for_engine = returns_for_engine.where(liq_mask)
            self._cache["portfolio_returns"] = engine.run(
                self._cache["positions"], returns_for_engine
            )
            pbar.update(1)

            # Metrics
            pbar.set_postfix({"stage": "metrics"})
            self._cache["metrics_summary"] = metrics.summary(
                self._cache["portfolio_returns"],
                positions=self._cache["positions"],
                benchmark=self._cache["benchmark"],
            )
            pbar.update(1)

        return self

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, rolling_window: int = 90) -> Study:
        """Print a metrics summary and show the equity curve / drawdown / rolling Sharpe chart.

        Requires :meth:`run` (or :meth:`from_cache`) to have been called first.

        Args:
            rolling_window: Window for the rolling Sharpe panel.
        """
        from qstudy.charts.summary import summary_plot  # lazy import to avoid circular dependency

        self._require_run("report")
        print(self._cache["metrics_summary"].to_string())
        summary_plot(self._cache["portfolio_returns"], rolling_window=rolling_window)
        return self

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @property
    def cache(self) -> dict:
        """The study cache containing all intermediate and final DataFrames."""
        return self._cache

    def to_csv(self, path: str | Path) -> Study:
        """Write portfolio returns to a CSV file.

        Requires :meth:`run` to have been called first.

        Args:
            path: Destination file path.
        """
        self._require_run("to_csv")
        pd.DataFrame({"portfolio_returns": self._cache["portfolio_returns"]}).to_csv(
            Path(path), index=True
        )
        return self

    def save(self, path: str | Path) -> Study:
        """Pickle the study cache (DataFrames, metrics, positions) to disk.

        Only the cache is saved -- the pipeline definition and custom callables are
        not persisted. Use :meth:`from_cache` to reload and call :meth:`report` etc.

        Args:
            path: Destination file path (e.g. "my_study.pkl").
        """
        with open(Path(path), "wb") as f:
            pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    @classmethod
    def from_cache(cls, path: str | Path) -> Study:
        """Load a Study from a previously saved cache file.

        Returns a Study with the cache pre-populated. You can call :meth:`report`,
        :meth:`to_csv`, or inspect ``study.cache`` -- but you cannot call :meth:`run`
        again without redefining the pipeline.

        Args:
            path: Path to a file saved with :meth:`save`.
        """
        with open(Path(path), "rb") as f:
            cache = pickle.load(f)
        if not isinstance(cache, dict):
            raise TypeError(f"Expected a cache dict, got {type(cache)}")
        obj = cls.__new__(cls)
        obj._name = None
        obj._factors_data = None
        obj._steps = []
        obj._residualize = False
        obj._weighting_fn = None
        obj._cache = cache
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_base_signal(self, fn: Callable, label: str) -> None:
        existing = [i for i, s in enumerate(self._steps) if s[0] == "base_signal"]
        if existing:
            warnings.warn(f"Replacing existing base signal with '{label}'.", stacklevel=3)
            for i in reversed(existing):
                self._steps.pop(i)
        self._steps.insert(0, ("base_signal", fn))

    def _append_signal_filter(self, fn: Callable, label: str) -> None:  # noqa: ARG002
        last_sig_idx = max(
            (i for i, s in enumerate(self._steps) if s[0] in ("base_signal", "signal_filter")),
            default=-1,
        )
        self._steps.insert(last_sig_idx + 1, ("signal_filter", fn))

    def _set_position_builder(self, fn: Callable, label: str) -> None:
        existing = [i for i, s in enumerate(self._steps) if s[0] == "position_builder"]
        if existing:
            warnings.warn(f"Replacing existing position builder with '{label}'.", stacklevel=3)
            for i in reversed(existing):
                self._steps.pop(i)
        self._steps.append(("position_builder", fn))

    def _set_weighting(self, fn: Callable, label: str) -> None:
        if self._weighting_fn is not None:
            warnings.warn(f"Replacing existing weighting scheme with '{label}'.", stacklevel=3)
        self._weighting_fn = fn

    def _validate_pipeline(self) -> None:
        types = [s[0] for s in self._steps]
        if "base_signal" not in types:
            raise RuntimeError(
                "No signal source defined. Call .mean_reversion(), .momentum(), "
                "or .base_signal(fn) before .run()."
            )
        if "position_builder" not in types:
            raise RuntimeError(
                "No position builder defined. Call .build_long_short() or "
                ".build_long_only() before .run()."
            )
        base_idx = types.index("base_signal")
        filter_idxs = [i for i, t in enumerate(types) if t == "signal_filter"]
        builder_idx = types.index("position_builder")
        if filter_idxs and min(filter_idxs) < base_idx:
            raise RuntimeError(
                "Signal filters cannot appear before the base signal in the pipeline."
            )
        if builder_idx < base_idx:
            raise RuntimeError("Position builder cannot appear before the base signal.")

    def _build_stage_list(self) -> list[str]:
        stages: list[str] = []
        if self._residualize:
            stages.append("residualize")
        stages.append("setup")
        for step_type, _ in self._steps:
            stages.append(step_type)
        if self._weighting_fn is not None:
            stages.append("weighting")
        stages.append("backtest")
        stages.append("metrics")
        return stages

    def _run_residualize(self) -> None:
        returns = self._cache["returns"]
        if self._cache["factor_returns"] is not None:
            factor_returns = self._cache["factor_returns"]
        elif self._cache["benchmark"] is not None:
            factor_returns = self._cache["benchmark"].to_frame()
        else:
            raise ValueError(
                "residualize_returns() requires factors= or benchmark= in the Study constructor."
            )
        residuals, _, _ = residualize(returns, factor_returns)
        self._cache["residual_returns"] = residuals

    def _execute_step(self, step_type: str, fn: Callable) -> None:
        snapshot = self._cache.copy()

        if step_type == "base_signal":
            sig = fn(**snapshot)
            self._cache["base_signal"] = sig
            self._cache["signal"] = sig.copy()
            self._cache["_signal_history"].append(("base_signal", sig.copy()))

        elif step_type == "signal_filter":
            # Remove "signal" from snapshot to avoid collision with the positional arg
            cache_kw = {k: v for k, v in snapshot.items() if k != "signal"}
            sig = fn(self._cache["signal"], **cache_kw)
            self._cache["signal"] = sig
            self._cache["_signal_history"].append((fn.__name__, sig.copy()))

        elif step_type == "position_builder":
            pos = fn(self._cache["signal"])
            self._cache["positions"] = pos
            self._cache["_position_history"].append(("position_builder", pos.copy()))

        elif step_type == "position_scaler":
            # Remove "positions" from snapshot to avoid collision with the positional arg
            cache_kw = {k: v for k, v in snapshot.items() if k != "positions"}
            pos = fn(self._cache["positions"], **cache_kw)
            self._cache["positions"] = pos
            self._cache["_position_history"].append((fn.__name__, pos.copy()))

    def _require_run(self, method: str) -> None:
        if self._cache.get("portfolio_returns") is None:
            raise RuntimeError(f"Call .run() before .{method}().")

    def __repr__(self) -> str:
        steps = [s[0] for s in self._steps]
        ran = self._cache.get("portfolio_returns") is not None
        return (
            f"Study(name={self._name!r}, steps={steps}, "
            f"weighting={self._weighting_fn is not None}, ran={ran})"
        )
