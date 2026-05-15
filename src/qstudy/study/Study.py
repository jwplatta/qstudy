"""Study: a chainable pipeline for running cross-sectional equity backtests."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import qstudy.study.engine as engine
import qstudy.study.metrics as metrics
from qstudy.data.loader import StudyData
from qstudy.signals import transforms
from qstudy.signals.factors import BarraLiteFactorModel, residualize
from qstudy.signals.filters import (
    momentum_context_filter,
    vix_contango_filter,
    vol_filter,
    volume_zscore_filter,
)
from qstudy.study.metrics import StudyMetrics
from qstudy.study.portfolio import (
    build_long_only as _build_long_only,
)
from qstudy.study.portfolio import (
    build_long_short_positions,
    build_proportional_positions,
    liquidity,
    rebalance,
    rebalance_on,
)
from qstudy.study.weighting import apply_equal


class Study:
    """Chainable pipeline for cross-sectional equity backtests.

    Usage::

        universe_data = qs.download(SP500, "2018-01-01", "2024-12-31")
        benchmark_data = qs.download(["SPY"], "2018-01-01", "2024-12-31")
        factors_data = qs.download(["SPY", "QQQ"], "2018-01-01", "2024-12-31")

        def my_signal(**cache):
            return -cache["residual_returns"].rolling(20).mean().shift(1)

        study = (
            Study(universe=universe_data, benchmark=benchmark_data, factors=factors_data)
            .residualize_returns()
            .base_signal(my_signal)
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
        universe: StudyData | None = None,
        benchmark: StudyData | None = None,
        factors: StudyData | None = None,
        name: str | None = None,
        cost_bps: float = 0.0,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            universe:  :class:`~qstudy.data.loader.StudyData` for the trading universe.
                       Returned by ``qs.download(tickers, start, end)``.
                       May be omitted when the Study will be run via a
                       :class:`~qstudy.study.PortfolioStudy` which injects shared data.
            benchmark: Optional :class:`~qstudy.data.loader.StudyData` for the benchmark
                       (e.g. ``qs.download(["SPY"], ...)``). Used for metrics and/or
                       residualization when no factors are provided.
            factors:   Optional :class:`~qstudy.data.loader.StudyData` for residualization
                       factors (takes priority over benchmark).
            name:      Optional label shown in the tqdm progress bar.
            verbose:   If False, suppress the tqdm progress bar. Default True.
        """
        if universe is not None and not isinstance(universe, StudyData):
            raise TypeError(
                "universe must be a StudyData object returned by qs.download(). "
                f"Got {type(universe).__name__}."
            )

        self._name = name
        self._factors_data = factors
        self._cost_bps: float = float(cost_bps)
        self._verbose: bool = verbose

        # Pipeline state
        self._steps: list[tuple[str, str, Callable]] = []
        self._residualize: bool = False
        self._factor_model: BarraLiteFactorModel | None = None
        self._tradeable_constraint_fns: list[Callable] = []

        # Initialize empty cache; populated below if universe is provided, or later via _inject_data
        self._cache: dict = {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
            "volume": None,
            "returns": None,
            "log_returns": None,
            "benchmark": None,
            "factor_returns": None,
            "residual_returns": None,
            "_active_returns": None,
            "base_signal": None,
            "signal": None,
            "positions": None,
            "portfolio_returns": None,
            "gross_portfolio_returns": None,
            "metrics_summary": None,
            "_signal_history": [],
            "_position_history": [],
            "_liquidity_mask": None,
            "_tradeable_mask": None,
            "factor_exposures": None,
            "factor_model": None,
            "_xs_daily_r2": None,
        }
        self._data_injected: bool = False

        if universe is not None:
            self._inject_data(universe, benchmark, factors)

    def _inject_data(
        self,
        universe: StudyData,
        benchmark: StudyData | None = None,
        factors: StudyData | None = None,
    ) -> None:
        """Populate (or replace) cache data from StudyData objects.

        Called automatically from ``__init__`` when universe is provided, and
        called by :class:`~qstudy.study.PortfolioStudy` before running each
        strategy to ensure all strategies share the same aligned dataset.

        Resets signal/position history so the study can be re-run cleanly.

        When called by :class:`~qstudy.study.PortfolioStudy`, ``factors`` is ``None``
        because the portfolio only shares universe and benchmark.  If the study was
        originally constructed with its own ``factors=`` argument (stored in
        ``self._factors_data``), those per-strategy factors are used automatically.
        """
        # Per-strategy factors take priority; fall back to whatever was passed in
        effective_factors = self._factors_data if self._factors_data is not None else factors

        # Canonical alignment: intersection of all provided date ranges
        idx = universe.returns.index
        if benchmark is not None:
            idx = idx.intersection(benchmark.returns.index)
        if effective_factors is not None:
            idx = idx.intersection(effective_factors.returns.index)

        self._cache["open"] = universe.open.reindex(idx).copy()
        self._cache["high"] = universe.high.reindex(idx).copy()
        self._cache["low"] = universe.low.reindex(idx).copy()
        self._cache["close"] = universe.close.reindex(idx).copy()
        self._cache["volume"] = universe.volume.reindex(idx).copy()
        self._cache["returns"] = universe.returns.reindex(idx).copy()
        self._cache["log_returns"] = universe.log_returns.reindex(idx).copy()
        self._cache["benchmark"] = (
            benchmark.returns.reindex(idx).iloc[:, 0].copy() if benchmark is not None else None
        )
        self._cache["factor_returns"] = (
            effective_factors.returns.reindex(idx).copy() if effective_factors is not None else None
        )
        # Reset derived/output fields so run() starts fresh
        for key in (
            "residual_returns",
            "_active_returns",
            "base_signal",
            "signal",
            "positions",
            "portfolio_returns",
            "gross_portfolio_returns",
            "metrics_summary",
            "_liquidity_mask",
            "_tradeable_mask",
            "factor_exposures",
            "factor_model",
            "_xs_daily_r2",
        ):
            self._cache[key] = None
        self._cache["_signal_history"] = []
        self._cache["_position_history"] = []
        self._data_injected = True

    # ------------------------------------------------------------------
    # Signal source methods (exactly one required)
    # ------------------------------------------------------------------

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
        self._residualize = True
        return self

    def with_transaction_costs(self, cost_bps: float) -> Study:
        """Set a per-trade transaction cost assumption.

        Applied after the final portfolio positions are determined.  Daily net returns
        are computed as: ``net_ret = gross_ret - turnover * (cost_bps / 10_000)``,
        where turnover is the one-way daily turnover of the final position DataFrame.
        Costs are attributed to the same day as the position change, consistent with
        the engine's convention that ``position[T-1]`` generates ``return[T]``.

        Args:
            cost_bps: One-way cost in basis points per dollar traded
                      (e.g. ``10`` = 10 bps = 0.10%).  Default is ``0`` (no costs).
        """
        self._cost_bps = float(cost_bps)
        return self

    def add_factor_model(
        self,
        model: str = "barra-lite",
        factors: list[str] = ("market", "sector"),
        sector_map: dict | None = None,
        beta_window: int = 60,
        momentum_window: int = 20,
        vol_window: int = 20,
    ) -> Study:
        """Attach a cross-sectional factor model for residualization and neutralization.

        The factor model is fitted at :meth:`run` time using data from the study cache.
        When set, :meth:`residualize_returns` uses this model instead of the ETF time-series
        OLS path. The fitted exposures are stored in ``cache["factor_exposures"]`` for use by
        :meth:`neutralize_signal` and :meth:`neutralize_positions`.

        Requires ``benchmark=`` to be set in the constructor (used as the market factor).

        Args:
            model:            Model type. Only ``"barra-lite"`` is supported.
            factors:          Factor names to include. Supported: ``"market"``, ``"sector"``,
                              ``"momentum"``, ``"volatility"``, ``"size"``.
            sector_map:       Dict of ticker -> GICS sector string. Required when ``"sector"``
                              is in ``factors``. Fetch with ``qs.get_sector_map(tickers)``.
            beta_window:      Rolling window for market beta estimation (days).
            momentum_window:  Window for momentum exposure.
            vol_window:       Window for volatility exposure.
        """
        if model != "barra-lite":
            raise ValueError(f"Unknown factor model '{model}'. Only 'barra-lite' is supported.")
        self._factor_model = BarraLiteFactorModel(
            factors=list(factors),
            beta_window=beta_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            sector_map=sector_map,
        )
        return self

    # ------------------------------------------------------------------
    # Signal filters, transforms, and neutralization
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

    def transform_signal(self, fn: Callable) -> Study:
        """Apply a signal transformation that changes signal geometry.

        Use this for operations that reshape all signal values but keep the same candidates:
        cross-sectional demeaning, z-scoring, clipping, orthogonalization, etc.

        Args:
            fn: ``fn(signal, **cache) -> pd.DataFrame``. Same shape, different geometry.
        """
        self._append_signal_filter(fn, label=getattr(fn, "__name__", "transform_signal"))
        return self

    def winsorize(self, lower: float = 0.05, upper: float = 0.95) -> Study:
        """Clip cross-sectional outliers to percentile bounds on each date.

        Args:
            lower: Lower percentile bound (e.g. 0.05 = 5th percentile).
            upper: Upper percentile bound (e.g. 0.95 = 95th percentile).
        """

        def fn(signal, **cache):
            return transforms.winsorize(signal, lower=lower, upper=upper)

        fn.__name__ = f"winsorize(lower={lower}, upper={upper})"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def truncate(self, lower: float = 0.05, upper: float = 0.95) -> Study:
        """Remove cross-sectional outliers by NaN-ing values outside percentile bounds.

        Args:
            lower: Lower percentile bound — values below become NaN.
            upper: Upper percentile bound — values above become NaN.
        """

        def fn(signal, **cache):
            return transforms.truncate(signal, lower=lower, upper=upper)

        fn.__name__ = f"truncate(lower={lower}, upper={upper})"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def rank_transform(self) -> Study:
        """Rank signal cross-sectionally and normalize to [0, 1] on each date.

        Produces a uniform distribution. Use to remove distributional shape,
        fix skew, or when the precise signal value has no meaning beyond order.
        """

        def fn(signal, **cache):
            return transforms.rank_transform(signal)

        fn.__name__ = "rank_transform"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def rank_threshold(self, tail: float = 0.20) -> Study:
        """Rank cross-sectionally then zero out the middle, keeping only the tails.

        Args:
            tail: Fraction to keep on each end. Default 0.20 keeps top/bottom 20%.
        """

        def fn(signal, **cache):
            return transforms.rank_threshold(signal, tail=tail)

        fn.__name__ = f"rank_threshold(tail={tail})"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def inverse_cdf(self) -> Study:
        """Map signal to standard normal quantiles via the inverse CDF on each date.

        Produces a normal distribution with heavier tails than rank_transform.
        """

        def fn(signal, **cache):
            return transforms.inverse_cdf(signal)

        fn.__name__ = "inverse_cdf"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def tanh_scale(self, scale: float = 1.0) -> Study:
        """Soft-clip the signal to (-1, 1) using tanh on each date.

        Args:
            scale: Controls the inflection point. Smaller scale = more compression.
        """

        def fn(signal, **cache):
            return transforms.tanh_scale(signal, scale=scale)

        fn.__name__ = f"tanh_scale(scale={scale})"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def zscore_signal(self) -> Study:
        """Cross-sectional z-score the signal on each date.

        Subtracts the cross-sectional mean and divides by std. Use to normalize
        signal scale before position building.
        """

        def fn(signal, **cache):
            return transforms.zscore(signal)

        fn.__name__ = "zscore_signal"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def demean_signal(self) -> Study:
        """Subtract the cross-sectional mean from the signal on each date.

        Shifts signal rows to sum to approximately zero. First step toward
        dollar neutrality when building proportional-weight positions.
        """

        def fn(signal, **cache):
            return transforms.demean(signal)

        fn.__name__ = "demean_signal"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def filter_signal(self, fn: Callable) -> Study:
        """Apply a signal filter that removes candidates by zeroing their signals.

        Use this for quality or timing gates: realized vol threshold, volume z-score,
        momentum context, etc. The distinction from :meth:`transform_signal` is semantic:
        filters zero out some candidates; transforms reshape all values.

        Args:
            fn: ``fn(signal, **cache) -> pd.DataFrame``. Same shape, some values zeroed.
        """
        self._append_signal_filter(fn, label=getattr(fn, "__name__", "filter_signal"))
        return self

    def neutralize_signal(self, factors: list[str] = ("market", "sector")) -> Study:
        """Orthogonalize the signal against factor exposures cross-sectionally.

        Each day, projects the signal onto the null space of the factor exposure matrix,
        removing systematic factor tilts from the signal. Useful when you want raw returns
        (no residualization) but a factor-neutral signal.

        Requires :meth:`add_factor_model` to have been called.

        Args:
            factors: Factor names to neutralize against. Must be a subset of the factors
                     passed to :meth:`add_factor_model`.
        """
        factors = list(factors)

        def fn(signal, **cache):
            factor_exposures = cache.get("factor_exposures")
            if factor_exposures is None:
                return signal

            neutralized = signal.copy()
            for date in signal.index:
                s_t = signal.loc[date].dropna()
                if s_t.empty:
                    continue

                # Build exposure matrix for requested factors on this date
                parts = []
                for fname in factors:
                    if fname in factor_exposures:
                        parts.append(factor_exposures[fname].loc[date, s_t.index].rename(fname))
                if not parts:
                    continue

                x_t = pd.concat(parts, axis=1).reindex(s_t.index).fillna(0.0)
                # Orthogonal projection: s - X(X'X)^-1 X' s
                try:
                    xtx_inv = pd.DataFrame(
                        np.linalg.pinv(x_t.values.T @ x_t.values),
                        index=x_t.columns,
                        columns=x_t.columns,
                    )
                    proj = x_t.values @ xtx_inv.values @ x_t.values.T @ s_t.values
                    neutralized.loc[date, s_t.index] = s_t.values - proj
                except np.linalg.LinAlgError:
                    pass

            return neutralized

        fn.__name__ = f"neutralize_signal({factors})"
        self._append_signal_filter(fn, label=fn.__name__)
        return self

    def add_tradeable_constraint(self, fn: Callable) -> Study:
        """Add a tradeable universe constraint.

        Tradeable constraints define which assets are eligible on each date. They are
        applied *after* all signal processing (transforms, filters, neutralization) and
        *before* position building. Ineligible assets have their signals zeroed out and
        their returns masked in the backtest engine.

        This is semantically distinct from signal filters (quality preferences on eligible
        assets) and signal transforms (geometry changes).

        Multiple constraints are ANDed together into ``cache["_tradeable_mask"]``.

        Built-in constraint factories in ``qstudy``:
            ``qs.liquidity(top_n=250, window=60)``  — top N by rolling dollar volume
            ``qs.min_price(threshold=5.0)``          — minimum close price
            ``qs.min_adv(threshold=1e6)``            — minimum average daily dollar volume

        Args:
            fn: ``fn(close, volume, returns, **cache) -> pd.DataFrame[bool]``.
        """
        self._tradeable_constraint_fns.append(fn)

        # Wrap as a signal_filter step so it runs exactly where declared in the chain,
        # before position_builder. Also stores the mask in cache["_tradeable_mask"] so
        # position scalers (e.g. equity_curve_regime_scale) can apply it to returns.
        constraint_fn = fn

        def apply_constraint(signal, **cache):
            m = constraint_fn(
                close=cache["close"],
                volume=cache["volume"],
                **{k: v for k, v in cache.items() if k not in ("close", "volume")},
            )
            m = m.reindex(columns=signal.columns).fillna(False)
            existing = cache.get("_tradeable_mask")
            combined = (existing & m) if existing is not None else m
            self._cache["_tradeable_mask"] = combined
            return signal.where(combined, other=float("nan"))

        apply_constraint.__name__ = getattr(fn, "__name__", "tradeable_constraint")
        self._append_signal_filter(apply_constraint, label=apply_constraint.__name__)
        return self

    def add_liquidity_filter(self, top_n: int = 250, window: int = 60) -> Study:
        """Deprecated alias for ``.add_tradeable_constraint(qs.liquidity(top_n, window))``.

        Prefer :meth:`add_tradeable_constraint` with the :func:`~qstudy.study.portfolio.liquidity`
        factory. Unlike the old implementation (which ran before position building as a signal
        filter), tradeable constraints now run after all signal processing.

        Args:
            top_n:  Keep only the top N assets by rolling dollar volume.
            window: Lookback for rolling average dollar volume.
        """
        warnings.warn(
            "add_liquidity_filter() is deprecated. "
            "Use .add_tradeable_constraint(qs.liquidity(top_n, window)) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.add_tradeable_constraint(liquidity(top_n=top_n, window=window))

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
            returns = (
                cache["residual_returns"]
                if cache.get("residual_returns") is not None
                else cache["returns"]
            )
            return vol_filter(signal, returns, vol_window=vol_window, quantile=quantile, keep=keep)

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
            returns = (
                cache["residual_returns"]
                if cache.get("residual_returns") is not None
                else cache["returns"]
            )
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
    ) -> Study:
        """Build a dollar-neutral long/short portfolio.

        Args:
            n_long:  Number of long positions.
            n_short: Number of short positions.
        """

        def fn(signal):
            return build_long_short_positions(signal, n_long=n_long, n_short=n_short)

        self._set_position_builder(
            fn, label=f"build_long_short(n_long={n_long}, n_short={n_short})"
        )
        return self

    def build_positions(self, fn: Callable) -> Study:
        """Build positions using a custom function.

        Use this when the built-in long/short or long-only builders don't fit your needs,
        e.g. proportional signal weighting, volatility-scaled weights, or custom bucketing.

        Your function is responsible for producing a valid weight vector. Two utilities
        are available in ``qstudy`` to help:

        - ``qs.demean_weights(weights)`` — subtracts the cross-sectional mean to achieve
          dollar neutrality (long $ == short $). Use before normalizing for long/short books.
          Do not use for long-only builders.
        - ``qs.normalize_weights(weights)`` — divides by abs sum so the book is fully
          invested (abs weights sum to 1.0). Use as the final step in any custom builder.

        Example::

            def my_builder(signal):
                # proportional signal weighting, dollar-neutral
                weights = signal.fillna(0.0)
                weights = qs.demean_weights(weights)
                return qs.normalize_weights(weights)

            study.build_positions(my_builder)

        Args:
            fn: ``fn(signal) -> pd.DataFrame`` — takes the fully filtered signal DataFrame
                (dates x tickers, NaN = ineligible) and returns a positions DataFrame of the
                same shape.
        """
        self._set_position_builder(fn, label=getattr(fn, "__name__", "custom_positions"))
        return self

    def build_long_only(self, n: int = 10) -> Study:
        """Build a long-only equal-weighted portfolio.

        Args:
            n: Number of long positions.
        """

        def fn(signal):
            return _build_long_only(signal, n=n)

        self._set_position_builder(fn, label=f"build_long_only(n={n})")
        return self

    def build_proportional_positions(self, clip_zscore: float = 3.0) -> Study:
        """Build a dollar-neutral portfolio sized by signal strength.

        Each date's cross-section is z-scored, clipped, demeaned, and normalized
        so stronger signals get larger absolute weights while gross exposure stays
        fixed at 1.0.

        Args:
            clip_zscore: Maximum absolute z-score retained before normalization.
        """

        def fn(signal):
            return build_proportional_positions(signal, clip_zscore=clip_zscore)

        self._set_position_builder(
            fn, label=f"build_proportional_positions(clip_zscore={clip_zscore})"
        )
        return self

    # ------------------------------------------------------------------
    # Position processing
    # ------------------------------------------------------------------

    def rebalance(self, every: int = 5) -> Study:
        """Forward-fill positions on a fixed rebalance schedule.

        Positions are only updated on rebalance dates (every N trading days);
        in between, yesterday's weights are carried forward unchanged.

        Args:
            every: Rebalance every N trading days (1 = daily, 5 = weekly, 21 = monthly).
        """

        def fn(positions, **cache):
            return rebalance(positions, every=every)

        fn.__name__ = f"rebalance(every={every})"
        self._steps.append(("position_scaler", "rebalance", fn))
        return self

    def rebalance_on(self, trigger_fn: Callable) -> Study:
        """Threshold-triggered rebalance: only adopt new positions when trigger fires.

        On each date, ``trigger_fn(current_positions, proposed_positions)`` is called.
        If it returns ``True``, the new portfolio is adopted; otherwise yesterday's
        weights are carried forward. The trigger always fires on the first date.

        Use the built-in trigger factories from ``qstudy``:

        .. code-block:: python

            import qstudy as qs

            # Rebalance only when rank ordering shifts significantly
            study.rebalance_on(qs.rank_change_trigger(threshold=0.7))

            # Rebalance only when the signal is extreme cross-sectionally
            study.rebalance_on(qs.signal_zscore_trigger(signal_df, threshold=1.5))

        Or pass any callable with signature ``(current: pd.Series, proposed: pd.Series) -> bool``.

        Args:
            trigger_fn: ``(current, proposed) -> bool`` — return True to rebalance.
        """

        def fn(positions, **cache):
            signal = cache.get("signal")
            return rebalance_on(positions, trigger_fn=trigger_fn, signal=signal)

        fn.__name__ = f"rebalance_on({getattr(trigger_fn, '__name__', 'custom')})"
        self._steps.append(("position_scaler", "rebalance", fn))
        return self

    def neutralize_positions(
        self,
        constraints: dict[str, float | tuple[float, float]],
    ) -> Study:
        """Enforce portfolio-level factor exposure constraints after position building.

        For each date, adjusts active positions so that the weighted sum of each factor
        exposure meets the specified constraint target. This prevents ranking/rebalancing
        from reintroducing factor bets even when the signal is already factor-neutral.

        Requires :meth:`add_factor_model` to have been called.

        Args:
            constraints: Dict of factor name -> target or tolerance band.
                         Examples:
                           ``{"beta": 0}``             — zero net beta exposure
                           ``{"sector": 0}``           — zero net sector exposure
                           ``{"momentum": (-0.05, 0.05)}`` — tolerance band
        """
        targets = {}
        for fname, val in constraints.items():
            if isinstance(val, (int, float)):
                targets[fname] = (float(val), float(val))
            else:
                targets[fname] = (float(val[0]), float(val[1]))

        def scaler(positions, **cache):
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

                    # Only adjust if outside the tolerance band
                    target_val = (lo + hi) / 2.0
                    if lo <= net_exp <= hi:
                        continue

                    # Neutralize by subtracting the projection onto the exposure vector
                    exp_norm = float((exp**2).sum())
                    if exp_norm == 0.0:
                        continue
                    adj = (net_exp - target_val) / exp_norm
                    w_valid = w_valid - adj * exp
                    w = w.copy()
                    w.loc[exp.index] = w_valid.values

                # Renormalize to preserve dollar-neutrality
                gross = w.abs().sum()
                if gross > 0:
                    w = w / gross
                adjusted.loc[date, w.index] = w.values

            return adjusted.fillna(0.0)

        scaler.__name__ = f"neutralize_positions({list(constraints.keys())})"
        self._steps.append(("position_scaler", "neutralize", scaler))
        return self

    def scale_risk(self, fn: Callable | None = None, vol_target: float | None = None) -> Study:
        """Scale position size for risk or exposure control.

        Either pass a custom scaler function (same contract as the old ``scale_returns()``)
        or specify a ``vol_target`` to automatically scale positions to a target annualized
        portfolio volatility.

        Args:
            fn:         ``fn(positions, **cache) -> pd.DataFrame``. Receives the current
                        positions and a shallow copy of the cache. Mutually exclusive with
                        ``vol_target``.
            vol_target: Target annualized portfolio volatility (e.g. ``0.10`` for 10%).
                        Uses a 63-day rolling estimate of realized vol to scale.
        """
        if fn is not None and vol_target is not None:
            raise ValueError("Provide either fn or vol_target, not both.")
        if fn is None and vol_target is None:
            raise ValueError("Provide either fn or vol_target.")

        if fn is not None:
            self._steps.append(("position_scaler", "scale_risk", fn))
            return self

        # vol_target path
        target = float(vol_target)

        def vol_scaler(positions, **cache):
            port_ret = (positions.shift(1) * cache["returns"]).sum(axis=1)
            realized_vol = port_ret.rolling(63).std() * (252**0.5)
            scale = (target / realized_vol.replace(0.0, float("nan"))).clip(upper=2.0).shift(1)
            return positions.mul(scale.fillna(1.0), axis=0)

        vol_scaler.__name__ = f"scale_risk(vol_target={vol_target})"
        self._steps.append(("position_scaler", "scale_risk", vol_scaler))
        return self

    def scale_returns(self, fn: Callable) -> Study:
        """Deprecated alias for :meth:`scale_risk`.

        Args:
            fn: ``fn(positions, **cache) -> pd.DataFrame``.
        """
        warnings.warn(
            "scale_returns() is deprecated. Use .scale_risk(fn) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scale_risk(fn=fn)

    # ------------------------------------------------------------------
    # Weighting schemes
    # ------------------------------------------------------------------

    def weight_equal(self) -> Study:
        """Equal dollar weights (default — no-op, positions are already normalized).

        This is the default behavior. Weighting schemes for combining multiple
        strategies (equal-vol, equal-Sharpe, mean-variance optimal) are available
        on :class:`~qstudy.study.PortfolioStudy`.
        """
        self._steps.append(("position_scaler", "weight", apply_equal))
        return self

    def fully_invest(self) -> Study:
        """Rescale positions so abs(weights).sum(axis=1) == 1.0 on each date.

        Useful after a custom position builder or scaler that may leave the book
        under- or over-invested. No-op on rows where the abs sum is already zero.
        """
        from qstudy.study.portfolio import normalize_weights

        def _fully_invest(positions, **cache):
            return normalize_weights(positions)

        _fully_invest.__name__ = "fully_invest"
        self._steps.append(("position_scaler", "scale_risk", _fully_invest))
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
        self._cache["_cost_bps_config"] = self._cost_bps

        stages = self._build_stage_list()
        with tqdm(total=len(stages), desc=self._name or "Study.run", disable=not self._verbose) as pbar:
            # Stage: fit factor model (optional, must run before residualize)
            if self._factor_model is not None:
                pbar.set_postfix({"stage": "factor_model"})
                self._factor_model.fit(
                    returns=self._cache["returns"],
                    benchmark_returns=self._cache["benchmark"],
                    close=self._cache["close"],
                )
                self._cache["factor_model"] = self._factor_model
                self._cache["factor_exposures"] = self._factor_model.factor_exposures_
                pbar.update(1)

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

            # Pipeline steps — signal generation, transforms, filters, position building, scalers
            for step_type, _minor, fn in self._steps:
                pbar.set_postfix({"stage": step_type})
                self._execute_step(step_type, fn)
                pbar.update(1)

            # Backtest engine
            pbar.set_postfix({"stage": "backtest"})
            returns_for_engine = self._cache["returns"]
            tradeable_mask = self._cache.get("_tradeable_mask")
            liq_mask = self._cache.get("_liquidity_mask")  # legacy compat
            combined_mask = tradeable_mask if tradeable_mask is not None else liq_mask
            if combined_mask is not None:
                returns_for_engine = returns_for_engine.where(combined_mask)
            gross_returns = engine.run(self._cache["positions"], returns_for_engine)
            self._cache["gross_portfolio_returns"] = gross_returns
            self._cache["portfolio_returns"] = gross_returns
            pbar.update(1)

            # Transaction costs (optional)
            if self._cost_bps > 0.0:
                pbar.set_postfix({"stage": "transaction_costs"})
                cost_per_dollar = self._cost_bps / 10_000.0
                cost_series = (
                    metrics.turnover(self._cache["positions"])
                    .reindex(gross_returns.index)
                    .fillna(0.0)
                    * cost_per_dollar
                )
                self._cache["portfolio_returns"] = gross_returns - cost_series
                pbar.update(1)

            # Metrics
            pbar.set_postfix({"stage": "metrics"})
            positions_for_metrics = (
                self._cache["positions"]
                if self._cost_bps > 0.0
                else self._cache.get("_unscaled_positions", self._cache["positions"])
            )
            self._cache["metrics_summary"] = metrics.summary(
                self._cache["portfolio_returns"],
                positions=positions_for_metrics,
                benchmark=self._cache["benchmark"],
                gross_returns=(
                    self._cache["gross_portfolio_returns"] if self._cost_bps > 0.0 else None
                ),
                cost_bps=self._cost_bps,
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

    def audit(self) -> pd.DataFrame:
        """Return a summary table of all pipeline steps and their intermediate state.

        Each row is one pipeline step in execution order. Call after :meth:`run` to
        inspect how candidates and positions changed at each stage — useful for
        comparing against a manual implementation or diagnosing unexpected behavior.

        Signal rows show how the eligible candidate set shrinks through filters.
        Position rows show whether weights stay dollar-neutral through scalers.

        Returns:
            DataFrame with columns:
                ``step``              — step name
                ``stage``             — ``"signal"`` or ``"position"``
                ``eligible_tickers``  — (signal) tickers with any non-NaN value
                ``total_notna``       — (signal) total non-NaN cell count
                ``nonzero_tickers``   — (position) tickers with any nonzero weight
                ``abs_sum_mean``      — (position) mean of abs(w).sum(axis=1), target ~1.0
                ``net_sum_mean``      — (position) mean of w.sum(axis=1), target ~0 for L/S
        """
        self._require_run("audit")
        rows = []
        for entry in self._cache["_signal_history"]:
            rows.append(
                {
                    "step": entry["step"],
                    "stage": "signal",
                    "eligible_tickers": entry["eligible_tickers"],
                    "total_notna": entry["total_notna"],
                    "nonzero_tickers": None,
                    "abs_sum_mean": None,
                    "net_sum_mean": None,
                }
            )
        for entry in self._cache["_position_history"]:
            rows.append(
                {
                    "step": entry["step"],
                    "stage": "position",
                    "eligible_tickers": None,
                    "total_notna": None,
                    "nonzero_tickers": entry["nonzero_tickers"],
                    "abs_sum_mean": round(entry["abs_sum_mean"], 4),
                    "net_sum_mean": round(entry["net_sum_mean"], 4),
                }
            )
        return pd.DataFrame(rows)

    def metrics_dict(self) -> dict:
        """Return the metrics summary as a plain dictionary.

        Requires :meth:`run` to have been called first.
        """
        self._require_run("metrics_dict")
        return self._cache["metrics_summary"].to_dict()

    def metrics_json(self) -> str:
        """Return the metrics summary as a JSON string.

        Requires :meth:`run` to have been called first.
        """
        import json

        self._require_run("metrics_json")
        return json.dumps(self._cache["metrics_summary"].to_dict(), default=str)

    @property
    def metrics(self) -> StudyMetrics:
        """Performance metrics as a typed dataclass.

        Requires :meth:`run` to have been called first.

        Example::

            study.metrics.sharpe_ratio
            study.metrics.max_drawdown
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
        obj._factor_model = None
        obj._tradeable_constraint_fns = []
        obj._cost_bps = float(cache.get("_cost_bps_config", 0.0))
        obj._data_injected = True
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
        self._steps.insert(0, ("base_signal", "", fn))

    def _append_signal_filter(self, fn: Callable, label: str) -> None:  # noqa: ARG002
        last_sig_idx = max(
            (i for i, s in enumerate(self._steps) if s[0] in ("base_signal", "signal_filter")),
            default=-1,
        )
        self._steps.insert(last_sig_idx + 1, ("signal_filter", "", fn))

    def _set_position_builder(self, fn: Callable, label: str) -> None:
        existing = [i for i, s in enumerate(self._steps) if s[0] == "position_builder"]
        if existing:
            warnings.warn(f"Replacing existing position builder with '{label}'.", stacklevel=3)
            for i in reversed(existing):
                self._steps.pop(i)
        self._steps.append(("position_builder", "", fn))

    def _validate_pipeline(self) -> None:
        if not self._data_injected:
            raise RuntimeError(
                "No data loaded. Either pass universe= to Study() or run via PortfolioStudy."
            )
        if self._residualize and (
            self._cache["factor_returns"] is None and self._cache["benchmark"] is None
        ):
            raise ValueError(
                "residualize_returns() requires either factors= or benchmark= "
                "to be set (pass them to Study() or use PortfolioStudy with a benchmark)."
            )
        if self._factor_model is not None and self._cache["benchmark"] is None:
            raise ValueError(
                "add_factor_model() requires benchmark= to be set "
                "(pass it to Study() or use PortfolioStudy with a benchmark)."
            )
        types = [s[0] for s in self._steps]
        if "base_signal" not in types:
            raise RuntimeError("No signal source defined. Call .base_signal(fn) before .run().")
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
        # Canonical position-scaler order: weight → neutralize → scale_risk → rebalance
        # neutralize runs before scale_risk so factor constraints are applied to raw weights;
        # scale_risk then applies a scalar vol multiplier without disturbing neutrality.
        _SCALER_PRIORITY = {"weight": 1, "neutralize": 2, "scale_risk": 3, "rebalance": 4}
        last_priority = 0
        for major_type, minor_type, fn in self._steps:
            if major_type == "position_scaler":
                p = _SCALER_PRIORITY.get(minor_type, 0)
                if p < last_priority:
                    raise ValueError(
                        f"'{fn.__name__}' (type '{minor_type}') declared after a "
                        f"higher-priority step (priority {last_priority}). "
                        f"Canonical order: weight → neutralize → scale_risk → rebalance."
                    )
                last_priority = p

    def _build_stage_list(self) -> list[str]:
        stages: list[str] = []
        if self._factor_model is not None:
            stages.append("factor_model")
        if self._residualize:
            stages.append("residualize")
        stages.append("setup")
        for step_type, _, _ in self._steps:
            stages.append(step_type)
        stages.append("backtest")
        if self._cost_bps > 0.0:
            stages.append("transaction_costs")
        stages.append("metrics")
        return stages

    def _run_residualize(self) -> None:
        returns = self._cache["returns"]
        # Use factor model (cross-sectional) if available
        if self._factor_model is not None and self._cache.get("factor_model") is not None:
            print("Using barra-lite factor model for residuals...")
            residuals, daily_r2 = self._factor_model.residualize(returns)
            self._cache["residual_returns"] = residuals
            self._cache["_xs_daily_r2"] = daily_r2
            return
        # Fall back to ETF time-series OLS
        if self._cache["factor_returns"] is not None:
            print("Using factors for residuals...")
            factor_returns = self._cache["factor_returns"]
        elif self._cache["benchmark"] is not None:
            print("Using benchmark for residuals...")
            factor_returns = self._cache["benchmark"].to_frame()
        else:
            raise ValueError(
                "residualize_returns() requires factors= or benchmark= in the Study constructor."
            )
        residuals, _, _ = residualize(returns, factor_returns)
        self._cache["residual_returns"] = residuals

    @staticmethod
    def _signal_entry(step: str, df: pd.DataFrame) -> dict:
        sig_copy = df.copy()
        return {
            "step": step,
            "df": sig_copy,
            "eligible_tickers": int(sig_copy.notna().any(axis=0).sum()),
            "total_notna": int(sig_copy.notna().sum().sum()),
        }

    @staticmethod
    def _position_entry(step: str, df: pd.DataFrame) -> dict:
        pos_copy = df.copy()
        abs_sum = pos_copy.abs().sum(axis=1)
        net_sum = pos_copy.sum(axis=1)
        return {
            "step": step,
            "df": pos_copy,
            "nonzero_tickers": int((pos_copy != 0).any(axis=0).sum()),
            "abs_sum_mean": float(abs_sum[abs_sum > 0].mean()) if (abs_sum > 0).any() else 0.0,
            "net_sum_mean": float(net_sum[abs_sum > 0].mean()) if (abs_sum > 0).any() else 0.0,
        }

    def _execute_step(self, step_type: str, fn: Callable) -> None:
        snapshot = self._cache.copy()
        returns_index = self._cache["returns"].index
        expected_shape = self._cache["signal"].shape if self._cache["signal"] is not None else None

        if step_type == "base_signal":
            sig = fn(**snapshot)
            self._cache["base_signal"] = sig
            self._cache["signal"] = sig.copy()
            self._cache["_signal_history"].append(self._signal_entry("base_signal", sig))

        elif step_type == "signal_filter":
            # Remove "signal" from snapshot to avoid collision with the positional arg
            cache_kw = {k: v for k, v in snapshot.items() if k != "signal"}
            sig = fn(self._cache["signal"], **cache_kw)
            assert sig.shape == expected_shape, (
                f"{fn.__name__} returned shape {sig.shape}, expected {expected_shape}"
            )
            assert sig.index.equals(returns_index), f"{fn.__name__} returned misaligned index"
            self._cache["signal"] = sig
            self._cache["_signal_history"].append(self._signal_entry(fn.__name__, sig))

        elif step_type == "position_builder":
            pos = fn(self._cache["signal"])
            assert pos.index.equals(returns_index), "position_builder returned misaligned index"
            self._cache["positions"] = pos
            self._cache["_unscaled_positions"] = pos.copy()
            self._cache["_position_history"].append(self._position_entry("position_builder", pos))

        elif step_type == "position_scaler":
            # Remove "positions" from snapshot to avoid collision with the positional arg
            cache_kw = {k: v for k, v in snapshot.items() if k != "positions"}
            pos = fn(self._cache["positions"], **cache_kw)
            self._cache["positions"] = pos
            self._cache["_position_history"].append(self._position_entry(fn.__name__, pos))

    def _require_run(self, method: str) -> None:
        if self._cache.get("portfolio_returns") is None:
            raise RuntimeError(f"Call .run() before .{method}().")

    def __repr__(self) -> str:
        steps = [s[0] for s in self._steps]
        ran = self._cache.get("portfolio_returns") is not None
        weighting = any(s[1] == "weight" for s in self._steps)
        return f"Study(name={self._name!r}, steps={steps}, weighting={weighting}, ran={ran})"
