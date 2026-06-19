"""
Shared helpers for the top-5-by-signal-family portfolio scripts.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import pandas as pd
from sig_fam_utils import SleeveSpec, build_top5_by_sig_fam_sleeve_specs  # noqa: F401

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics
from qstudy import Study

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTORS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]
GICS_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Communication Services": "XLK",
}
N_LONG = N_SHORT = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(start: str, end: str):
    """Return (universe, benchmark, factors) StudyData objects."""
    universe = qs.download(index_code="SP500", start=start, end=end)
    benchmark = qs.download(["SPY"], start=start, end=end)
    factors = qs.download(FACTORS, start=start, end=end)
    return universe, benchmark, factors


# ---------------------------------------------------------------------------
# Distance partners
# ---------------------------------------------------------------------------


def compute_distance_partners(
    universe,
    train_end: str,
    zw_list: list[int] | None = None,
) -> dict[int, dict[str, list[str]]]:
    """Compute top-3 nearest distance partners by normalised price correlation.

    Parameters
    ----------
    universe:
        StudyData object — ``universe.log_returns`` is used.
    train_end:
        Inclusive end date for the training slice (e.g. ``"2020-12-31"``).
    zw_list:
        Z-windows to compute partners for. Defaults to ``[10, 20, 60]``.
    """
    if zw_list is None:
        zw_list = [10, 20, 60]
    log_price = universe.log_returns.loc[:train_end].cumsum()
    norm = (log_price - log_price.mean()) / log_price.std().clip(lower=1e-8)
    dist = 1 - norm.corr()
    partners: dict[int, dict[str, list[str]]] = {}
    for zw in zw_list:
        p: dict[str, list[str]] = {}
        for ticker in dist.columns:
            p[ticker] = dist[ticker].drop(ticker).nsmallest(3).index.tolist()
        partners[zw] = p
    return partners


# ---------------------------------------------------------------------------
# Sector ETF map
# ---------------------------------------------------------------------------


def get_sector_etf_map_for(universe) -> dict[str, str]:
    """Return a ticker -> sector ETF mapping for all tickers in universe."""
    sector_map = qs.get_sector_map(list(universe.returns.columns))
    return {t: GICS_TO_ETF.get(s, "SPY") for t, s in sector_map.items()}


# ---------------------------------------------------------------------------
# Risk scalers
# ---------------------------------------------------------------------------


def equity_curve_regime_scale(positions: pd.DataFrame, **cache) -> pd.DataFrame:
    """Scale to 25% exposure when sleeve equity curve is below its 20-day MA."""
    returns = cache["returns"]
    _tm = cache.get("_tradeable_mask")
    _lm = cache.get("_liquidity_mask")
    mask = _tm if _tm is not None else _lm
    r = returns.where(mask) if mask is not None else returns
    raw = (positions.shift(1) * r).sum(axis=1)
    equity = (1 + raw).cumprod()
    scale = pd.Series(
        np.where(equity > equity.rolling(20).mean(), 1.0, 0.1),
        index=equity.index,
    )
    return positions.mul(scale.shift(1), axis=0)


equity_curve_regime_scale.__name__ = "equity_curve_regime_scale"


# ---------------------------------------------------------------------------
# Sleeve runner
# ---------------------------------------------------------------------------


def run_sleeve_for_spec(
    spec: SleeveSpec,
    universe,
    benchmark,
    factors,
    sector_map: dict[str, str],
) -> Study:
    """Build and run a single sleeve defined by *spec*."""
    builder = Study(
        universe=universe,
        benchmark=benchmark,
        factors=factors,
        verbose=False,
    )
    if spec.use_factor_model:
        builder = builder.add_factor_model(
            factors=["market", "sector"],
            sector_map=sector_map,
        )
    if spec.use_factor_model or spec.use_etf_resid or spec.needs_resid_cache:
        builder = builder.residualize_returns()

    builder = builder.base_signal(spec.signal_fn)

    if spec.conditioning_filter is not None:
        builder = builder.add_filter(spec.conditioning_filter)

    builder = (
        builder.add_tradeable_constraint(qs.liquidity(top_n=300))
        .rank_transform()
        .build_long_short(n_long=N_LONG, n_short=N_SHORT)
        .fully_invest()
        .scale_risk(fn=equity_curve_regime_scale)
    )

    for scaler_fn in spec.risk_scalers:
        builder = builder.scale_risk(fn=scaler_fn)

    return builder.rebalance(every=spec.rebalance_every).run()


def run_sleeve_pool(
    specs: dict[str, SleeveSpec],
    universe,
    benchmark,
    factors,
    sector_map: dict[str, str],
    verbose: bool = True,
) -> dict[str, Study]:
    """Run all specs and return a dict of name -> Study."""
    studies: dict[str, Study] = {}
    for name, spec in specs.items():
        if verbose:
            print(f"  Running {name} ...", end=" ", flush=True)
        studies[name] = run_sleeve_for_spec(spec, universe, benchmark, factors, sector_map)
        if verbose:
            m = studies[name].metrics_dict()
            print(f"sharpe={m.get('sharpe', float('nan')):.3f}")
    return studies


# ---------------------------------------------------------------------------
# Position combining
# ---------------------------------------------------------------------------


def combine_positions_fixed_weights(
    studies: dict[str, Study],
    weights: dict[str, float],
    names: list[str],
) -> pd.DataFrame:
    """Combine sleeve positions with fixed weights, renormalised to gross=1 each day."""
    all_cols: set[str] = set()
    all_idx: set = set()
    for n in names:
        pos = studies[n].cache["positions"]
        all_cols |= set(pos.columns)
        all_idx |= set(pos.index)

    sorted_cols = sorted(all_cols)
    sorted_idx = sorted(all_idx)

    combined = pd.DataFrame(0.0, index=sorted_idx, columns=sorted_cols)
    for n in names:
        pos = (
            studies[n]
            .cache["positions"]
            .reindex(index=sorted_idx, columns=sorted_cols, fill_value=0.0)
            .fillna(0.0)
        )
        combined += weights[n] * pos

    gross = combined.abs().sum(axis=1).replace(0, np.nan)
    return combined.div(gross, axis=0)


# ---------------------------------------------------------------------------
# Portfolio evaluation
# ---------------------------------------------------------------------------


def evaluate_fixed_weight_portfolio(
    combined_positions: pd.DataFrame,
    universe,
    benchmark,
    cost_bps: float,
) -> pd.Series:
    """Evaluate a combined position DataFrame using StudyData objects."""
    universe_returns = universe.returns.reindex(columns=combined_positions.columns).fillna(0)
    bm = benchmark.returns["SPY"].reindex(universe_returns.index).fillna(0)
    return evaluate_fixed_weight_portfolio_raw(combined_positions, universe_returns, bm, cost_bps)


def evaluate_fixed_weight_portfolio_raw(
    combined_positions: pd.DataFrame,
    universe_returns: pd.DataFrame,
    benchmark_series: pd.Series,
    cost_bps: float,
) -> pd.Series:
    """Evaluate a combined position DataFrame using raw DataFrames."""
    universe_returns = universe_returns.reindex(columns=combined_positions.columns).fillna(0)
    gross_returns = qs_engine.run(combined_positions, universe_returns)
    to = qs_metrics.turnover(combined_positions)
    cost = to * cost_bps / 10_000
    net_returns = gross_returns - cost
    bm = benchmark_series.reindex(gross_returns.index).fillna(0)
    return qs_metrics.summary(
        net_returns,
        positions=combined_positions,
        benchmark=bm,
        gross_returns=gross_returns,
        cost_bps=cost_bps,
    )


# ---------------------------------------------------------------------------
# Weight estimators
# ---------------------------------------------------------------------------


def estimate_weights_equal(names: list[str]) -> dict[str, float]:
    n = len(names)
    return {name: 1.0 / n for name in names}


def estimate_weights_equal_vol(
    names: list[str],
    sleeve_returns_train: pd.DataFrame,
) -> dict[str, float]:
    vols = sleeve_returns_train[names].std()
    valid = vols[(vols > 0) & vols.notna()]
    if valid.empty:
        return estimate_weights_equal(names)
    inv_vol = 1.0 / valid
    total = inv_vol.sum()
    weights = {n: float(inv_vol.get(n, 0.0) / total) for n in names}
    total_w = sum(weights.values())
    if total_w <= 0:
        return estimate_weights_equal(names)
    return {n: w / total_w for n, w in weights.items()}


def estimate_weights_equal_sharpe(
    names: list[str],
    sleeve_returns_train: pd.DataFrame,
) -> dict[str, float]:
    sharpes = sleeve_returns_train[names].apply(
        lambda s: s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan
    )
    pos_sharpes = sharpes[(sharpes > 0) & sharpes.notna()]
    if pos_sharpes.empty:
        return estimate_weights_equal_vol(names, sleeve_returns_train)
    total = pos_sharpes.sum()
    weights = {n: float(pos_sharpes.get(n, 0.0) / total) for n in names}
    total_w = sum(weights.values())
    if total_w <= 0:
        return estimate_weights_equal_vol(names, sleeve_returns_train)
    return {n: w / total_w for n, w in weights.items()}


def estimate_weights_optimal(
    names: list[str],
    sleeve_returns_train: pd.DataFrame,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Ridge-regularized mean-variance optimal weights, fitted on training returns.

    Matches the approach used by ``PortfolioStudy.weight_optimal``.
    Falls back to ``equal_vol`` if the covariance matrix is singular or
    there are fewer than 10 observations.
    """
    r = sleeve_returns_train[names].dropna()
    if len(r) < 10:
        return estimate_weights_equal_vol(names, sleeve_returns_train)
    mu = r.mean().values
    sigma = r.cov().values
    try:
        ridge = gamma * np.diag(sigma).mean() * np.eye(len(sigma))
        w = np.linalg.solve(sigma + ridge, mu)
        abs_sum = np.abs(w).sum()
        if abs_sum < 1e-12:
            return estimate_weights_equal_vol(names, sleeve_returns_train)
        w = w / abs_sum
        return {n: float(w[i]) for i, n in enumerate(names)}
    except np.linalg.LinAlgError:
        return estimate_weights_equal_vol(names, sleeve_returns_train)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def get_net_sharpe(m: pd.Series) -> float:
    return float(m.get("net_sharpe", m.get("sharpe", float("nan"))))


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

_WEIGHTING_SCHEMES = ["equal", "equal_vol", "equal_sharpe"]


def run_greedy_selection(
    spec_names: list[str],
    studies: dict[str, Study],
    sleeve_returns: pd.DataFrame,
    corr: pd.DataFrame,
    universe,
    benchmark,
    cost_bps: float,
    weighting_schemes: list[str] | None = None,
    max_pairwise_abs_corr: float = 0.75,
    min_net_sharpe_delta: float = 0.05,
    max_dd_regression: float = 0.01,
    max_turnover_regression: float = 0.001,
    min_dd_improvement: float = 0.0025,
    min_turnover_improvement: float = 0.00025,
    min_sleeves: int = 1,
    evaluate_fn=None,
    random_selection: bool = False,
    random_seed: int | None = None,
    seed_sleeve: str | None = None,
) -> tuple[list[str], str, dict[str, float], list[dict], list[dict]]:
    """Greedy forward-stepwise sleeve selector.

    Parameters
    ----------
    spec_names:
        Ordered list of all sleeve names in the pool.
    studies:
        Dict of name -> Study (already run).
    sleeve_returns:
        DataFrame of daily sleeve returns (columns = sleeve names).
    corr:
        Pairwise correlation matrix (computed from *sleeve_returns*).
    universe, benchmark:
        StudyData objects for evaluation. Pass ``None`` if using *evaluate_fn*.
    cost_bps:
        Transaction cost in basis points.
    weighting_schemes:
        List of scheme names to try. Default: ``["equal", "equal_vol", "equal_sharpe"]``.
    min_sleeves:
        Minimum number of sleeves to include. Until this count is reached the
        improvement gates (sharpe delta, dd regression, turnover regression,
        material improvement) are bypassed — only the correlation cap applies.
        The best-scoring candidate (by net_sharpe) that passes the corr cap is
        always accepted until ``min_sleeves`` is reached.
    evaluate_fn:
        Optional callable ``(combined_positions) -> pd.Series``. If ``None``,
        ``evaluate_fixed_weight_portfolio`` is called with *universe* and *benchmark*.
    random_selection:
        If ``True``, when multiple candidates pass all acceptance criteria at a
        given step, one is chosen uniformly at random rather than always picking
        the highest-net-Sharpe candidate. Useful for sensitivity analysis.
    random_seed:
        Seed for the random number generator used when ``random_selection=True``.
        ``None`` means non-deterministic.
    seed_sleeve:
        If set, this sleeve is always chosen as the first sleeve regardless of
        standalone Sharpe. The greedy search then builds from it. ``None`` means
        the highest standalone-Sharpe sleeve is used as the seed (default).

    Returns
    -------
    selected, final_scheme, final_weights, buildout_rows, candidate_rows
    """
    if weighting_schemes is None:
        weighting_schemes = _WEIGHTING_SCHEMES

    rng = random.Random(random_seed) if random_selection else None

    if evaluate_fn is None:

        def evaluate_fn(combined_positions):  # type: ignore[misc]
            return evaluate_fixed_weight_portfolio(
                combined_positions, universe, benchmark, cost_bps
            )

    # --- seed: use provided seed_sleeve or fall back to highest standalone sharpe ---
    standalone_sharpe = {n: get_net_sharpe(studies[n].metrics_dict()) for n in spec_names}
    if seed_sleeve is not None:
        if seed_sleeve not in standalone_sharpe:
            raise ValueError(f"seed_sleeve {seed_sleeve!r} not found in spec_names")
        seed = seed_sleeve
    else:
        seed = max(standalone_sharpe, key=standalone_sharpe.get)  # type: ignore[arg-type]

    seed_weights = estimate_weights_equal([seed])
    seed_positions = combine_positions_fixed_weights(studies, seed_weights, [seed])
    current_metrics = evaluate_fn(seed_positions)
    current_scheme = "equal"
    current_weights = seed_weights

    buildout_rows: list[dict] = [
        {
            "step": 0,
            "sleeve_added": seed,
            "scheme": "equal",
            "sleeve_count": 1,
            "net_sharpe": get_net_sharpe(current_metrics),
            "max_drawdown": current_metrics.get("max_drawdown"),
            "avg_daily_turnover": current_metrics.get("avg_daily_turnover"),
            "max_pairwise_corr": float("nan"),
        }
    ]
    candidate_rows: list[dict] = []
    selected = [seed]

    for _step in range(1, len(spec_names)):
        remaining = [n for n in spec_names if n not in selected]
        if not remaining:
            break

        step_candidates: list[dict] = []

        for sleeve in remaining:
            # --- correlation gate ---
            max_corr = float(corr.loc[sleeve, selected].abs().max())
            if max_corr > max_pairwise_abs_corr:
                candidate_rows.append(
                    {
                        "step": len(selected),
                        "sleeve": sleeve,
                        "scheme": None,
                        "max_corr": max_corr,
                        "net_sharpe": float("nan"),
                        "max_drawdown": float("nan"),
                        "avg_daily_turnover": float("nan"),
                        "sharpe_delta": float("nan"),
                        "dd_delta": float("nan"),
                        "to_delta": float("nan"),
                        "reject_reason": "corr_cap",
                    }
                )
                continue

            # --- try each weighting scheme ---
            best: dict | None = None
            for scheme in weighting_schemes:
                candidate_set = selected + [sleeve]
                if scheme == "equal":
                    weights = estimate_weights_equal(candidate_set)
                elif scheme == "equal_vol":
                    weights = estimate_weights_equal_vol(candidate_set, sleeve_returns)
                elif scheme == "equal_sharpe":
                    weights = estimate_weights_equal_sharpe(candidate_set, sleeve_returns)
                else:
                    weights = estimate_weights_optimal(candidate_set, sleeve_returns)

                combined_pos = combine_positions_fixed_weights(studies, weights, candidate_set)
                m = evaluate_fn(combined_pos)

                if best is None or get_net_sharpe(m) > get_net_sharpe(best["metrics"]):
                    best = {
                        "scheme": scheme,
                        "weights": weights,
                        "metrics": m,
                        "positions": combined_pos,
                    }

            assert best is not None

            cur_ns = get_net_sharpe(current_metrics)
            new_ns = get_net_sharpe(best["metrics"])
            cur_dd = float(current_metrics.get("max_drawdown", float("nan")))
            new_dd = float(best["metrics"].get("max_drawdown", float("nan")))
            cur_to = float(current_metrics.get("avg_daily_turnover", float("nan")))
            new_to = float(best["metrics"].get("avg_daily_turnover", float("nan")))

            sharpe_delta = new_ns - cur_ns
            dd_delta = new_dd - cur_dd  # positive = improved (less negative)
            to_delta = new_to - cur_to  # positive = worsened

            reject_reason: str | None = None
            below_min = len(selected) < min_sleeves
            if not below_min:
                if sharpe_delta < min_net_sharpe_delta:
                    reject_reason = "sharpe_delta"
                elif dd_delta < -max_dd_regression:
                    reject_reason = "dd_regression"
                elif to_delta > max_turnover_regression:
                    reject_reason = "turnover_regression"
                elif dd_delta < min_dd_improvement and -to_delta < min_turnover_improvement:
                    reject_reason = "no_material_profile_improvement"

            candidate_rows.append(
                {
                    "step": len(selected),
                    "sleeve": sleeve,
                    "scheme": best["scheme"],
                    "max_corr": max_corr,
                    "net_sharpe": new_ns,
                    "max_drawdown": new_dd,
                    "avg_daily_turnover": new_to,
                    "sharpe_delta": sharpe_delta,
                    "dd_delta": dd_delta,
                    "to_delta": to_delta,
                    "reject_reason": reject_reason or "admitted",
                }
            )

            if reject_reason is None:
                step_candidates.append(
                    {
                        "sleeve": sleeve,
                        "scheme": best["scheme"],
                        "weights": best["weights"],
                        "metrics": best["metrics"],
                        "max_corr": max_corr,
                    }
                )

        if not step_candidates:
            break

        # Sort: best net_sharpe desc, then lowest max_corr,
        # then highest drawdown (least negative), then lowest turnover
        step_candidates.sort(
            key=lambda c: (
                -get_net_sharpe(c["metrics"]),
                c["max_corr"],
                -float(c["metrics"].get("max_drawdown", float("-inf"))),
                float(c["metrics"].get("avg_daily_turnover", float("inf"))),
            )
        )

        if random_selection and rng is not None and len(step_candidates) > 1:
            winner = rng.choice(step_candidates)
        else:
            winner = step_candidates[0]
        selected.append(winner["sleeve"])
        current_metrics = winner["metrics"]
        current_scheme = winner["scheme"]
        current_weights = winner["weights"]

        buildout_rows.append(
            {
                "step": len(selected) - 1,
                "sleeve_added": winner["sleeve"],
                "scheme": winner["scheme"],
                "sleeve_count": len(selected),
                "net_sharpe": get_net_sharpe(winner["metrics"]),
                "max_drawdown": winner["metrics"].get("max_drawdown"),
                "avg_daily_turnover": winner["metrics"].get("avg_daily_turnover"),
                "max_pairwise_corr": winner["max_corr"],
            }
        )

    return selected, current_scheme, current_weights, buildout_rows, candidate_rows
