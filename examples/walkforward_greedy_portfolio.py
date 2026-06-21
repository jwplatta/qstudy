"""
Walkforward validation: expanding train windows, fixed validation years 2021-2023.

For each fold the greedy selection algorithm is run on training data; the
frozen sleeve set and weights are then evaluated on the held-out validation
year.

Outputs are written to examples/out/walkforward_top5_by_sig_fam_portfolio/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import portfolio_utils as pu
from sig_fam_utils import build_sleeve_specs

import qstudy as qs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOLDS = [
    ("2015-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
    ("2015-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]

COST_BPS = 10.0
WARMUP_YEARS = 1
WEIGHTING_SCHEMES = ["equal", "equal_vol", "equal_sharpe"]
# Optional: fix the first sleeve picked by the greedy algorithm.
# When set, this sleeve is always selected first and the greedy search
# builds the portfolio from there. Set to None to let greedy choose freely.
# "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
# "etf_factor_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
# "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",

# Top seed candidates from find_seed_sleeves.py (run to refresh):
#   #1  avg_SR=1.15  neg_yrs=11%  "bear_reversal_20d__cond__bear_narrow_lt40__r21__trend_20_100_mr"
#   #2  avg_SR=1.20  neg_yrs=11%  "vol_accel_20_120d__cond__breadth_lt40__r10__vol_10_60_up"
#   #3  avg_SR=0.95  neg_yrs= 0%  "vol_accel_10_90d__cond__bear_narrow_lt40__r10__trend_50_200_mr"
#   #4  avg_SR=0.95  neg_yrs=11%  "low_vol_mom_120d__cond__breadth_lt50__r21__vol_20_60"  (non-bear-breadth)
#   #5  avg_SR=1.05  neg_yrs=14%  "gap_accum_3d__r21__trend_20_100_off"  (event; good 2022 hedge)
# Note: ranks 1-5 are bear-narrow-breadth — use a non-bear-breadth seed for family diversification.
SEED_SLEEVE: str | None = None
MAX_PAIRWISE_ABS_CORR = 0.35
MIN_NET_SHARPE_DELTA = 0.01
MAX_DD_REGRESSION = 0.001
MAX_TURNOVER_REGRESSION = 0.001
MIN_DD_IMPROVEMENT = 0.001
MIN_TURNOVER_IMPROVEMENT = 0.0001
MIN_SLEEVES = 5

OUT_DIR = Path(__file__).parent / "out" / "walkforward_greedy_portfolio"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    all_summary_rows: list[dict] = []
    all_selection_detail: list[dict] = []
    all_buildout_rows: list[dict] = []
    all_selected_weights: list[dict] = []
    all_val_returns: list[pd.DataFrame] = []

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(FOLDS):
        fold_label = f"{val_start[:4]}"
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}: train {train_start}..{train_end}, val {val_start}..{val_end}")
        print("=" * 60)

        # --- Load combined window data ---
        warmup_start = str(int(train_start[:4]) - WARMUP_YEARS) + train_start[4:]
        print(f"  Loading data ({warmup_start} to {val_end}) ...")
        universe, benchmark, factors = pu.load_data(warmup_start, val_end)
        print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

        # --- Distance partners on training slice only ---
        print("  Computing distance partners on train slice ...")
        partners = pu.compute_distance_partners(
            universe, train_end=train_end, train_start=train_start
        )
        get_distance_partners = lambda: partners  # noqa: E731

        sector_etf_map = pu.get_sector_etf_map_for(universe)
        get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
        sector_map = qs.get_sector_map(list(universe.returns.columns))

        # --- Build specs and run all 30 sleeves on the full combined window ---
        specs = build_sleeve_specs(
            get_distance_partners=get_distance_partners,
            get_sector_etf_map=get_sector_etf_map,
        )
        print(f"  Running {len(specs)} sleeves on full window ...")
        studies = pu.run_sleeve_pool(
            specs,
            universe,
            benchmark,
            factors,
            sector_map,
            verbose=False,
            residualize_fit_start=train_start,
            scaler_start=train_start,
        )
        print("  Done running sleeves.")

        # --- Build sleeve returns; slice to training period for selection ---
        sleeve_returns_full = pd.DataFrame(
            {n: studies[n].cache["portfolio_returns"] for n in specs}
        )
        sleeve_returns_train = sleeve_returns_full.loc[:train_end]
        corr_train = sleeve_returns_train.corr()

        # --- Build training-sliced combined positions + evaluator ---
        def make_train_positions(names: list[str], weights: dict[str, float]) -> pd.DataFrame:
            # Slice each sleeve's positions to the training period
            train_studies_proxy: dict[str, object] = {}
            for n in names:
                pos_train = studies[n].cache["positions"].loc[:train_end]

                class _FakeStudy:
                    cache = {"positions": pos_train}  # type: ignore[assignment]

                train_studies_proxy[n] = _FakeStudy()  # type: ignore[assignment]
            return pu.combine_positions_fixed_weights(train_studies_proxy, weights, names)  # type: ignore[arg-type]

        universe_train_returns = universe.returns.loc[:train_end]
        bm_train_series = benchmark.returns["SPY"].loc[:train_end]

        def train_evaluate_fn(combined_positions: pd.DataFrame) -> pd.Series:
            return pu.evaluate_fixed_weight_portfolio_raw(
                combined_positions,
                universe_train_returns,
                bm_train_series,
                COST_BPS,
            )

        # Proxy studies dict that exposes training-sliced positions
        class _SliceProxy:
            def __init__(self, study, end: str, sleeve_ret_train: pd.Series) -> None:
                self._pos = study.cache["positions"].loc[:end]
                self.cache = {"positions": self._pos}
                self._sleeve_ret_train = sleeve_ret_train

            def metrics_dict(self) -> dict:  # type: ignore[override]
                s = self._sleeve_ret_train
                sharpe = float(s.mean() / s.std() * (252**0.5)) if s.std() > 0 else float("nan")
                return {"sharpe": sharpe}

        train_proxy_studies: dict[str, object] = {
            n: _SliceProxy(studies[n], train_end, sleeve_returns_train[n]) for n in specs
        }

        print("  Running greedy selection on training data ...")
        (
            selected,
            final_scheme,
            final_weights,
            buildout_rows,
            candidate_rows,
        ) = pu.run_greedy_selection(
            spec_names=list(specs.keys()),
            studies=train_proxy_studies,  # type: ignore[arg-type]
            sleeve_returns=sleeve_returns_train,
            corr=corr_train,
            universe=None,
            benchmark=None,
            cost_bps=COST_BPS,
            weighting_schemes=WEIGHTING_SCHEMES,
            max_pairwise_abs_corr=MAX_PAIRWISE_ABS_CORR,
            min_net_sharpe_delta=MIN_NET_SHARPE_DELTA,
            max_dd_regression=MAX_DD_REGRESSION,
            max_turnover_regression=MAX_TURNOVER_REGRESSION,
            min_dd_improvement=MIN_DD_IMPROVEMENT,
            min_turnover_improvement=MIN_TURNOVER_IMPROVEMENT,
            min_sleeves=MIN_SLEEVES,
            evaluate_fn=train_evaluate_fn,
            seed_sleeve=SEED_SLEEVE,
        )

        print(f"  Selected {len(selected)} sleeves: {selected}")

        # Tag all rows with fold info
        for row in buildout_rows:
            row["fold"] = fold_label
        for row in candidate_rows:
            row["fold"] = fold_label
        all_selection_detail.extend(buildout_rows)
        all_selection_detail.extend(candidate_rows)
        all_buildout_rows.extend(buildout_rows)

        # --- Evaluate on training period ---
        train_combined = make_train_positions(selected, final_weights)
        train_metrics = train_evaluate_fn(train_combined)
        train_ns = pu.get_net_sharpe(train_metrics)

        # --- Evaluate on validation period using full-window positions ---
        # Build combined positions on full window, then slice to val
        full_combined = pu.combine_positions_fixed_weights(studies, final_weights, selected)
        val_combined = full_combined.loc[val_start:val_end]

        val_universe_returns = universe.returns.loc[val_start:val_end]
        val_bm_series = benchmark.returns["SPY"].loc[val_start:val_end]

        val_metrics = pu.evaluate_fixed_weight_portfolio_raw(
            val_combined, val_universe_returns, val_bm_series, COST_BPS
        )
        val_ns = pu.get_net_sharpe(val_metrics)

        print(f"  Train net_sharpe={train_ns:.3f}, Val net_sharpe={val_ns:.3f}")

        all_summary_rows.append(
            {
                "fold": fold_label,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "n_selected": len(selected),
                "final_scheme": final_scheme,
                "train_net_sharpe": train_ns,
                "train_ann_return": train_metrics.get("ann_return"),
                "train_ann_vol": train_metrics.get("ann_vol"),
                "train_max_drawdown": train_metrics.get("max_drawdown"),
                "train_avg_daily_turnover": train_metrics.get("avg_daily_turnover"),
                "val_net_sharpe": val_ns,
                "val_ann_return": val_metrics.get("ann_return"),
                "val_ann_vol": val_metrics.get("ann_vol"),
                "val_max_drawdown": val_metrics.get("max_drawdown"),
                "val_avg_daily_turnover": val_metrics.get("avg_daily_turnover"),
                "portfolio_sleeves": "|".join(selected),
            }
        )

        for name in selected:
            all_selected_weights.append(
                {
                    "fold": fold_label,
                    "sleeve": name,
                    "weight": final_weights.get(name, 0.0),
                    "scheme": final_scheme,
                }
            )

        # --- Daily val returns ---
        import qstudy.study.engine as qs_engine
        import qstudy.study.metrics as qs_metrics

        val_universe_aligned = val_universe_returns.reindex(columns=val_combined.columns).fillna(0)
        val_gross = qs_engine.run(val_combined, val_universe_aligned)
        val_to = qs_metrics.turnover(val_combined)
        val_net = val_gross - val_to * COST_BPS / 10_000
        val_bm_aligned = val_bm_series.reindex(val_gross.index).fillna(0)

        fold_ret_df = pd.DataFrame(
            {"gross": val_gross, "net": val_net, "benchmark": val_bm_aligned}
        )
        fold_ret_df.index.name = "date"
        fold_ret_df["fold"] = fold_label
        all_val_returns.append(fold_ret_df)

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    print("\nWriting outputs ...")

    # --- Fold results summary ---
    summary_df = pd.DataFrame(all_summary_rows)
    fold_results = summary_df.rename(
        columns={
            "fold": "Fold",
            "train_net_sharpe": "Train SR (net)",
            "train_ann_return": "Train Return",
            "train_ann_vol": "Train Vol",
            "train_max_drawdown": "Train Drawdown",
            "train_avg_daily_turnover": "Train Turnover",
            "val_net_sharpe": "Validation SR (net)",
            "val_ann_return": "Validation Return",
            "val_ann_vol": "Validation Vol",
            "val_max_drawdown": "Validation Drawdown",
            "val_avg_daily_turnover": "Validation Turnover",
            "portfolio_sleeves": "Portfolio Sleeves",
        }
    )[
        [
            "Fold",
            "Train SR (net)",
            "Train Return",
            "Train Vol",
            "Train Drawdown",
            "Train Turnover",
            "Validation SR (net)",
            "Validation Return",
            "Validation Vol",
            "Validation Drawdown",
            "Validation Turnover",
            "Portfolio Sleeves",
        ]
    ]
    fold_results.to_csv(OUT_DIR / "walkforward_fold_results.csv", index=False)
    print("Saved walkforward_fold_results.csv")

    # --- Sleeve selection frequency (pivot: sleeve x fold, True if selected) ---
    weights_df = pd.DataFrame(all_selected_weights)
    all_sleeves = sorted(weights_df["sleeve"].unique())
    folds = [f"{val_start[:4]}" for _, _, val_start, _ in FOLDS]
    selected_by_fold = {
        fold: set(weights_df.loc[weights_df["fold"] == fold, "sleeve"]) for fold in folds
    }
    freq_rows = [
        {"sleeve": s, **{fold: s in selected_by_fold[fold] for fold in folds}} for s in all_sleeves
    ]
    freq_df = pd.DataFrame(freq_rows)
    freq_df["times_selected"] = freq_df[folds].sum(axis=1)
    freq_df = freq_df.sort_values("times_selected", ascending=False).reset_index(drop=True)
    freq_df.to_csv(OUT_DIR / "walkforward_sleeve_selection_frequency.csv", index=False)
    print("Saved walkforward_sleeve_selection_frequency.csv")

    # --- Selected weights ---
    weights_df.to_csv(OUT_DIR / "walkforward_selected_weights.csv", index=False)
    print("Saved walkforward_selected_weights.csv")

    # --- Validation returns ---
    val_returns_df = pd.concat(all_val_returns).reset_index()
    val_returns_df.to_csv(OUT_DIR / "walkforward_validation_returns.csv", index=False)
    print("Saved walkforward_validation_returns.csv")

    stitched = val_returns_df.drop(columns=["fold"]).set_index("date").sort_index()
    stitched.to_csv(OUT_DIR / "walkforward_validation_returns_stitched.csv")
    print("Saved walkforward_validation_returns_stitched.csv")

    # --- Selection detail (all candidate + buildout rows) ---
    detail_df = pd.DataFrame(all_selection_detail)
    detail_df.to_csv(OUT_DIR / "walkforward_selection_detail.csv", index=False)
    print("Saved walkforward_selection_detail.csv")

    # --- Chart: training net Sharpe vs sleeve count per fold ---
    buildout_df = pd.DataFrame(all_buildout_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for fold in folds:
        fold_bo = buildout_df[buildout_df["fold"] == fold].sort_values("sleeve_count")
        ax.plot(
            fold_bo["sleeve_count"],
            fold_bo["net_sharpe"],
            marker="o",
            markersize=5,
            label=f"Train {fold}",
        )
    ax.set_xlabel("Number of Sleeves")
    ax.set_ylabel("Net Sharpe (training)")
    ax.set_title("Training Net Sharpe vs Sleeves Added — per Fold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "walkforward_train_sharpe_buildout.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved walkforward_train_sharpe_buildout.png")

    print("\nWalkforward complete.")
    cols = ["Fold", "Train SR (net)", "Validation SR (net)", "Portfolio Sleeves"]
    print(fold_results[cols].to_string(index=False))


if __name__ == "__main__":
    main()
