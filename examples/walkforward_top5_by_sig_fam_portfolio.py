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

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import top5_by_sig_fam_portfolio_utils as pu
from top5_by_sig_fam_utils import build_top5_by_sig_fam_sleeve_specs

import qstudy as qs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOLDS = [
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]

COST_BPS = 10.0
WEIGHTING_SCHEMES = ["equal", "equal_vol", "equal_sharpe"]
MAX_PAIRWISE_ABS_CORR = 0.35
MIN_NET_SHARPE_DELTA = 0.01
MAX_DD_REGRESSION = 0.001
MAX_TURNOVER_REGRESSION = 0.001
MIN_DD_IMPROVEMENT = 0.001
MIN_TURNOVER_IMPROVEMENT = 0.0001
MIN_SLEEVES = 5

OUT_DIR = Path(__file__).parent / "out" / "walkforward_top5_by_sig_fam_portfolio"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    all_summary_rows: list[dict] = []
    all_selection_detail: list[dict] = []
    all_selected_weights: list[dict] = []
    all_val_returns: list[pd.DataFrame] = []

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(FOLDS):
        fold_label = f"{val_start[:4]}"
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}: train {train_start}..{train_end}, val {val_start}..{val_end}")
        print("=" * 60)

        # --- Load combined window data ---
        print(f"  Loading data ({train_start} to {val_end}) ...")
        universe, benchmark, factors = pu.load_data(train_start, val_end)
        print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

        # --- Distance partners on training slice only ---
        print("  Computing distance partners on train slice ...")
        partners = pu.compute_distance_partners(universe, train_end=train_end)
        get_distance_partners = lambda: partners  # noqa: E731

        sector_etf_map = pu.get_sector_etf_map_for(universe)
        get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
        sector_map = qs.get_sector_map(list(universe.returns.columns))

        # --- Build specs and run all 30 sleeves on the full combined window ---
        specs = build_top5_by_sig_fam_sleeve_specs(
            get_distance_partners=get_distance_partners,
            get_sector_etf_map=get_sector_etf_map,
        )
        print(f"  Running {len(specs)} sleeves on full window ...")
        studies = pu.run_sleeve_pool(specs, universe, benchmark, factors, sector_map, verbose=False)
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
        )

        print(f"  Selected {len(selected)} sleeves: {selected}")

        # Tag all rows with fold info
        for row in buildout_rows:
            row["fold"] = fold_label
        for row in candidate_rows:
            row["fold"] = fold_label
        all_selection_detail.extend(buildout_rows)
        all_selection_detail.extend(candidate_rows)

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
                "train_max_drawdown": train_metrics.get("max_drawdown"),
                "train_avg_daily_turnover": train_metrics.get("avg_daily_turnover"),
                "val_net_sharpe": val_ns,
                "val_max_drawdown": val_metrics.get("max_drawdown"),
                "val_avg_daily_turnover": val_metrics.get("avg_daily_turnover"),
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

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUT_DIR / "walkforward_summary.csv", index=False)
    print("Saved walkforward_summary.csv")

    detail_df = pd.DataFrame(all_selection_detail)
    detail_df.to_csv(OUT_DIR / "walkforward_selection_detail.csv", index=False)
    print("Saved walkforward_selection_detail.csv")

    # Selected weights: fold x sleeve with weight and scheme
    weights_df = pd.DataFrame(all_selected_weights)
    weights_df.to_csv(OUT_DIR / "walkforward_selected_weights.csv", index=False)
    print("Saved walkforward_selected_weights.csv")

    # Selected sleeves: just fold x sleeve name (no weights)
    sleeves_df = weights_df[["fold", "sleeve"]].copy()
    sleeves_df.to_csv(OUT_DIR / "walkforward_selected_sleeves.csv", index=False)
    print("Saved walkforward_selected_sleeves.csv")

    # Per-fold validation returns
    val_returns_df = pd.concat(all_val_returns).reset_index()
    val_returns_df.to_csv(OUT_DIR / "walkforward_validation_returns.csv", index=False)
    print("Saved walkforward_validation_returns.csv")

    # Stitched (drop fold column for clean time series)
    stitched = val_returns_df.drop(columns=["fold"]).set_index("date").sort_index()
    stitched.to_csv(OUT_DIR / "walkforward_validation_returns_stitched.csv")
    print("Saved walkforward_validation_returns_stitched.csv")

    print("\nWalkforward complete.")
    cols = ["fold", "n_selected", "final_scheme", "train_net_sharpe", "val_net_sharpe"]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
