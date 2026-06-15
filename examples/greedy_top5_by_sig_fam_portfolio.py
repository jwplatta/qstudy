"""
Greedy forward-stepwise sleeve selector over the 30-sleeve pool.

Outputs are written to examples/out/greedy_top5_by_sig_fam_portfolio/.
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

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
MAX_PAIRWISE_ABS_CORR = 0.75
MIN_NET_SHARPE_DELTA = 0.05
MAX_DD_REGRESSION = 0.01
MAX_TURNOVER_REGRESSION = 0.001
MIN_DD_IMPROVEMENT = 0.0025
MIN_TURNOVER_IMPROVEMENT = 0.00025
WEIGHTING_SCHEMES = ["equal", "equal_vol", "equal_sharpe", "optimal"]
MIN_SLEEVES = 3
COST_BPS = 10.0

OUT_DIR = Path(__file__).parent / "out" / "greedy_top5_by_sig_fam_portfolio"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    # --- Load data ---
    print(f"\nLoading data ({TRAIN_START} to {TRAIN_END}) ...")
    universe, benchmark, factors = pu.load_data(TRAIN_START, TRAIN_END)
    print(f"  Universe : {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

    # --- Distance partners & sector maps ---
    print("\nComputing distance partners on training data ...")
    partners = pu.compute_distance_partners(universe, train_end=TRAIN_END)
    get_distance_partners = lambda: partners  # noqa: E731

    sector_etf_map = pu.get_sector_etf_map_for(universe)
    get_sector_etf_map = lambda: sector_etf_map  # noqa: E731

    sector_map = qs.get_sector_map(list(universe.returns.columns))

    # --- Build 30 sleeve specs ---
    specs = build_top5_by_sig_fam_sleeve_specs(
        get_distance_partners=get_distance_partners,
        get_sector_etf_map=get_sector_etf_map,
    )
    print(f"\nRunning {len(specs)} sleeves ...")
    studies = pu.run_sleeve_pool(specs, universe, benchmark, factors, sector_map)
    print("Done.")

    # --- Sleeve return correlations ---
    sleeve_returns = pd.DataFrame({n: studies[n].cache["portfolio_returns"] for n in specs})
    corr = sleeve_returns.corr()

    # --- Greedy selection ---
    print("\nRunning greedy forward selection ...")
    selected, final_scheme, final_weights, buildout_rows, candidate_rows = pu.run_greedy_selection(
        spec_names=list(specs.keys()),
        studies=studies,
        sleeve_returns=sleeve_returns,
        corr=corr,
        universe=universe,
        benchmark=benchmark,
        cost_bps=COST_BPS,
        weighting_schemes=WEIGHTING_SCHEMES,
        max_pairwise_abs_corr=MAX_PAIRWISE_ABS_CORR,
        min_net_sharpe_delta=MIN_NET_SHARPE_DELTA,
        max_dd_regression=MAX_DD_REGRESSION,
        max_turnover_regression=MAX_TURNOVER_REGRESSION,
        min_dd_improvement=MIN_DD_IMPROVEMENT,
        min_turnover_improvement=MIN_TURNOVER_IMPROVEMENT,
        min_sleeves=MIN_SLEEVES,
    )

    print(f"\nSelected {len(selected)} sleeves:")
    for name in selected:
        print(f"  {name}")

    # --- Build final combined portfolio ---
    final_combined_positions = pu.combine_positions_fixed_weights(studies, final_weights, selected)
    final_metrics = pu.evaluate_fixed_weight_portfolio(
        final_combined_positions, universe, benchmark, COST_BPS
    )
    print(f"\nFinal portfolio net_sharpe: {pu.get_net_sharpe(final_metrics):.3f}")

    # --- Gross/net returns for selected portfolio ---
    universe_returns = universe.returns.reindex(columns=final_combined_positions.columns).fillna(0)
    bm_series = benchmark.returns["SPY"].reindex(universe_returns.index).fillna(0)

    import qstudy.study.engine as qs_engine
    import qstudy.study.metrics as qs_metrics

    gross_returns = qs_engine.run(final_combined_positions, universe_returns)
    to = qs_metrics.turnover(final_combined_positions)
    net_returns = gross_returns - to * COST_BPS / 10_000

    # --- Write CSVs ---
    buildout_df = pd.DataFrame(buildout_rows)
    buildout_df.to_csv(OUT_DIR / "selection_buildout.csv", index=False)
    print("Saved selection_buildout.csv")

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(OUT_DIR / "candidate_evaluations.csv", index=False)
    print("Saved candidate_evaluations.csv")

    selected_df = pd.DataFrame(
        [
            {"sleeve": n, "weight": final_weights.get(n, 0.0), "scheme": final_scheme}
            for n in selected
        ]
    )
    selected_df.to_csv(OUT_DIR / "selected_sleeves.csv", index=False)
    print("Saved selected_sleeves.csv")

    selected_corr = sleeve_returns[selected].corr()
    selected_corr.to_csv(OUT_DIR / "selected_sleeve_return_correlations.csv")
    print("Saved selected_sleeve_return_correlations.csv")

    returns_df = pd.DataFrame({"gross": gross_returns, "net": net_returns, "benchmark": bm_series})
    returns_df.to_csv(OUT_DIR / "selected_portfolio_daily_returns.csv")
    print("Saved selected_portfolio_daily_returns.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
