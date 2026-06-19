"""
Portfolio combinations: evaluate all combinations of tier-2 and tier-3 sleeves
added to the 2 fixed core sleeves.

Core sleeves are always included. All non-empty subsets of tier-2 and tier-3
sleeves are tried (128 combinations). Results are ranked by best net Sharpe
across all weighting schemes.

Outputs (examples/out/portfolio_combinations/):
  - portfolio_combinations_results.csv — all evaluations (combination x scheme)
  - portfolio_combinations_heatmap.png — net Sharpe heatmap: top 20 x scheme
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import portfolio_utils as pu
from sig_fam_utils import build_top5_by_sig_fam_sleeve_specs

import qstudy as qs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
COST_BPS = 10.0
WEIGHTING_SCHEMES = ["equal_sharpe", "optimal"]
TOP_N_HEATMAP = 10

# CORE_SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "monoton_120d__r21__vol_20_60__cond__none",
# ]

CORE_SLEEVES = [
    "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
]

CANDIDATE_SLEEVES = [
    "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "monoton_120d__r21__vol_20_60__cond__none",
    # "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "dist_mr_k3_z60__r10__cond__none",
    "dist_mr_k3_z60__r5__cond__vol_expansion_10_60",
    "dist_mr_k3_z10__r10__cond__panic_10d_minus5",
    "dist_mr_k3_z10__r10__cond__vol_contraction_10_60",
    "factor_model_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "etf_factor_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "etf_factor_resid_mr_5d__r10__trend_20_100__cond__market_trend_down_20_100",
    "resid_gap_reversion__r10__trend_50_200__cond__vol_contraction_10_60",
    "resid_gap_reversion__r10__trend_50_200__cond__sector_dislocation_5_q80",
    "resid_gap_reversion__r10__none__cond__none",
    "resid_gap_reversion__r10__trend_50_200__cond__none",
    "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "sector_rel_zscore_5_60__r10__trend_20_100__cond__vol_contraction_10_60",
    "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
    "resid_gap_reversion__r10__trend_50_200__cond__breadth_strong_55",
]

OUT_DIR = Path(__file__).parent / "out" / "portfolio_combinations"

SHORT = {
    "dist_mr_k3_z20__r21__cond__vol_contraction_10_60": "dpmr_z20_vc",
    "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "mr5d_rdg",
    "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75": "dpmr_z10_rdg",
    "monoton_120d__r21__vol_20_60__cond__none": "mon120_vol",
    "resid_gap_reversion__r10__trend_50_200__cond__breadth_strong_55": "gap_bstr",
    "dist_mr_k3_z60__r10__cond__none": "dpmr_z60",
    "dist_mr_k3_z60__r5__cond__vol_expansion_10_60": "dpmr_z60_ve",
    "mr_5d__r10__trend_50_200__cond__breadth_weak_40": "mr5d_bwk",
    "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "zscore_rdg",
    "resid_gap_reversion__r10__none__cond__none": "gap_no_cond",
    "factor_model_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "Factor Resid MR 5d",
    "etf_factor_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "ETF Factor Resid MR",
    "factor_model_resid_mr_2d__r10__trend_50_200__cond__dispersion_high_60_q75": "Factor Resid MR 2d",
    "mr_5d__r10__trend_20_100__cond__breadth_weak_40": "MR 5d (breadth weak)",
    "mr_5d__r10__trend_20_100_h__cond__breadth_weak_40": "MR 5d Half (breadth weak)",
    "dist_mr_k3_z10__r10__cond__vol_contraction_10_60": "DPMR z10 (vol contract)",
    "etf_factor_resid_mr_5d__r10__trend_20_100__cond__market_trend_down_20_100": "ETF Factor MR (downtrend)",
    "etf_factor_resid_mr_5d__r10__trend_20_100_h__cond__market_trend_down_20_100": "ETF Factor MR Half (downtrend)",
    "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "Sector Rel MR",
    "resid_gap_reversion__r10__trend_50_200__cond__vol_contraction_10_60": "Gap Rev (vol contract)",
    "sector_rel_zscore_5_60__r10__trend_20_100__cond__vol_contraction_10_60": "Sector ZScore (vol contract)",
    "sector_rel_zscore_5_60__r10__trend_20_100_h__cond__vol_contraction_10_60": "Sector ZScore Half (vol contract)",
    "resid_gap_reversion__r10__trend_50_200__cond__sector_dislocation_5_q80": "Gap Rev (sector disloc)",
    "monoton_120d__r21__disp_60_q30__cond__none": "Mon120 (disp q30)",
    "sector_rel_zscore_5_60__r10__trend_20_100__cond__breadth_weak_40": "Sector ZScore (breadth weak)",
    "monoton_120d__r21__crash_10_5pct__cond__market_trend_down_20_100": "Mon120 (crash)",
    "monoton_120d__r21__vol_20_60__cond__panic_10d_minus5": "Mon120 (panic)",
    "monoton_120d__r21__disp_60_q20__cond__none": "Mon120 (disp q20)",
    "sector_rel_zscore_5_60__r10__trend_20_100_h__cond__breadth_weak_40": "Sector ZScore Half (breadth weak)",
    "resid_gap_reversion__r10__trend_50_200__cond__none": "Gap Rev (trend)",
    "dist_mr_k3_z10__r10__cond__panic_10d_minus5": "DPMR z10 (panic)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluate_all_schemes(
    label: str,
    names: list[str],
    studies: dict,
    sleeve_returns: pd.DataFrame,
    universe,
    benchmark,
) -> tuple[list[dict], list[dict]]:
    """Returns (full_period_rows, annual_rows)."""
    rows = []
    annual_rows = []

    universe_returns = universe.returns.fillna(0)
    bm_series = benchmark.returns["SPY"].reindex(universe_returns.index).fillna(0)
    years = sorted(universe_returns.index.year.unique())

    for scheme in WEIGHTING_SCHEMES:
        if scheme == "equal_vol":
            weights = pu.estimate_weights_equal_vol(names, sleeve_returns)
        elif scheme == "equal_sharpe":
            weights = pu.estimate_weights_equal_sharpe(names, sleeve_returns)
        else:
            weights = pu.estimate_weights_optimal(names, sleeve_returns)

        combined = pu.combine_positions_fixed_weights(studies, weights, names)
        m = pu.evaluate_fixed_weight_portfolio(combined, universe, benchmark, COST_BPS)
        rows.append(
            {
                "portfolio": label,
                "sleeve_count": len(names),
                "weighting_scheme": scheme,
                "net_sharpe": pu.get_net_sharpe(m),
                "gross_sharpe": float(m.get("gross_sharpe", m.get("sharpe", float("nan")))),
                "ann_return": float(m.get("ann_return", float("nan"))),
                "ann_vol": float(m.get("ann_vol", float("nan"))),
                "max_drawdown": float(m.get("max_drawdown", float("nan"))),
                "avg_daily_turnover": float(m.get("avg_daily_turnover", float("nan"))),
                "sleeves": "|".join(names),
            }
        )

        # Per-year breakdown
        for year in years:
            yr_mask = combined.index.year == year
            yr_combined = combined[yr_mask]
            yr_universe = universe_returns[universe_returns.index.year == year]
            yr_bm = bm_series[bm_series.index.year == year]
            if yr_combined.empty:
                continue
            ym = pu.evaluate_fixed_weight_portfolio_raw(yr_combined, yr_universe, yr_bm, COST_BPS)
            annual_rows.append(
                {
                    "portfolio": label,
                    "year": year,
                    "weighting_scheme": scheme,
                    "net_sharpe": pu.get_net_sharpe(ym),
                    "ann_return": float(ym.get("ann_return", float("nan"))),
                    "ann_vol": float(ym.get("ann_vol", float("nan"))),
                    "max_drawdown": float(ym.get("max_drawdown", float("nan"))),
                    "avg_daily_turnover": float(ym.get("avg_daily_turnover", float("nan"))),
                }
            )

    return rows, annual_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    # --- Load data ---
    print(f"\nLoading data ({TRAIN_START} to {TRAIN_END}) ...")
    universe, benchmark, factors = pu.load_data(TRAIN_START, TRAIN_END)
    print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

    # --- Distance partners & sector maps ---
    partners = pu.compute_distance_partners(universe, train_end=TRAIN_END)
    get_distance_partners = lambda: partners  # noqa: E731
    sector_etf_map = pu.get_sector_etf_map_for(universe)
    get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
    sector_map = qs.get_sector_map(list(universe.returns.columns))

    # --- Build combos ---
    # To run the full grid, set target_portfolios = None.
    # To evaluate specific portfolios only, list their full sleeve sets here.
    target_portfolios: list[list[str]] | None = None

    # --- Build specs and run only needed sleeves ---
    all_needed = set(CORE_SLEEVES + CANDIDATE_SLEEVES)
    if target_portfolios is not None:
        for sleeve_list in target_portfolios:
            all_needed.update(sleeve_list)
    specs = build_top5_by_sig_fam_sleeve_specs(
        get_distance_partners=get_distance_partners,
        get_sector_etf_map=get_sector_etf_map,
    )
    specs_needed = {n: s for n, s in specs.items() if n in all_needed}
    print(f"\nRunning {len(specs_needed)} sleeves ...")
    studies = pu.run_sleeve_pool(specs_needed, universe, benchmark, factors, sector_map)
    print("Done.")

    sleeve_returns = pd.DataFrame({n: studies[n].cache["portfolio_returns"] for n in specs_needed})

    if target_portfolios is not None:
        # Evaluate only the specified portfolios
        named_combos = [
            ("+".join(SHORT[s] for s in names if s not in CORE_SLEEVES), names)
            for names in target_portfolios
        ]
        named_combos = [
            (f"core+{label}" if label else "core", names) for label, names in named_combos
        ]
    else:
        # Core-only baseline + core + each candidate sleeve individually
        named_combos = [("core", CORE_SLEEVES)]
        for sleeve in CANDIDATE_SLEEVES:
            label = f"core+{SHORT.get(sleeve, sleeve)}"
            named_combos.append((label, CORE_SLEEVES + [sleeve]))

    print(f"\nEvaluating {len(named_combos)} combinations x {len(WEIGHTING_SCHEMES)} schemes ...")

    all_rows: list[dict] = []
    all_annual_rows: list[dict] = []
    for i, (label, names) in enumerate(named_combos):
        rows, annual_rows = _evaluate_all_schemes(
            label, names, studies, sleeve_returns, universe, benchmark
        )
        best_ns = max(r["net_sharpe"] for r in rows)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(named_combos)}] {label}: best_ns={best_ns:.3f}")
        all_rows.extend(rows)
        all_annual_rows.extend(annual_rows)

    # --- Write results CSVs ---
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUT_DIR / "portfolio_combinations_results.csv", index=False)
    print(f"\nSaved portfolio_combinations_results.csv ({len(results_df)} rows)")

    annual_df = pd.DataFrame(all_annual_rows)
    annual_df.to_csv(OUT_DIR / "portfolio_combinations_annual.csv", index=False)
    print(f"Saved portfolio_combinations_annual.csv ({len(annual_df)} rows)")

    # --- Pick top N by best net Sharpe across any scheme ---
    best_per_combo = (
        results_df.groupby("portfolio")["net_sharpe"]
        .max()
        .sort_values(ascending=False)
        .head(TOP_N_HEATMAP)
    )
    top_labels = best_per_combo.index.tolist()

    # --- Heatmap: top N portfolios x weighting schemes ---
    heatmap_df = results_df[results_df["portfolio"].isin(top_labels)]
    pivot = heatmap_df.pivot_table(
        index="portfolio", columns="weighting_scheme", values="net_sharpe", aggfunc="first"
    ).reindex(columns=WEIGHTING_SCHEMES)
    pivot = pivot.reindex(top_labels)

    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.45 + 1.5)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="YlGnBu")
    plt.colorbar(im, ax=ax, label="Net Sharpe")

    ax.set_xticks(range(len(WEIGHTING_SCHEMES)))
    ax.set_xticklabels(WEIGHTING_SCHEMES, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(f"Net Sharpe: Top {TOP_N_HEATMAP} Portfolio Combinations (2015-2023)", pad=10)

    valid = pivot.values[~np.isnan(pivot.values)]
    vmax = valid.max() if len(valid) else 1.0
    for i in range(len(pivot)):
        for j in range(len(WEIGHTING_SCHEMES)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "black" if val < vmax * 0.85 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "portfolio_combinations_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved portfolio_combinations_heatmap.png")

    # --- Print top 20 summary ---
    summary = (
        results_df[results_df["portfolio"].isin(top_labels)]
        .sort_values("net_sharpe", ascending=False)
        .drop_duplicates("portfolio")[
            [
                "portfolio",
                "weighting_scheme",
                "net_sharpe",
                "ann_return",
                "ann_vol",
                "max_drawdown",
                "avg_daily_turnover",
                "sleeve_count",
            ]
        ]
        .set_index("portfolio")
        .reindex(top_labels)
    )
    print(f"\nTop {TOP_N_HEATMAP} combinations by best net Sharpe:")
    print(summary.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
