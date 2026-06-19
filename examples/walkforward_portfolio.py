"""
Walkforward analysis of the Target Portfolio.

Portfolio sleeves:
  - dist_mr_k3_z20__r21__cond__vol_contraction_10_60
  - monoton_120d__r21__vol_20_60__cond__none
  - mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
  - zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75
  - resid_gap_reversion__r10__trend_50_200__cond__none
  - dist_mr_k3_z60__r5__cond__vol_expansion_10_60
  - mr_5d__r10__trend_20_100_h__cond__breadth_weak_40

Folds (expanding train window):
  Fold 1: train 2015-2020, val 2021
  Fold 2: train 2015-2021, val 2022
  Fold 3: train 2015-2022, val 2023

For each year:
  - IS years (2015-2020): full-IS equal-vol weights, period="IS"
  - OOS years (2021-2023): weights calibrated on the fold's training period

Outputs (examples/out/walkforward_target_portfolio/):
  - walkforward_fold_summary.csv    — train/val metrics per fold
  - annual_performance.csv          — net sharpe, return, vol, drawdown, turnover by year
  - sleeve_contribution_by_year.csv        — per-sleeve contributions (ret, var, to) by year
  - sleeve_return_contribution_by_year.png — stacked bar chart of return contributions by year
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
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_PORTFOLIO = [
    "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
    "monoton_120d__r21__vol_20_60__cond__none",
    "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "resid_gap_reversion__r10__trend_50_200__cond__none",
    "dist_mr_k3_z60__r5__cond__vol_expansion_10_60",
    "mr_5d__r10__trend_20_100_h__cond__breadth_weak_40",
]

FOLDS = [
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]

FULL_START = "2015-01-01"
FULL_END = "2023-12-31"

COST_BPS = 10.0
WEIGHTING_SCHEMES = ["equal", "equal_vol", "equal_sharpe", "optimal"]

OUT_DIR = Path(__file__).parent / "out" / "walkforward_portfolio"

SHORT_LABELS = {
    "dist_mr_k3_z20__r21__cond__vol_contraction_10_60": "DPMR z20 (vol contract)",
    "monoton_120d__r21__vol_20_60__cond__none": "Monoton 120d (vol scaler)",
    "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "MR 5d (resid disp gate)",
    "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "ZScore MR (resid disp gate)",
    "resid_gap_reversion__r10__trend_50_200__cond__none": "Gap Rev (uncond)",
    "dist_mr_k3_z60__r5__cond__vol_expansion_10_60": "DPMR z60 (vol expand)",
    "mr_5d__r10__trend_20_100_h__cond__breadth_weak_40": "MR 5d half (breadth weak)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_weights(
    names: list[str], sleeve_returns_train: pd.DataFrame
) -> tuple[str, dict[str, float]]:
    best_scheme, best_ns = "equal", float("-inf")
    best_w: dict[str, float] = pu.estimate_weights_equal(names)
    for scheme in WEIGHTING_SCHEMES:
        if scheme == "equal":
            w = pu.estimate_weights_equal(names)
        elif scheme == "equal_vol":
            w = pu.estimate_weights_equal_vol(names, sleeve_returns_train)
        elif scheme == "equal_sharpe":
            w = pu.estimate_weights_equal_sharpe(names, sleeve_returns_train)
        else:
            w = pu.estimate_weights_optimal(names, sleeve_returns_train)
        port_ret = sum(sleeve_returns_train[n] * w.get(n, 0.0) for n in names)
        std = float(port_ret.std())
        ns = float(port_ret.mean() / std * (252**0.5)) if std > 0 else float("-inf")
        if ns > best_ns:
            best_ns, best_scheme, best_w = ns, scheme, w
    return best_scheme, best_w


def _portfolio_gross_net_to(
    combined_positions: pd.DataFrame,
    universe_returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (gross, net, turnover) for combined positions."""
    ur = universe_returns.reindex(columns=combined_positions.columns).fillna(0)
    gross = qs_engine.run(combined_positions, ur)
    to = qs_metrics.turnover(combined_positions)
    net = gross - to * COST_BPS / 10_000
    return gross, net, to


def _annual_metrics(
    net: pd.Series,
    gross: pd.Series,
    bm: pd.Series,
    to: pd.Series,
) -> list[dict]:
    rows = []
    for year in sorted(net.index.year.unique()):
        mask = net.index.year == year
        nr = net[mask]
        gr = gross[mask]
        b = bm.reindex(nr.index).fillna(0)
        t = to.reindex(nr.index).fillna(0)
        if len(nr) < 10 or nr.std() == 0:
            continue
        ann = np.sqrt(252)
        eq = (1 + nr.fillna(0)).cumprod()
        dd = ((eq - eq.cummax()) / eq.cummax()).min()
        rows.append(
            {
                "year": year,
                "net_sharpe": round(float(nr.mean() / nr.std() * ann), 3),
                "gross_sharpe": round(
                    float(gr.mean() / gr.std() * ann) if gr.std() > 0 else float("nan"), 3
                ),
                "ann_return": round(float((1 + nr).prod() - 1), 4),
                "ann_vol": round(float(nr.std() * ann), 4),
                "max_drawdown": round(float(dd), 4),
                "avg_daily_turnover": round(float(t.mean()), 5),
                "bm_corr": round(float(nr.corr(b)), 3),
            }
        )
    return rows


def _sleeve_contributions_for_year(
    year: int,
    combined_positions: pd.DataFrame,
    weights: dict[str, float],
    sleeve_positions: dict[str, pd.DataFrame],
    universe_returns: pd.DataFrame,
    bm_series: pd.Series,
    names: list[str],
) -> list[dict]:
    """Compute sleeve return/variance/turnover contributions for a single year."""
    mask = combined_positions.index.year == year
    comb_yr = combined_positions[mask]
    ur = universe_returns.reindex(columns=comb_yr.columns).fillna(0)
    ur_yr = ur[ur.index.year == year]

    port_gross, port_net, port_to = _portfolio_gross_net_to(comb_yr, ur_yr)
    if port_net.std() == 0 or len(port_net) < 10:
        return []

    port_mu = float(port_net.mean())
    port_var = float(port_net.var())
    port_to_mean = float(port_to.mean())

    rows = []
    for name in names:
        w = weights.get(name, 0.0)
        slv_pos_yr = (
            sleeve_positions[name]
            .reindex(index=comb_yr.index, columns=comb_yr.columns, fill_value=0.0)
            .fillna(0.0)
        )
        slv_gross, slv_net, slv_to = _portfolio_gross_net_to(slv_pos_yr, ur_yr)
        slv_net_yr = slv_net
        weighted = slv_net_yr * w

        # Return contribution
        ret_contrib = float(weighted.mean()) / port_mu if abs(port_mu) > 1e-14 else float("nan")

        # Variance contribution: cov(w*sleeve, portfolio) / var(portfolio)
        cov = float(weighted.cov(port_net)) if port_var > 1e-14 else float("nan")
        var_contrib = cov / port_var if port_var > 1e-14 else float("nan")

        # Turnover contribution
        slv_to_yr = float(slv_to.mean())
        to_contrib = (w * slv_to_yr) / port_to_mean if port_to_mean > 1e-12 else float("nan")

        slv_std = float(slv_net_yr.std())
        ann = np.sqrt(252)
        rows.append(
            {
                "year": year,
                "sleeve": SHORT_LABELS.get(name, name),
                "weight": round(w, 4),
                "sleeve_net_sharpe": round(
                    float(slv_net_yr.mean() / slv_std * ann) if slv_std > 0 else float("nan"), 3
                ),
                "sleeve_ann_return": round(float((1 + slv_net_yr).prod() - 1), 4),
                "ret_contrib": round(ret_contrib, 4),
                "var_contrib": round(var_contrib, 4),
                "to_contrib": round(to_contrib, 4),
                "sleeve_avg_to": round(slv_to_yr, 5),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    # --- Load data and run sleeves once on full window ---
    print(f"\nLoading data ({FULL_START} to {FULL_END}) ...")
    universe, benchmark, factors = pu.load_data(FULL_START, FULL_END)
    print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

    partners = pu.compute_distance_partners(universe, train_end=FULL_END)
    get_distance_partners = lambda: partners  # noqa: E731
    sector_etf_map = pu.get_sector_etf_map_for(universe)
    get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
    sector_map = qs.get_sector_map(list(universe.returns.columns))

    specs = build_top5_by_sig_fam_sleeve_specs(
        get_distance_partners=get_distance_partners,
        get_sector_etf_map=get_sector_etf_map,
    )
    specs_needed = {n: s for n, s in specs.items() if n in TARGET_PORTFOLIO}
    print(f"Running {len(specs_needed)} sleeves ...")
    studies = pu.run_sleeve_pool(
        specs_needed, universe, benchmark, factors, sector_map, verbose=True
    )
    print("Done.\n")

    sleeve_returns_full = pd.DataFrame(
        {n: studies[n].cache["portfolio_returns"] for n in TARGET_PORTFOLIO}
    )
    sleeve_positions = {n: studies[n].cache["positions"] for n in TARGET_PORTFOLIO}

    # Aligned universe returns (union of all sleeve position columns)
    all_cols = sorted({col for n in TARGET_PORTFOLIO for col in sleeve_positions[n].columns})
    universe_returns = universe.returns.reindex(columns=all_cols).fillna(0)
    bm_series = benchmark.returns["SPY"].reindex(universe_returns.index).fillna(0)

    # -----------------------------------------------------------------------
    # Walkforward folds: calibrate weights on train, evaluate on val
    # -----------------------------------------------------------------------
    fold_summary_rows: list[dict] = []

    # Map: year -> (scheme, weights) to use for that year's contribution analysis
    year_weights: dict[int, tuple[str, dict[str, float]]] = {}

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(FOLDS):
        fold_label = val_start[:4]
        val_year = int(fold_label)
        print(f"{'=' * 60}")
        print(f"Fold {fold_idx + 1}: train {train_start}..{train_end}, val {val_start}..{val_end}")

        sleeve_returns_train = sleeve_returns_full.loc[:train_end]
        best_scheme, weights = _best_weights(TARGET_PORTFOLIO, sleeve_returns_train)
        year_weights[val_year] = (best_scheme, weights)

        print(f"  Scheme: {best_scheme}")
        for n in TARGET_PORTFOLIO:
            print(f"    {SHORT_LABELS[n]:<35}: {weights[n]:.3f}")

        combined = pu.combine_positions_fixed_weights(studies, weights, TARGET_PORTFOLIO)

        # Train
        train_combined = combined.loc[:train_end]
        _, train_net, train_to = _portfolio_gross_net_to(train_combined, universe_returns)
        train_bm = bm_series.loc[:train_end]
        train_m = pu.evaluate_fixed_weight_portfolio_raw(
            train_combined,
            universe_returns.loc[:train_end],
            train_bm,
            COST_BPS,
        )

        # Val
        val_combined = combined.loc[val_start:val_end]
        _, val_net, val_to = _portfolio_gross_net_to(val_combined, universe_returns)
        val_bm = bm_series.loc[val_start:val_end]
        val_m = pu.evaluate_fixed_weight_portfolio_raw(
            val_combined,
            universe_returns.loc[val_start:val_end],
            val_bm,
            COST_BPS,
        )

        train_ns = pu.get_net_sharpe(train_m)
        val_ns = pu.get_net_sharpe(val_m)
        print(f"  Train net_sharpe={train_ns:.3f}, Val net_sharpe={val_ns:.3f}")

        fold_summary_rows.append(
            {
                "fold": fold_label,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "best_scheme": best_scheme,
                "train_net_sharpe": round(train_ns, 3),
                "train_ann_return": round(float(train_m.get("ann_return", float("nan"))), 4),
                "train_ann_vol": round(float(train_m.get("ann_vol", float("nan"))), 4),
                "train_max_drawdown": round(float(train_m.get("max_drawdown", float("nan"))), 4),
                "train_avg_daily_turnover": round(
                    float(train_m.get("avg_daily_turnover", float("nan"))), 5
                ),
                "val_net_sharpe": round(val_ns, 3),
                "val_ann_return": round(float(val_m.get("ann_return", float("nan"))), 4),
                "val_ann_vol": round(float(val_m.get("ann_vol", float("nan"))), 4),
                "val_max_drawdown": round(float(val_m.get("max_drawdown", float("nan"))), 4),
                "val_avg_daily_turnover": round(
                    float(val_m.get("avg_daily_turnover", float("nan"))), 5
                ),
            }
        )

    # -----------------------------------------------------------------------
    # IS weights (full 2015-2023): for IS years 2015-2020
    # -----------------------------------------------------------------------
    is_scheme, is_weights = _best_weights(TARGET_PORTFOLIO, sleeve_returns_full)
    print(f"\nFull-IS best scheme: {is_scheme}")
    for year in range(2015, 2021):
        year_weights[year] = (is_scheme, is_weights)

    # -----------------------------------------------------------------------
    # Annual performance and sleeve contributions
    # For each year, build combined positions using that year's calibrated weights,
    # then compute metrics on just that year's slice.
    # -----------------------------------------------------------------------
    annual_rows: list[dict] = []
    contrib_rows: list[dict] = []

    all_years = sorted(year_weights.keys())
    print(f"\nComputing per-year metrics for years: {all_years}")

    for year in all_years:
        period = "OOS" if year >= 2021 else "IS"
        scheme, weights = year_weights[year]
        combined = pu.combine_positions_fixed_weights(studies, weights, TARGET_PORTFOLIO)

        year_start = f"{year}-01-01"
        year_end = f"{year}-12-31"
        comb_yr = combined.loc[year_start:year_end]
        ur_yr = universe_returns.loc[year_start:year_end]
        bm_yr = bm_series.loc[year_start:year_end]

        gross_yr, net_yr, to_yr = _portfolio_gross_net_to(comb_yr, ur_yr)

        ann = _annual_metrics(net_yr, gross_yr, bm_yr, to_yr)
        for r in ann:
            r["period"] = period
            r["weights_scheme"] = scheme
        annual_rows.extend(ann)

        # Sleeve contributions
        yr_contrib = _sleeve_contributions_for_year(
            year, combined, weights, sleeve_positions, universe_returns, bm_series, TARGET_PORTFOLIO
        )
        for r in yr_contrib:
            r["period"] = period
            r["weights_scheme"] = scheme
        contrib_rows.extend(yr_contrib)

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    print("\nWriting outputs ...")

    pd.DataFrame(fold_summary_rows).to_csv(OUT_DIR / "walkforward_fold_summary.csv", index=False)
    print("Saved walkforward_fold_summary.csv")

    annual_df = (
        pd.DataFrame(annual_rows)
        .sort_values("year")
        .reset_index(drop=True)[
            [
                "year",
                "period",
                "weights_scheme",
                "net_sharpe",
                "gross_sharpe",
                "ann_return",
                "ann_vol",
                "max_drawdown",
                "avg_daily_turnover",
                "bm_corr",
            ]
        ]
    )
    annual_df.to_csv(OUT_DIR / "annual_performance.csv", index=False)
    print("Saved annual_performance.csv")

    contrib_df = (
        pd.DataFrame(contrib_rows)
        .sort_values(["year", "sleeve"])
        .reset_index(drop=True)[
            [
                "year",
                "period",
                "weights_scheme",
                "sleeve",
                "weight",
                "sleeve_net_sharpe",
                "sleeve_ann_return",
                "ret_contrib",
                "var_contrib",
                "to_contrib",
                "sleeve_avg_to",
            ]
        ]
    )
    contrib_df.to_csv(OUT_DIR / "sleeve_contribution_by_year.csv", index=False)
    print("Saved sleeve_contribution_by_year.csv")

    # --- Figure: sleeve return contribution by year (grouped bar) ---
    pivot_ret = contrib_df.pivot_table(
        index="year", columns="sleeve", values="ret_contrib", aggfunc="first"
    ).fillna(0)

    sleeve_order = [SHORT_LABELS[n] for n in TARGET_PORTFOLIO]
    pivot_ret = pivot_ret.reindex(columns=sleeve_order)

    colors = [
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#76b7b2",
        "#edc948",
        "#b07aa1",
    ]

    n_sleeves = len(sleeve_order)
    n_years = len(pivot_ret)
    width = 0.8 / n_sleeves
    x = np.arange(n_years)

    fig, ax = plt.subplots(figsize=(14, 5.5))

    for i, sleeve in enumerate(sleeve_order):
        offsets = x + (i - n_sleeves / 2 + 0.5) * width
        ax.bar(
            offsets,
            pivot_ret[sleeve].values.astype(float),
            width=width * 0.9,
            color=colors[i % len(colors)],
            label=sleeve,
            alpha=0.85,
        )

    # Mark OOS years with a lighter background
    for idx, yr in enumerate(pivot_ret.index):
        if yr >= 2021:
            ax.axvspan(idx - 0.45, idx + 0.45, color="lightyellow", zorder=0, alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{yr}\n(OOS)" if yr >= 2021 else str(yr) for yr in pivot_ret.index],
        fontsize=9,
    )
    ax.set_ylabel("Return Contribution (fraction of portfolio return)")
    ax.set_title("Sleeve Return Contributions by Year — Target Portfolio", pad=10)
    ax.legend(
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        borderaxespad=0,
    )
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "sleeve_return_contribution_by_year.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved sleeve_return_contribution_by_year.png")

    # --- Figure: annual net Sharpe / max drawdown / turnover (3 stacked subplots) ---
    years = annual_df["year"].tolist()
    x = np.arange(len(years))
    oos_mask = [yr >= 2021 for yr in years]

    bar_colors = ["#e15759" if oos else "#4e79a7" for oos in oos_mask]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    def _shade_oos(ax):
        for idx, oos in enumerate(oos_mask):
            if oos:
                ax.axvspan(idx - 0.45, idx + 0.45, color="lightyellow", zorder=0, alpha=0.6)

    # Panel 1: Net Sharpe
    ax1.bar(x, annual_df["net_sharpe"].values, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    ax1.axhline(0, color="black", linewidth=0.8)
    _shade_oos(ax1)
    ax1.set_ylabel("Net Sharpe")
    ax1.set_title("Annual Portfolio Performance — Target Portfolio", pad=10)
    ax1.grid(True, axis="y", alpha=0.25, zorder=0)
    for idx, val in enumerate(annual_df["net_sharpe"].values):
        ax1.text(idx, val + (0.05 if val >= 0 else -0.1), f"{val:.2f}",
                 ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    # Panel 2: Max Drawdown (show as positive percentage for readability)
    mdd_vals = annual_df["max_drawdown"].values * 100  # already negative, keep sign
    ax2.bar(x, mdd_vals, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.8)
    _shade_oos(ax2)
    ax2.set_ylabel("Max Drawdown (%)")
    ax2.grid(True, axis="y", alpha=0.25, zorder=0)
    for idx, val in enumerate(mdd_vals):
        ax2.text(idx, val - 0.3, f"{val:.1f}%",
                 ha="center", va="top", fontsize=8)

    # Panel 3: Avg Daily Turnover
    to_vals = annual_df["avg_daily_turnover"].values * 100
    ax3.bar(x, to_vals, color=bar_colors, alpha=0.85, width=0.6, zorder=3)
    _shade_oos(ax3)
    ax3.set_ylabel("Avg Daily Turnover (%)")
    ax3.grid(True, axis="y", alpha=0.25, zorder=0)
    ax3.set_xticks(x)
    ax3.set_xticklabels(
        [f"{yr}\n(OOS)" if yr >= 2021 else str(yr) for yr in years], fontsize=9
    )
    for idx, val in enumerate(to_vals):
        ax3.text(idx, val + 0.05, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#4e79a7", alpha=0.85, label="IS"),
        plt.Rectangle((0, 0), 1, 1, color="#e15759", alpha=0.85, label="OOS"),
    ]
    ax1.legend(handles=legend_handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "annual_performance_chart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved annual_performance_chart.png")

    # -----------------------------------------------------------------------
    # Print summaries
    # -----------------------------------------------------------------------
    print("\n--- Walkforward Fold Summary ---")
    fold_df = pd.DataFrame(fold_summary_rows)
    print(
        fold_df[
            [
                "fold",
                "best_scheme",
                "train_net_sharpe",
                "val_net_sharpe",
                "val_ann_return",
                "val_max_drawdown",
            ]
        ].to_string(index=False)
    )

    print("\n--- Annual Portfolio Performance ---")
    print(annual_df.to_string(index=False))

    print("\n--- Sleeve Contributions by Year ---")
    pivot_ret = contrib_df.pivot_table(
        index="sleeve", columns="year", values="ret_contrib", aggfunc="first"
    )
    print("\nReturn Contributions:")
    print(pivot_ret.to_string(float_format=lambda x: f"{x:.3f}"))

    pivot_var = contrib_df.pivot_table(
        index="sleeve", columns="year", values="var_contrib", aggfunc="first"
    )
    print("\nVariance Contributions:")
    print(pivot_var.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
