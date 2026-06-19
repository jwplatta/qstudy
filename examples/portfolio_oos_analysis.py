from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import portfolio_utils as pu
from sig_fam_utils import build_top5_by_sig_fam_sleeve_specs

import qstudy as qs
import qstudy.study.engine as qs_engine
import qstudy.study.metrics as qs_metrics
from qstudy.study.metrics import drawdown_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
OOS_START = "2015-01-01"
OOS_END = "2026-05-29"
OOS_SPLIT = pd.Timestamp("2024-01-01")
VOL_TARGET = 0.10
MAX_LEVERAGE = 15.0
ROLLING_WINDOW = 90
COST_BPS = 10.0
OUT_DIR = Path(__file__).parent / "out" / "portfolio_oos_analysis"

# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     # "mr_5d__r10__trend_20_100_h__cond__breadth_weak_40",
#     "resid_gap_reversion__r10__trend_50_200__cond__none",
#     # "dist_mr_k3_z10__r10__cond__panic_10d_minus5",
#     # "dist_mr_k3_z60__r10__cond__none",
#     "dist_mr_k3_z60__r5__cond__vol_expansion_10_60",
# ]

# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "mr_5d__r10__trend_50_200__cond__breadth_weak_40",
# ]


# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
#     "resid_gap_reversion__r10__trend_50_200__cond__breadth_strong_55",
# ]

# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "sector_rel_zscore_5_60__r10__trend_20_100__cond__vol_contraction_10_60",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "mr_5d__r10__trend_50_200__cond__breadth_weak_40",
#     "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
# ]


# NOTE: inital core portfolio
# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "monoton_120d__r21__vol_20_60__cond__none",
# ]

# NOTE: replacing mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
# with sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75
# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
# ]

# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
# ]

# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
# ]

# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "mr_5d__r10__trend_50_200__cond__breadth_weak_40",
# ]

# SLEEVES = [
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     # "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
#     # "mr_5d__r10__trend_50_200__cond__breadth_weak_40",
#     # "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
# ]

# "monoton_120d__r21__vol_20_60__cond__none",
# "monoton_120d__r21__disp_60_q30__cond__none",
# "etf_factor_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
# "factor_model_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
# SLEEVES = [
#     # "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "monoton_120d__r21__vol_20_60__cond__panic_10d_minus5",
#     "monoton_120d__r21__crash_10_5pct__cond__market_trend_down_20_100",
#     "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
# ]

SLEEVES = [
    "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
    "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
    "monoton_120d__r21__vol_20_60__cond__none",
    "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
    # "dist_mr_k3_z60__r5__cond__vol_expansion_10_60",
]

# SLEEVES = [
#     "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75",
#     "dist_mr_k3_z20__r21__cond__vol_contraction_10_60",
#     "monoton_120d__r21__vol_20_60__cond__none",
#     "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75",
#     # "mr_5d__r10__trend_50_200__cond__breadth_weak_40",
# ]


SHORT_LABELS = {
    "dist_mr_k3_z20__r21__cond__vol_contraction_10_60": "Dist Pairs MR (z20, vol contraction)",
    "monoton_120d__r21__vol_20_60__cond__none": "Monotonic Momentum (120d, vol-scaled)",
    "mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": (
        "Active-Return MR (5d, resid-disp gate)"
    ),
    "zscore_rev_5_60__r10__trend_20_100__cond__residual_dispersion_high_20_q75": (
        "Z-Score MR (5/60, resid-disp gate)"
    ),
    "mr_5d__r10__trend_20_100_h__cond__breadth_weak_40": (
        "Active-Return MR (5d, breadth-weak, half-scale)"
    ),
    "resid_gap_reversion__r10__trend_50_200__cond__none": "Resid Gap Reversion (unconditioned)",
    "dist_mr_k3_z10__r10__cond__panic_10d_minus5": "Dist Pairs MR Panic",
    "dist_mr_k3_z60__r10__cond__none": "Dist Pair Always On",
    "dist_mr_k3_z60__r5__cond__vol_expansion_10_60": "Dis Pairs MR (vol expansion)",
    "mr_5d__r10__trend_50_200__cond__breadth_weak_40": "mr5d_bwk",
    "dist_mr_k3_z10__r10__cond__residual_dispersion_high_20_q75": "Dist Pairs MR (z10, resid disp)",
    "resid_gap_reversion__r10__trend_50_200__cond__breadth_strong_55": "Resid Gap Reversion (trend)",
    "sector_rel_zscore_5_60__r10__trend_20_100__cond__vol_contraction_10_60": "Sector Relative",
    "sector_rel_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "Sector Relative Dispersion",
    "factor_model_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "factor_model_resid_mr_5d__r10",
    "etf_factor_resid_mr_5d__r10__trend_20_100__cond__residual_dispersion_high_20_q75": "etf_factor_resid_mr_5d",
    "monoton_120d__r21__vol_20_60__cond__panic_10d_minus5": "monoton_120d(panic)",
    "monoton_120d__r21__disp_60_q30__cond__none": "monoton_120d__r21__disp_60_q30__cond__none",
    "monoton_120d__r21__crash_10_5pct__cond__market_trend_down_20_100": "monoton_120d__r21__crash",
}

COST_VALS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]

COLORS = [
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759",
    "#76b7b2", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading data ...")
    universe, benchmark, factors = pu.load_data(TRAIN_START, TRAIN_END)
    partners = pu.compute_distance_partners(universe, train_end=TRAIN_END)
    sector_etf_map = pu.get_sector_etf_map_for(universe)
    sector_map = qs.get_sector_map(list(universe.returns.columns))

    # --- Build sleeve specs ---
    all_specs = build_top5_by_sig_fam_sleeve_specs(
        get_distance_partners=lambda: partners,
        get_sector_etf_map=lambda: sector_etf_map,
    )
    specs_needed = {n: all_specs[n] for n in SLEEVES}

    # --- Run sleeves ---
    print("Running sleeves ...")
    studies = pu.run_sleeve_pool(specs_needed, universe, benchmark, factors, sector_map)

    # --- Build sleeve returns DataFrame ---
    sleeve_returns = pd.DataFrame(
        {n: studies[n].cache["portfolio_returns"] for n in SLEEVES}
    ).dropna()

    # --- Optimal portfolio weights ---
    weights = pu.estimate_weights_optimal(SLEEVES, sleeve_returns)
    print("Optimal weights:")
    for n, w in weights.items():
        print(f"  {SHORT_LABELS[n]}: {w:.4f}")

    # --- Combined portfolio ---
    combined = pu.combine_positions_fixed_weights(studies, weights, SLEEVES)
    portfolio_metrics = pu.evaluate_fixed_weight_portfolio(
        combined, universe, benchmark, cost_bps=COST_BPS
    )
    net_sharpe_val = portfolio_metrics.get("net_sharpe", portfolio_metrics.get("sharpe"))
    print(f"\nPortfolio net Sharpe @ {COST_BPS:.0f} bps: {net_sharpe_val:.3f}")

    # -----------------------------------------------------------------------
    # Output 1: Sleeve Attribution CSV
    # -----------------------------------------------------------------------
    print("\nBuilding sleeve attribution ...")

    # Equal-vol weights for attribution
    vols = sleeve_returns[SLEEVES].std()
    inv_vol = 1.0 / vols.clip(lower=1e-12)
    ev_weights = (inv_vol / inv_vol.sum()).to_dict()

    w_arr = np.array([ev_weights[n] for n in SLEEVES])

    cov = sleeve_returns[SLEEVES].cov().values
    mu = sleeve_returns[SLEEVES].mean().values

    portfolio_mean = float(w_arr @ mu)
    portfolio_variance = float(w_arr @ cov @ w_arr)

    rows = []
    for i, n in enumerate(SLEEVES):
        m = studies[n].metrics_dict()
        wi = w_arr[i]
        sleeve_to = float(qs_metrics.turnover(studies[n].cache["positions"]).mean())
        portfolio_to = float(
            sum(
                w_arr[j] * qs_metrics.turnover(studies[j_name].cache["positions"]).mean()
                for j, j_name in enumerate(SLEEVES)
            )
        )

        ret_contrib = (
            (wi * float(mu[i]) / portfolio_mean * 100)
            if abs(portfolio_mean) > 1e-16
            else float("nan")
        )
        sigma_w = cov @ w_arr
        var_contrib = (
            (wi * float(sigma_w[i]) / portfolio_variance * 100)
            if abs(portfolio_variance) > 1e-16
            else float("nan")
        )
        efficiency = (ret_contrib / var_contrib) if abs(var_contrib) > 1e-16 else float("nan")
        to_contrib = (
            (wi * sleeve_to / portfolio_to * 100) if abs(portfolio_to) > 1e-16 else float("nan")
        )

        rows.append(
            {
                "sleeve": SHORT_LABELS[n],
                "weight_pct": round(wi * 100, 2),
                "sharpe": round(float(m.get("sharpe", float("nan"))), 3),
                "ann_return_pct": round(float(m.get("ann_return", float("nan"))) * 100, 2),
                "ret_contrib_pct": round(ret_contrib, 2),
                "var_contrib_pct": round(var_contrib, 2),
                "efficiency": round(efficiency, 3),
                "to_contrib_pct": round(to_contrib, 2),
            }
        )

    attr_df = pd.DataFrame(rows).set_index("sleeve")
    out_csv = OUT_DIR / "sleeve_attribution.csv"
    attr_df.to_csv(out_csv)
    print(f"Saved: {out_csv}")
    print(attr_df.to_string())

    # -----------------------------------------------------------------------
    # Output 2: Correlation Heatmap
    # -----------------------------------------------------------------------
    print("\nBuilding correlation heatmap ...")

    corr_df = sleeve_returns.rename(columns=SHORT_LABELS).corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("P9c Portfolio: Sleeve Return Correlations (2015-2023)", fontsize=13)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    out_heatmap = OUT_DIR / "sleeve_correlation_heatmap.png"
    fig.savefig(out_heatmap, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_heatmap}")

    # -----------------------------------------------------------------------
    # Output 3: Net Sharpe vs Transaction Cost
    # -----------------------------------------------------------------------
    print("\nBuilding cost sensitivity chart ...")

    # Align universe returns to combined position columns
    universe_returns_aligned = universe.returns.reindex(columns=combined.columns).fillna(0)

    # Portfolio gross returns and turnover
    port_gross = qs_engine.run(combined, universe_returns_aligned)
    port_to = qs_metrics.turnover(combined)

    # Portfolio net Sharpe at each cost
    port_net_sharpes = []
    for cost in COST_VALS:
        net = port_gross - port_to * cost / 10_000
        ns = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan")
        port_net_sharpes.append(ns)

    # Per-sleeve net Sharpe at each cost
    sleeve_net_sharpes: dict[str, list[float]] = {n: [] for n in SLEEVES}
    for n in SLEEVES:
        gross = studies[n].cache["portfolio_returns"]
        to = qs_metrics.turnover(studies[n].cache["positions"])
        for cost in COST_VALS:
            net = gross - to * cost / 10_000
            ns = float(net.mean() / net.std() * np.sqrt(252)) if net.std() > 0 else float("nan")
            sleeve_net_sharpes[n].append(ns)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        COST_VALS,
        port_net_sharpes,
        linestyle="-",
        linewidth=1.5,
        color="#333333",
        marker="o",
        markersize=4,
        label="Portfolio",
    )
    for i, n in enumerate(SLEEVES):
        ax.plot(
            COST_VALS,
            sleeve_net_sharpes[n],
            linestyle="-",
            linewidth=1.0,
            marker="o",
            markersize=3,
            color=COLORS[i % len(COLORS)],
            label=SHORT_LABELS[n],
        )
    ax.axvline(x=10, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Net Sharpe")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_cost = OUT_DIR / "cost_sensitivity.png"
    fig.savefig(out_cost, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_cost}")

    # -----------------------------------------------------------------------
    # Output 4: OOS analysis (2015-2026)
    # -----------------------------------------------------------------------
    print(f"\nLoading extended data ({OOS_START} to {OOS_END}) ...")
    oos_universe, oos_benchmark, oos_factors = pu.load_data(OOS_START, OOS_END)
    print(f"  Universe: {oos_universe.returns.shape}")

    # Distance partners computed on IS data only — do not refit on OOS window
    oos_partners = pu.compute_distance_partners(universe, train_end=TRAIN_END)
    oos_sector_etf_map = pu.get_sector_etf_map_for(oos_universe)
    oos_sector_map = qs.get_sector_map(list(oos_universe.returns.columns))

    all_oos_specs = build_top5_by_sig_fam_sleeve_specs(
        get_distance_partners=lambda: oos_partners,
        get_sector_etf_map=lambda: oos_sector_etf_map,
    )
    oos_specs_needed = {n: all_oos_specs[n] for n in SLEEVES}

    print("Running sleeves on full window (2015-2026) ...")
    oos_studies = pu.run_sleeve_pool(
        oos_specs_needed, oos_universe, oos_benchmark, oos_factors, oos_sector_map
    )

    # IS-calibrated optimal weights (from training returns)
    oos_sleeve_returns = pd.DataFrame(
        {n: oos_studies[n].cache["portfolio_returns"] for n in SLEEVES}
    )
    is_sleeve_returns = oos_sleeve_returns.loc[:TRAIN_END].dropna()
    oos_weights = pu.estimate_weights_optimal(SLEEVES, is_sleeve_returns)

    print("OOS weights (IS-calibrated):")
    for n, w in oos_weights.items():
        print(f"  {SHORT_LABELS[n]}: {w:.4f}")

    # Build combined positions on full window
    oos_combined = pu.combine_positions_fixed_weights(oos_studies, oos_weights, SLEEVES)

    oos_universe_aligned = oos_universe.returns.reindex(columns=oos_combined.columns).fillna(0)
    oos_gross = qs_engine.run(oos_combined, oos_universe_aligned)
    oos_to = qs_metrics.turnover(oos_combined)
    oos_net_unlev = oos_gross - oos_to * COST_BPS / 10_000

    # Vol-target leverage: compute rolling IS vol, apply as forward-looking lever
    # Use a 63-day rolling vol estimate; cap leverage at MAX_LEVERAGE
    is_vol = oos_net_unlev.loc[:TRAIN_END].std() * np.sqrt(252)
    lever_scale = min(VOL_TARGET / is_vol, MAX_LEVERAGE) if is_vol > 0 else 1.0
    oos_net_lev = oos_net_unlev * lever_scale

    bm_oos = oos_benchmark.returns["SPY"]

    # --- Annual performance table ---
    print("\nBuilding annual performance table ...")
    annual_rows = []
    for year in sorted(oos_net_lev.index.year.unique()):
        lev_yr = oos_net_lev[oos_net_lev.index.year == year]
        bm_yr = bm_oos.reindex(lev_yr.index)
        if len(lev_yr) < 20:
            continue
        s = qs_metrics.summary(lev_yr)
        bm_ret_yr = float((1 + bm_yr.fillna(0)).prod() - 1)
        annual_rows.append(
            {
                "Year": year,
                "Period": "OOS" if year >= OOS_SPLIT.year else "IS",
                "Lev. Ret.": f"{s['ann_return']:.1%}",
                "Lev. SR": round(float(s["sharpe"]), 3),
                "Lev. Vol": f"{s['ann_vol']:.1%}",
                "Lev. Max DD": f"{s['max_drawdown']:.1%}",
                "SPY Ret.": f"{bm_ret_yr:.1%}",
            }
        )

    annual_df = pd.DataFrame(annual_rows).set_index("Year")
    annual_df.to_csv(OUT_DIR / "annual_performance.csv")
    print("Saved annual_performance.csv")
    print(annual_df.to_string())

    # --- Figure: equity curve, drawdown, rolling Sharpe ---
    print("\nBuilding OOS summary chart ...")

    unlev_eq = (1 + oos_net_unlev.fillna(0)).cumprod()
    lev_eq = (1 + oos_net_lev.fillna(0)).cumprod()
    bm_eq = (1 + bm_oos.fillna(0)).cumprod().reindex(lev_eq.index)

    def rolling_sharpe(ret: pd.Series, window: int) -> pd.Series:
        r = ret.fillna(0)
        mu = r.rolling(window).mean()
        sigma = r.rolling(window).std()
        return (mu / sigma.clip(lower=1e-10) * np.sqrt(252)).where(sigma > 1e-10)

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    ax = axes[0]
    ax.plot(unlev_eq.index, unlev_eq, color=COLORS[0], lw=1.0, label="Unlevered")
    ax.plot(
        lev_eq.index,
        lev_eq,
        color=COLORS[2],
        lw=1.0,
        label=f"Levered ({VOL_TARGET:.0%} Target Vol)",
    )
    ax.plot(bm_eq.index, bm_eq, color=COLORS[9], lw=1.0, ls="--", label="SPY", alpha=0.9)
    ax.axvline(OOS_SPLIT, color="gray", ls="--", lw=1.2, label="IS/OOS split")
    ax.set_ylabel("Equity")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    unlev_dd = drawdown_series(oos_net_unlev)
    lev_dd = drawdown_series(oos_net_lev)
    ax.plot(unlev_dd.index, unlev_dd, color=COLORS[0], lw=1.0, alpha=0.75, label="Unlevered")
    ax.plot(
        lev_dd.index,
        lev_dd,
        color=COLORS[2],
        lw=1.0,
        alpha=0.75,
        label=f"Levered ({VOL_TARGET:.0%} Target Vol)",
    )
    ax.axvline(OOS_SPLIT, color="gray", ls="--", lw=1.0)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(
        rolling_sharpe(oos_net_unlev, ROLLING_WINDOW).index,
        rolling_sharpe(oos_net_unlev, ROLLING_WINDOW),
        color=COLORS[0],
        lw=0.9,
    )
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(OOS_SPLIT, color="gray", ls="--", lw=1.0)
    ax.set_ylabel(f"Rolling Sharpe ({ROLLING_WINDOW}d)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)

    # Annual x-axis ticks on all panels
    import matplotlib.dates as mdates

    for a in axes:
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "oos_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved oos_summary.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
