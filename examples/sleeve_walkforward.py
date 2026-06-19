"""
Walkforward validation for every sleeve in sig_fam_utils.py.

For each fold (expanding train window, fixed val year), runs all 31 sleeves
on the full window, then evaluates each sleeve's out-of-sample performance on
the validation year using the positions computed from the full window.

Training windows:
  Fold 2021: train 2015-2020, val 2021
  Fold 2022: train 2015-2021, val 2022
  Fold 2023: train 2015-2022, val 2023

Outputs (examples/out/sleeve_walkforward/):
  - sleeve_walkforward.csv          — one row per sleeve x val year
  - sleeve_walkforward_sharpe.png   — heatmap: sleeve x val year (net Sharpe)
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

FOLDS = [
    ("2015-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2015-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2015-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
]

COST_BPS = 10.0

OUT_DIR = Path(__file__).parent / "out" / "sleeve_walkforward"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")

    all_rows: list[dict] = []

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(FOLDS):
        val_year = val_start[:4]
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}: train {train_start}..{train_end}, val {val_start}..{val_end}")
        print("=" * 60)

        print(f"  Loading data ({train_start} to {val_end}) ...")
        universe, benchmark, factors = pu.load_data(train_start, val_end)
        print(f"  Universe: {universe.returns.shape[0]} days x {universe.returns.shape[1]} tickers")

        partners = pu.compute_distance_partners(universe, train_end=train_end)
        get_distance_partners = lambda: partners  # noqa: E731
        sector_etf_map = pu.get_sector_etf_map_for(universe)
        get_sector_etf_map = lambda: sector_etf_map  # noqa: E731
        sector_map = qs.get_sector_map(list(universe.returns.columns))

        specs = build_top5_by_sig_fam_sleeve_specs(
            get_distance_partners=get_distance_partners,
            get_sector_etf_map=get_sector_etf_map,
        )

        print(f"  Running {len(specs)} sleeves on full window ...")
        studies = pu.run_sleeve_pool(specs, universe, benchmark, factors, sector_map, verbose=False)
        print("  Done.")

        val_universe = universe.returns.loc[val_start:val_end].fillna(0)
        val_bm = benchmark.returns["SPY"].loc[val_start:val_end].fillna(0)

        for name, study in studies.items():
            positions = study.cache["positions"].loc[val_start:val_end]
            if positions.empty:
                continue

            univ_aligned = val_universe.reindex(columns=positions.columns).fillna(0)
            gross_ret = qs_engine.run(positions, univ_aligned)
            to = qs_metrics.turnover(positions)
            net_ret = gross_ret - to * COST_BPS / 10_000

            n = len(net_ret.dropna())
            if n < 20 or net_ret.std() == 0:
                all_rows.append(
                    {
                        "sleeve": name,
                        "val_year": val_year,
                        "net_sharpe": float("nan"),
                        "gross_sharpe": float("nan"),
                        "ann_return": float("nan"),
                        "ann_vol": float("nan"),
                        "max_drawdown": float("nan"),
                        "avg_daily_turnover": float("nan"),
                    }
                )
                continue

            ann_ret_net = float((1 + net_ret).prod() ** (252 / n) - 1)
            ann_vol_net = float(net_ret.std() * (252**0.5))
            net_sharpe = ann_ret_net / ann_vol_net if ann_vol_net > 0 else float("nan")

            gross_n = len(gross_ret.dropna())
            ann_ret_gross = float((1 + gross_ret).prod() ** (252 / gross_n) - 1)
            ann_vol_gross = float(gross_ret.std() * (252**0.5))
            gross_sharpe = ann_ret_gross / ann_vol_gross if ann_vol_gross > 0 else float("nan")

            cum = (1 + net_ret).cumprod()
            mdd = float((cum / cum.cummax() - 1).min())
            avg_to = float(to.loc[val_start:val_end].mean())

            all_rows.append(
                {
                    "sleeve": name,
                    "val_year": val_year,
                    "net_sharpe": net_sharpe,
                    "gross_sharpe": gross_sharpe,
                    "ann_return": ann_ret_net,
                    "ann_vol": ann_vol_net,
                    "max_drawdown": mdd,
                    "avg_daily_turnover": avg_to,
                }
            )

        print(f"  Recorded {len(specs)} sleeve results for {val_year}.")

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "sleeve_walkforward.csv", index=False)
    print(f"\nSaved sleeve_walkforward.csv ({len(df)} rows)")

    # ---------------------------------------------------------------------------
    # Heatmap: net Sharpe
    # ---------------------------------------------------------------------------
    val_years = sorted(df["val_year"].unique())
    pivot = df.pivot_table(index="sleeve", columns="val_year", values="net_sharpe", aggfunc="first")
    pivot = pivot.reindex(columns=val_years)

    # Sort by mean net Sharpe across val years descending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    bound = max(
        abs(np.nanpercentile(pivot.values.astype(float), 5)),
        abs(np.nanpercentile(pivot.values.astype(float), 95)),
    )
    bound = max(bound, 0.5)

    n_sleeves = len(pivot)
    n_years = len(val_years)
    # Wide enough that each year column is ~2.5 inches; tall enough for all sleeves
    fig_w = max(12, n_years * 2.5 + 6)
    fig_h = max(10, n_sleeves * 0.45 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        pivot.values.astype(float),
        aspect="auto",
        cmap="RdYlGn",
        vmin=-bound,
        vmax=bound,
    )
    plt.colorbar(im, ax=ax, label="Net Sharpe (OOS)", fraction=0.03, pad=0.02)

    ax.set_xticks(range(n_years))
    ax.set_xticklabels(val_years, fontsize=12)
    ax.set_yticks(range(n_sleeves))
    ax.set_yticklabels(pivot.index, fontsize=8, rotation=30, ha="right")
    ax.set_title(
        f"Sleeve OOS Net Sharpe — Walkforward ({val_years[0]}–{val_years[-1]})\n"
        f"Training: expanding window, {COST_BPS:.0f} bps costs",
        pad=12,
        fontsize=12,
    )

    for i in range(n_sleeves):
        for j in range(n_years):
            val = pivot.values[i, j]
            if not np.isnan(val):
                brightness = (val - (-bound)) / (2 * bound) if bound > 0 else 0.5
                color = "white" if brightness < 0.25 or brightness > 0.75 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "sleeve_walkforward_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved sleeve_walkforward_sharpe.png")

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    summary = (
        df.groupby("sleeve")["net_sharpe"]
        .mean()
        .sort_values(ascending=False)
        .rename("avg_oos_net_sharpe")
    )
    print("\nSleeves ranked by avg OOS net Sharpe:")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nDone.")


if __name__ == "__main__":
    main()
