from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

"""
Run with:

cat <<'CMD'
uv run python src/trade_lab/scripts/regime_forecast_tests.py
CMD
"""


INITIAL_CAPITAL = 50_000
REGIME_THRESHOLD = 51.698
SKIP_PROB_THRESHOLD = 0.45
LOW_RISK_THRESH = 0.0
LOW_RISK_CONTRACT_MULTIPLIER = 1.0
LOOKBACK_CALENDAR_DAYS = 14
THRESHOLD_GRID = [0.2, 0.4, 0.6, 0.8]
PROBABILITY_BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MAX_LOSS_THRESHOLD = -1000
GAP_MAG_SHOCK_THRESHOLD = 0.005
PRIOR_ABS_RET_SHOCK_THRESHOLD = 0.005
AVG_RANGE_5D_SHOCK_THRESHOLD = 45.0

LOGIT_INTERCEPT = -3.9225339071705148
LOGIT_WEIGHTS = {
    "prior_slope": 0.15572728210666714,
    "5d_avg_range": 0.06168979627510342,
    "prior_abs_ret": 36.499736265948194,
    "gap_mag": 40.380107779126185,
}

DATA_DIR = Path("research/data")
TRADES_PATH = DATA_DIR / "baseline_v1_trades_02_13_2026.csv"
SPX_PATH = DATA_DIR / "SPX_day_1980-01-01_2026-03-24.csv"
VIX_PATH = DATA_DIR / "VIX_day_1980-01-01_2026-03-24.csv"
VIX9D_PATH = DATA_DIR / "VIX9D_day_2000-01-01_2026-03-24.csv"
OUTPUT_DIR = Path("tmp")
OUTPUT_DIR.mkdir(exist_ok=True)

trades = pd.read_csv(TRADES_PATH, header=0, skipinitialspace=True)
if "Order Ids" in trades.columns:
    trades = trades.drop(columns=["Order Ids"])

trades["Entry Time"] = pd.to_datetime(trades["Entry Time"], utc=True)
trades["Exit Time"] = pd.to_datetime(trades["Exit Time"], utc=True)
trades = trades[trades["Entry Time"] >= pd.Timestamp("2022-04-01", tz="UTC")].reset_index(drop=True)
trades["Entry Time"] = trades["Entry Time"].dt.tz_convert("America/New_York")
trades["Exit Time"] = trades["Exit Time"].dt.tz_convert("America/New_York")

trades = (
    trades.sort_values("Entry Time")
    .groupby(["Entry Time", "Exit Time"], as_index=False)
    .agg(pnl=("P&L", "sum"), fees=("Fees", "sum"))
    .rename(columns={"Entry Time": "entry_time", "Exit Time": "exit_time"})
)
trades["total_pnl"] = trades["pnl"] - trades["fees"]
trades["date"] = trades["entry_time"].dt.date

spx_day = pd.read_csv(SPX_PATH)
vix_day = pd.read_csv(VIX_PATH)
vix9d_day = pd.read_csv(VIX9D_PATH)

spx_day["date"] = pd.to_datetime(spx_day["datetime"]).dt.date
vix_day["date"] = pd.to_datetime(vix_day["datetime"]).dt.date
vix9d_day["date"] = pd.to_datetime(vix9d_day["datetime"]).dt.date

spx_day = spx_day.sort_values("date").reset_index(drop=True)
vix_day = vix_day.sort_values("date").reset_index(drop=True)
vix9d_day = vix9d_day.sort_values("date").reset_index(drop=True)

start_date = trades["date"].min() - timedelta(days=LOOKBACK_CALENDAR_DAYS)
end_date = trades["date"].max()

spx_day = spx_day[(spx_day["date"] >= start_date) & (spx_day["date"] <= end_date)].copy()
vix_day = vix_day[(vix_day["date"] >= start_date) & (vix_day["date"] <= end_date)].copy()
vix9d_day = vix9d_day[(vix9d_day["date"] >= start_date) & (vix9d_day["date"] <= end_date)].copy()

spx_day["range"] = spx_day["high"] - spx_day["low"]
spx_day["ret"] = np.log(spx_day["close"] / spx_day["close"].shift(1))
spx_day["5d_avg_range"] = spx_day["range"].shift(1).rolling(5).mean()
spx_day["prior_abs_ret"] = spx_day["ret"].shift(1).abs()
spx_day["gap_mag"] = (
    (spx_day["open"] - spx_day["close"].shift(1)) / spx_day["close"].shift(1)
).abs()
spx_day["next_day_range"] = spx_day["range"].shift(-1)
spx_day["actual_high_regime"] = spx_day["next_day_range"] >= REGIME_THRESHOLD

vix_day["prior_vix_close"] = vix_day["close"].shift(1)
vix9d_day["prior_vix9d_close"] = vix9d_day["close"].shift(1)

vix_features = vix_day[["date", "prior_vix_close"]].merge(
    vix9d_day[["date", "prior_vix9d_close"]],
    on="date",
    how="inner",
)
vix_features["prior_slope"] = vix_features["prior_vix9d_close"] - vix_features["prior_vix_close"]

trade_features = trades.merge(
    spx_day[
        [
            "date",
            "range",
            "5d_avg_range",
            "prior_abs_ret",
            "gap_mag",
            "next_day_range",
            "actual_high_regime",
        ]
    ],
    on="date",
    how="left",
)
trade_features = trade_features.merge(vix_features[["date", "prior_slope"]], on="date", how="left")

logit = (
    LOGIT_INTERCEPT
    + LOGIT_WEIGHTS["prior_slope"] * trade_features["prior_slope"]
    + LOGIT_WEIGHTS["5d_avg_range"] * trade_features["5d_avg_range"]
    + LOGIT_WEIGHTS["prior_abs_ret"] * trade_features["prior_abs_ret"]
    + LOGIT_WEIGHTS["gap_mag"] * trade_features["gap_mag"]
)
trade_features["regime_prob"] = 1.0 / (1.0 + np.exp(-logit))
trade_features["skip_trade"] = trade_features["regime_prob"] >= SKIP_PROB_THRESHOLD
trade_features["keep_trade"] = ~trade_features["skip_trade"]

required_columns = [
    "prior_slope",
    "5d_avg_range",
    "prior_abs_ret",
    "gap_mag",
    "regime_prob",
]
valid_trades = trade_features.dropna(subset=required_columns).copy()
filtered_trades = valid_trades.loc[valid_trades["keep_trade"]].copy()
filtered_trades = filtered_trades.sort_values("entry_time").reset_index(drop=True)

filtered_trades["base_pnl"] = filtered_trades["pnl"]
filtered_trades["base_total_pnl"] = filtered_trades["total_pnl"]
filtered_trades["contract_multiplier"] = 1.0
filtered_trades["is_low_risk_day"] = filtered_trades["regime_prob"] <= LOW_RISK_THRESH
filtered_trades.loc[filtered_trades["is_low_risk_day"], "contract_multiplier"] = (
    LOW_RISK_CONTRACT_MULTIPLIER
)
filtered_trades["pnl"] = filtered_trades["base_pnl"] * filtered_trades["contract_multiplier"]
filtered_trades["total_pnl"] = (
    filtered_trades["base_total_pnl"] * filtered_trades["contract_multiplier"]
)

print("All trades:", len(valid_trades))
print("Kept trades:", len(filtered_trades))
print("Skipped trades:", len(valid_trades) - len(filtered_trades))
print("Low-risk scaled trades:", int(filtered_trades["is_low_risk_day"].sum()))
print()
print("Baseline final equity (PnL):", INITIAL_CAPITAL + valid_trades["pnl"].sum())
print("Baseline final equity (Total PnL):", INITIAL_CAPITAL + valid_trades["total_pnl"].sum())
print("Filtered final equity (PnL):", INITIAL_CAPITAL + filtered_trades["pnl"].sum())
print(
    "Filtered final equity (Total PnL):",
    INITIAL_CAPITAL + filtered_trades["total_pnl"].sum(),
)
print()

confusion_eval = valid_trades.dropna(subset=["actual_high_regime"]).copy()
y_true = confusion_eval["actual_high_regime"].astype(int)
y_pred = (confusion_eval["regime_prob"] >= SKIP_PROB_THRESHOLD).astype(int)
confusion_eval["predicted_high_regime"] = y_pred.astype(bool)
confusion_eval["classification"] = np.select(
    [
        (confusion_eval["predicted_high_regime"]) & (confusion_eval["actual_high_regime"]),
        (confusion_eval["predicted_high_regime"]) & (~confusion_eval["actual_high_regime"]),
        (~confusion_eval["predicted_high_regime"]) & (confusion_eval["actual_high_regime"]),
        (~confusion_eval["predicted_high_regime"]) & (~confusion_eval["actual_high_regime"]),
    ],
    ["true_positive", "false_positive", "false_negative", "true_negative"],
    default="unclassified",
)

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Low", "Actual High"],
    columns=["Pred Low", "Pred High"],
)
print(cm_df)
print()
print(classification_report(y_true, y_pred, target_names=["low_range", "high_range"]))

confusion_eval[
    [
        "date",
        "range",
        "5d_avg_range",
        "next_day_range",
        "regime_prob",
        "total_pnl",
        "actual_high_regime",
        "predicted_high_regime",
        "classification",
    ]
].rename(
    columns={
        "range": "trade_day_range",
        "5d_avg_range": "feature_5d_avg_range",
        "next_day_range": "actual_next_day_range",
        "regime_prob": "predicted_probability",
        "total_pnl": "trade_total_pnl",
    }
).to_csv(OUTPUT_DIR / "regime_classification_table.csv", index=False)

bucket_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
bucket_expectancy = valid_trades[["regime_prob", "total_pnl"]].copy()
bucket_expectancy["probability_bucket"] = pd.cut(
    bucket_expectancy["regime_prob"],
    bins=PROBABILITY_BUCKETS,
    labels=bucket_labels,
    include_lowest=True,
    right=True,
)
bucket_expectancy["is_max_loss"] = bucket_expectancy["total_pnl"] <= MAX_LOSS_THRESHOLD
bucket_expectancy["expectancy"] = bucket_expectancy["total_pnl"]

bucket_summary = (
    bucket_expectancy.groupby("probability_bucket", observed=False)
    .agg(
        trade_count=("total_pnl", "size"),
        mean_pnl=("total_pnl", "mean"),
        max_loss_frequency=("is_max_loss", "mean"),
        expectancy=("expectancy", "mean"),
    )
    .reset_index()
)
bucket_summary["max_loss_frequency"] = bucket_summary["max_loss_frequency"].fillna(0.0)
bucket_summary.to_csv(OUTPUT_DIR / "conditional_expectancy_by_probability_bucket.csv", index=False)

plt.figure(figsize=(14, 5))
plt.hist(valid_trades["regime_prob"], bins=30, alpha=0.8, color="tab:blue")
plt.axvline(SKIP_PROB_THRESHOLD, color="red", linestyle="--", label="Skip threshold")
plt.axvline(LOW_RISK_THRESH, color="green", linestyle="--", label="Low-risk threshold")
plt.title("Regime Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Trade Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "regime_probability_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 5))
kept_trades = valid_trades.loc[~valid_trades["skip_trade"]]
skipped_trades = valid_trades.loc[valid_trades["skip_trade"]]
plt.scatter(
    kept_trades["entry_time"],
    kept_trades["regime_prob"],
    s=20,
    alpha=0.7,
    label="Kept",
    color="tab:blue",
)
plt.scatter(
    skipped_trades["entry_time"],
    skipped_trades["regime_prob"],
    s=28,
    alpha=0.85,
    label="Skipped",
    color="tab:red",
)
plt.axhline(SKIP_PROB_THRESHOLD, color="red", linestyle="--", label="Skip threshold")
plt.axhline(LOW_RISK_THRESH, color="green", linestyle="--", label="Low-risk threshold")
plt.title("Predicted Probability by Trade Date")
plt.xlabel("Entry Time")
plt.ylabel("Predicted Probability")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predicted_probability_by_trade_date.png", dpi=150, bbox_inches="tight")
plt.close()

baseline_curve = valid_trades.sort_values("entry_time").reset_index(drop=True).copy()
baseline_curve["equity_pnl"] = INITIAL_CAPITAL + baseline_curve["pnl"].cumsum()
baseline_curve["equity_total_pnl"] = INITIAL_CAPITAL + baseline_curve["total_pnl"].cumsum()
baseline_curve["drawdown_pnl"] = (
    baseline_curve["equity_pnl"] / baseline_curve["equity_pnl"].cummax() - 1.0
)
baseline_curve["drawdown_total_pnl"] = (
    baseline_curve["equity_total_pnl"] / baseline_curve["equity_total_pnl"].cummax() - 1.0
)

filtered_curve = filtered_trades.sort_values("entry_time").reset_index(drop=True).copy()
filtered_curve["equity_pnl"] = INITIAL_CAPITAL + filtered_curve["pnl"].cumsum()
filtered_curve["equity_total_pnl"] = INITIAL_CAPITAL + filtered_curve["total_pnl"].cumsum()
filtered_curve["drawdown_pnl"] = (
    filtered_curve["equity_pnl"] / filtered_curve["equity_pnl"].cummax() - 1.0
)
filtered_curve["drawdown_total_pnl"] = (
    filtered_curve["equity_total_pnl"] / filtered_curve["equity_total_pnl"].cummax() - 1.0
)

baseline_total_returns = baseline_curve["equity_total_pnl"].pct_change().dropna()
filtered_total_returns = filtered_curve["equity_total_pnl"].pct_change().dropna()

baseline_sharpe = np.nan
if not baseline_total_returns.empty and baseline_total_returns.std() > 0:
    baseline_sharpe = np.sqrt(252) * baseline_total_returns.mean() / baseline_total_returns.std()

filtered_sharpe = np.nan
if not filtered_total_returns.empty and filtered_total_returns.std() > 0:
    filtered_sharpe = np.sqrt(252) * filtered_total_returns.mean() / filtered_total_returns.std()

print()
print(f"Baseline total_pnl Sharpe: {baseline_sharpe:.4f}")
print(f"Filtered total_pnl Sharpe: {filtered_sharpe:.4f}")

gap_mag_shock_trades = (
    valid_trades.loc[valid_trades["gap_mag"] <= GAP_MAG_SHOCK_THRESHOLD]
    .sort_values("entry_time")
    .reset_index(drop=True)
)
gap_mag_shock_trades["equity_total_pnl"] = (
    INITIAL_CAPITAL + gap_mag_shock_trades["total_pnl"].cumsum()
)

prior_abs_ret_shock_trades = (
    valid_trades.loc[valid_trades["prior_abs_ret"] <= PRIOR_ABS_RET_SHOCK_THRESHOLD]
    .sort_values("entry_time")
    .reset_index(drop=True)
)
prior_abs_ret_shock_trades["equity_total_pnl"] = (
    INITIAL_CAPITAL + prior_abs_ret_shock_trades["total_pnl"].cumsum()
)

avg_range_5d_shock_trades = (
    valid_trades.loc[valid_trades["5d_avg_range"] <= AVG_RANGE_5D_SHOCK_THRESHOLD]
    .sort_values("entry_time")
    .reset_index(drop=True)
)
avg_range_5d_shock_trades["equity_total_pnl"] = (
    INITIAL_CAPITAL + avg_range_5d_shock_trades["total_pnl"].cumsum()
)

print()
print(
    "Gap shock filter kept trades:",
    len(gap_mag_shock_trades),
    "final equity:",
    gap_mag_shock_trades["equity_total_pnl"].iloc[-1]
    if not gap_mag_shock_trades.empty
    else INITIAL_CAPITAL,
)
print(
    "Prior abs ret shock filter kept trades:",
    len(prior_abs_ret_shock_trades),
    "final equity:",
    (
        prior_abs_ret_shock_trades["equity_total_pnl"].iloc[-1]
        if not prior_abs_ret_shock_trades.empty
        else INITIAL_CAPITAL
    ),
)
print(
    "5d avg range shock filter kept trades:",
    len(avg_range_5d_shock_trades),
    "final equity:",
    (
        avg_range_5d_shock_trades["equity_total_pnl"].iloc[-1]
        if not avg_range_5d_shock_trades.empty
        else INITIAL_CAPITAL
    ),
)

plt.figure(figsize=(14, 6))
plt.plot(
    baseline_curve["entry_time"],
    baseline_curve["equity_pnl"],
    label="Baseline Equity (PnL)",
    alpha=0.35,
    linestyle="--",
    color="tab:blue",
)
plt.plot(
    baseline_curve["entry_time"],
    baseline_curve["equity_total_pnl"],
    label="Baseline Equity (Total PnL incl fees)",
    alpha=0.35,
    linestyle="--",
    color="tab:orange",
)
plt.plot(
    filtered_curve["entry_time"],
    filtered_curve["equity_pnl"],
    label="Filtered Equity (PnL)",
    linewidth=2,
    color="tab:blue",
)
plt.plot(
    filtered_curve["entry_time"],
    filtered_curve["equity_total_pnl"],
    label="Filtered Equity (Total PnL incl fees)",
    linewidth=2,
    color="tab:orange",
)
plt.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", alpha=0.7)
plt.title("Equity Curve")
plt.xlabel("Entry Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
for threshold in THRESHOLD_GRID:
    threshold_trades = valid_trades.loc[valid_trades["regime_prob"] < threshold].copy()
    threshold_trades = threshold_trades.sort_values("entry_time").reset_index(drop=True)

    threshold_trades["base_total_pnl"] = threshold_trades["total_pnl"]
    threshold_trades["contract_multiplier"] = 1.0
    threshold_trades["is_low_risk_day"] = threshold_trades["regime_prob"] <= LOW_RISK_THRESH
    threshold_trades.loc[threshold_trades["is_low_risk_day"], "contract_multiplier"] = (
        LOW_RISK_CONTRACT_MULTIPLIER
    )
    threshold_trades["total_pnl"] = (
        threshold_trades["base_total_pnl"] * threshold_trades["contract_multiplier"]
    )
    threshold_trades["equity_total_pnl"] = INITIAL_CAPITAL + threshold_trades["total_pnl"].cumsum()

    plt.plot(
        threshold_trades["entry_time"],
        threshold_trades["equity_total_pnl"],
        linewidth=2,
        label=f"threshold={threshold:.2f}",
    )

plt.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", alpha=0.7)
plt.title("Total PnL Equity Curve by Threshold")
plt.xlabel("Entry Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "equity_curve_threshold_grid.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(
    baseline_curve["entry_time"],
    baseline_curve["equity_total_pnl"],
    linewidth=2,
    color="black",
    label="Unfiltered",
)
plt.plot(
    gap_mag_shock_trades["entry_time"],
    gap_mag_shock_trades["equity_total_pnl"],
    linewidth=2,
    label=f"gap_mag <= {GAP_MAG_SHOCK_THRESHOLD:.3f}",
)
plt.plot(
    prior_abs_ret_shock_trades["entry_time"],
    prior_abs_ret_shock_trades["equity_total_pnl"],
    linewidth=2,
    label=f"prior_abs_ret <= {PRIOR_ABS_RET_SHOCK_THRESHOLD:.3f}",
)
plt.plot(
    avg_range_5d_shock_trades["entry_time"],
    avg_range_5d_shock_trades["equity_total_pnl"],
    linewidth=2,
    label=f"5d_avg_range <= {AVG_RANGE_5D_SHOCK_THRESHOLD:.1f}",
)
plt.axhline(INITIAL_CAPITAL, color="gray", linestyle=":", alpha=0.7)
plt.title("Shock Filter Equity Curves (Total PnL)")
plt.xlabel("Entry Time")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "shock_filter_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(
    baseline_curve["entry_time"],
    baseline_curve["drawdown_pnl"],
    alpha=0.35,
    linestyle="--",
    color="tab:blue",
    label="Baseline Drawdown (PnL)",
)
plt.plot(
    baseline_curve["entry_time"],
    baseline_curve["drawdown_total_pnl"],
    alpha=0.35,
    linestyle="--",
    color="tab:orange",
    label="Baseline Drawdown (Total PnL incl fees)",
)
plt.plot(
    filtered_curve["entry_time"],
    filtered_curve["drawdown_pnl"],
    linewidth=2,
    color="tab:blue",
    label="Filtered Drawdown (PnL)",
)
plt.plot(
    filtered_curve["entry_time"],
    filtered_curve["drawdown_total_pnl"],
    linewidth=2,
    color="tab:orange",
    label="Filtered Drawdown (Total PnL incl fees)",
)
plt.axhline(0, color="gray", linestyle=":", alpha=0.7)
plt.title("Drawdowns")
plt.xlabel("Entry Time")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "drawdowns.png", dpi=150, bbox_inches="tight")
plt.close()

trade_day_range = valid_trades.sort_values("entry_time").reset_index(drop=True).copy()
plt.figure(figsize=(14, 5))
plt.plot(
    trade_day_range["entry_time"],
    trade_day_range["range"],
    linewidth=1.8,
    color="tab:purple",
    label="Trade-day range",
)
plt.axhline(REGIME_THRESHOLD, color="red", linestyle="--", label="High-range threshold")
plt.title("Trade-Day Range")
plt.xlabel("Entry Time")
plt.ylabel("SPX Daily Range")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "trade_day_range.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low", "High"],
)
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix @ threshold={SKIP_PROB_THRESHOLD:.2f}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved charts to {OUTPUT_DIR.resolve()}")
