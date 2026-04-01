import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


REGIME_THRESHOLD = 51.698
SKIP_PROB_THRESHOLD = 0.50


def add_actual_next_day_regime(trade_features: pd.DataFrame, spx_day: pd.DataFrame) -> pd.DataFrame:
    spx = spx_day.copy().sort_values("date").reset_index(drop=True)
    spx["range"] = spx["high"] - spx["low"]
    spx["next_day_range"] = spx["range"].shift(-1)
    spx["actual_high_regime"] = spx["next_day_range"] >= REGIME_THRESHOLD

    actuals = spx[["date", "next_day_range", "actual_high_regime"]]
    return trade_features.merge(actuals, on="date", how="left")


def compute_confusion_outputs(
    trade_features_with_actuals: pd.DataFrame,
    prob_threshold: float = SKIP_PROB_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_df = trade_features_with_actuals.dropna(
        subset=["regime_prob", "actual_high_regime"]
    ).copy()

    y_true = eval_df["actual_high_regime"].astype(int)
    y_pred = (eval_df["regime_prob"] >= prob_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Low", "Actual High"],
        columns=["Pred Low", "Pred High"],
    )

    print(cm_df)
    print()
    print(classification_report(y_true, y_pred, target_names=["low_range", "high_range"]))

    eval_df["pred_high_regime"] = y_pred.astype(bool)
    return eval_df, cm_df


def plot_confusion_matrix(
    trade_features_with_actuals: pd.DataFrame,
    prob_threshold: float = SKIP_PROB_THRESHOLD,
) -> None:
    eval_df = trade_features_with_actuals.dropna(
        subset=["regime_prob", "actual_high_regime"]
    ).copy()

    y_true = eval_df["actual_high_regime"].astype(int)
    y_pred = (eval_df["regime_prob"] >= prob_threshold).astype(int)

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=["Low", "High"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix @ threshold={prob_threshold:.2f}")
    plt.tight_layout()
    plt.show()


def get_false_positives(eval_df: pd.DataFrame) -> pd.DataFrame:
    return eval_df[(eval_df["pred_high_regime"]) & (~eval_df["actual_high_regime"])].copy()


"""
Notebook usage example
----------------------

# `trade_features` should come from your regime-filter backtest workflow and already contain:
# - date
# - regime_prob
# - pnl
# - total_pnl
#
# `spx_day` should contain:
# - date
# - high
# - low

trade_features_with_actuals = add_actual_next_day_regime(trade_features, spx_day)
eval_df, cm_df = compute_confusion_outputs(
    trade_features_with_actuals,
    prob_threshold=0.50,
)
plot_confusion_matrix(trade_features_with_actuals, prob_threshold=0.50)

false_positives = get_false_positives(eval_df)
false_positives[["date", "regime_prob", "next_day_range", "pnl", "total_pnl"]].head(20)
"""
