"""Train the final logistic regime model from `qc/spxw_1dte_baseline/research.ipynb`.

This script keeps the notebook extraction intentionally simple:

1. Load SPX, VIX, and VIX9D daily CSVs.
2. Recreate the four final features from the notebook.
3. Fit a logistic regression on the training window.
4. Print a pasteable `LOGIT_INTERCEPT` / `LOGIT_WEIGHTS` block.

Run:

```bash
cat <<'CMD'
uv run python -m src.qc_utils.regime_forecast_model
CMD
```
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Daily input files. These are the only files the script needs.
SPX_PATH = "notebooks/data/SPX_daily_2022-2025.csv"
VIX_PATH = "notebooks/data/VIX_daily_2022-2025.csv"
VIX9D_PATH = "notebooks/data/VIX9D_daily_2022-2025.csv"

# The notebook uses this threshold to define a "high range" day.
RANGE_THRESHOLD = 51.698

# Same train/test split used in the notebook.
TRAIN_START = "2022-04-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Final feature set selected in the notebook.
FEATURE_COLUMNS = [
    "prior_slope",
    "5d_avg_range",
    "prior_abs_ret",
    "gap_mag",
]


def load_daily_csv(path: str) -> pd.DataFrame:
    """Load a daily CSV and normalize the date column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_feature_table(
    spx_daily: pd.DataFrame,
    vix_daily: pd.DataFrame,
    vix9d_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Recreate the feature engineering from the notebook."""
    spx = spx_daily.copy()
    vix = vix_daily.copy()
    vix9d = vix9d_daily.copy()

    # SPX realized-volatility features.
    spx["range"] = spx["high"] - spx["low"]
    spx["ret"] = np.log(spx["close"] / spx["close"].shift(1))
    spx["5d_avg_range"] = spx["range"].shift(1).rolling(5).mean()
    spx["prior_abs_ret"] = spx["ret"].shift(1).abs()
    spx["gap_mag"] = ((spx["open"] - spx["close"].shift(1)) / spx["close"].shift(1)).abs()

    # We score the decision on day t using data known by day t, so the target is the
    # regime on day t+1 rather than the same-day range.
    spx["regime_target"] = spx["range"].shift(-1) >= RANGE_THRESHOLD

    # VIX term-structure feature.
    vix["prior_vix_close"] = vix["close"].shift(1)
    vix9d["prior_vix9d_close"] = vix9d["close"].shift(1)

    vix_features = vix[["date", "prior_vix_close"]].merge(
        vix9d[["date", "prior_vix9d_close"]],
        on="date",
        how="left",
    )
    vix_features["prior_slope"] = (
        vix_features["prior_vix9d_close"] - vix_features["prior_vix_close"]
    )

    features = spx[
        [
            "date",
            "regime_target",
            "5d_avg_range",
            "prior_abs_ret",
            "gap_mag",
        ]
    ].merge(
        vix_features[["date", "prior_slope"]],
        on="date",
        how="left",
    )

    # Drop rows that do not have enough history to compute all lagged features.
    features = features.dropna().set_index("date")
    return features


def fit_model(features: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Fit the logistic model and convert it back to raw-feature coefficients."""
    train = features.loc[TRAIN_START:TRAIN_END].copy()
    test = features.loc[TEST_START:TEST_END].copy()

    # Keep the test slice in the script so the date split stays explicit, even though
    # this file's main job is coefficient generation.
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset.")

    x_train = train[FEATURE_COLUMNS]
    y_train = train["regime_target"].astype(int)

    # The notebook standardized inputs before fitting, so we do the same here.
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    model.fit(x_train_scaled, y_train)

    # sklearn fits on standardized inputs. Convert the fitted model back into a raw
    # intercept + raw weights form so another notebook can score features directly.
    scaled_weights = model.coef_[0]
    means = scaler.mean_
    scales = scaler.scale_

    raw_weights = scaled_weights / scales
    raw_intercept = float(model.intercept_[0] - np.sum(scaled_weights * means / scales))

    weights = {
        name: float(weight) for name, weight in zip(FEATURE_COLUMNS, raw_weights, strict=True)
    }
    return raw_intercept, weights


def main() -> None:
    spx_daily = load_daily_csv(SPX_PATH)
    vix_daily = load_daily_csv(VIX_PATH)
    vix9d_daily = load_daily_csv(VIX9D_PATH)

    features = build_feature_table(spx_daily, vix_daily, vix9d_daily)
    intercept, weights = fit_model(features)

    # Print a block that can be pasted directly into another notebook or script.
    print(f"LOGIT_INTERCEPT = {intercept}")
    print("LOGIT_WEIGHTS = {")
    for name in FEATURE_COLUMNS:
        print(f'    "{name}": {weights[name]},')
    print("}")


if __name__ == "__main__":
    main()
