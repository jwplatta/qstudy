from __future__ import annotations

import itertools
from collections.abc import Callable

import pandas as pd
from tqdm import tqdm

from qstudy.backtest import metrics as _metrics


def param_grid(
    param_dict: dict[str, list],
    backtest_fn: Callable[[dict], pd.Series],
    metric_fn: Callable[[pd.Series], float | pd.Series] | None = None,
) -> pd.DataFrame:
    """Run a parameter sweep over all combinations in param_dict.

    Args:
        param_dict:  Dict of param_name -> list of values to sweep.
                     e.g. {"window": [60, 90, 120], "quantile": [0.7, 0.8, 0.9]}
        backtest_fn: Callable that accepts a dict of params and returns a daily returns Series.
                     The caller is responsible for closing over data (close_df, returns_df, etc.).
        metric_fn:   Optional. If None, calls metrics.summary(). Otherwise called with the returns
                     Series and must return a scalar or a named Series.

    Returns:
        results_df: DataFrame with one row per parameter combination.
                    Columns are the param names plus all metric keys.
    """
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    combos = list(itertools.product(*values))
    rows = []

    for combo in tqdm(combos, desc="param_grid"):
        params = dict(zip(keys, combo))
        port_ret = backtest_fn(params)
        if metric_fn is not None:
            result = metric_fn(port_ret)
        else:
            result = _metrics.summary(port_ret)

        if isinstance(result, pd.Series):
            row = {**params, **result.to_dict()}
        else:
            row = {**params, "metric": result}
        rows.append(row)

    return pd.DataFrame(rows)
