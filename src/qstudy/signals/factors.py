import pandas as pd
import statsmodels.api as sm


def residualize(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cross-sectionally residualize each ticker's returns against a factor set via OLS.

    Runs one regression per ticker over the full overlapping sample. The residuals represent the
    idiosyncratic component unexplained by market/sector factors. Useful before mean-reversion
    signals to isolate stock-specific alpha.

    Args:
        returns:        Daily returns (dates x tickers).
        factor_returns: Factor returns (dates x factors), e.g. from load_factors(['SPY', 'XLK']).
                        Index must be compatible with returns.index.

    Returns:
        residuals_df: DataFrame of OLS residuals, same shape as the aligned returns.
        params_df:    DataFrame of regression coefficients (tickers x factors+const).
        rsquared_s:   Series of R-squared values per ticker.
    """
    common_index = returns.index.intersection(factor_returns.index)
    r = returns.loc[common_index]
    f = sm.add_constant(factor_returns.loc[common_index])

    residuals = pd.DataFrame(index=common_index, columns=r.columns, dtype=float)
    params: dict[str, pd.Series] = {}
    rsquared: dict[str, float] = {}

    for ticker in r.columns:
        y = r[ticker].dropna()
        x = f.loc[y.index]
        model = sm.OLS(y, x).fit()
        residuals.loc[y.index, ticker] = model.resid
        params[ticker] = model.params
        rsquared[ticker] = model.rsquared

    params_df = pd.DataFrame(params).T
    params_df.index.name = "ticker"
    rsquared_s = pd.Series(rsquared, name="rsquared")
    rsquared_s.index.name = "ticker"

    return residuals, params_df, rsquared_s
