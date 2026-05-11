from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


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


class BarraLiteFactorModel:
    """Daily cross-sectional factor model for stock-level risk decomposition.

    Unlike :func:`residualize` which runs one time-series regression per ticker,
    this model runs one regression per *day* across all tickers. Each day:

        r_i,t = alpha_t + beta_i,t * r_mkt,t + sector_dummies_i + [optional factors] + eps_i,t

    The residuals eps_i,t are the true idiosyncratic returns. Factor exposures are
    pre-computed from rolling windows and stored for downstream use (signal neutralization,
    portfolio exposure control).

    Supported factors:
        - ``"market"``:    Rolling beta vs benchmark (always computed, included by default).
        - ``"sector"``:    GICS sector one-hot dummies (requires ``sector_map``).
        - ``"momentum"``:  Rolling mean return cross-sectionally standardized.
        - ``"volatility"``: Rolling return std cross-sectionally standardized.
        - ``"size"``:      Log close price cross-sectionally standardized.

    Example::

        model = BarraLiteFactorModel(
            factors=["market", "sector", "momentum", "volatility"],
            sector_map=sector_map,
            beta_window=60,
        )
        model.fit(returns, benchmark_returns, close)
        residuals, daily_r2 = model.residualize(returns)
    """

    def __init__(
        self,
        factors: list[str] = ("market", "sector"),
        beta_window: int = 60,
        momentum_window: int = 20,
        vol_window: int = 20,
        sector_map: dict[str, str] | None = None,
        min_stocks: int = 30,
    ) -> None:
        self.factors = list(factors)
        self.beta_window = beta_window
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.sector_map = sector_map
        self.min_stocks = min_stocks

        # Set after fit()
        self.factor_exposures_: dict[str, pd.DataFrame] = {}
        self._sector_dummies: pd.DataFrame | None = None
        self._sector_cols: list[str] = []

    def fit(
        self,
        returns: pd.DataFrame,
        benchmark_returns: pd.Series,
        close: pd.DataFrame,
    ) -> BarraLiteFactorModel:
        """Pre-compute all rolling factor exposures.

        Must be called before :meth:`residualize` or :meth:`exposures_on`.

        Args:
            returns:           Daily returns (dates x tickers).
            benchmark_returns: Market returns Series (dates,), e.g. SPY.
            close:             Adjusted close prices (dates x tickers).

        Returns:
            self (for chaining).
        """
        bench = benchmark_returns.reindex(returns.index).fillna(0.0)

        # Market beta: rolling cov(r_i, r_mkt) / var(r_mkt), shifted 1 day (no lookahead)
        mean_r = returns.rolling(self.beta_window).mean()
        mean_b = bench.rolling(self.beta_window).mean()
        mean_rb = returns.mul(bench, axis=0).rolling(self.beta_window).mean()
        cov = mean_rb.sub(mean_r.mul(mean_b, axis=0))
        var_b = bench.rolling(self.beta_window).var().replace(0.0, np.nan)
        beta = cov.div(var_b, axis=0).shift(1)
        self.factor_exposures_["market"] = beta

        if "momentum" in self.factors:
            self.factor_exposures_["momentum"] = (
                returns.rolling(self.momentum_window).mean().shift(1)
            )

        if "volatility" in self.factors:
            self.factor_exposures_["volatility"] = returns.rolling(self.vol_window).std().shift(1)

        if "size" in self.factors:
            self.factor_exposures_["size"] = np.log(close.replace(0.0, np.nan)).shift(1)

        # Sector dummies (static — time-invariant snapshot)
        if "sector" in self.factors and self.sector_map is not None:
            sector_series = pd.Series(self.sector_map).reindex(returns.columns).fillna("Unknown")
            dummies = pd.get_dummies(sector_series, prefix="sector", dtype=float)
            # Drop most-populated sector as reference category
            ref_col = dummies.sum().idxmax()
            dummies = dummies.drop(columns=[ref_col])
            self._sector_dummies = dummies
            self._sector_cols = dummies.columns.tolist()

        return self

    def residualize(self, returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Run daily cross-sectional OLS and return idiosyncratic residuals.

        Args:
            returns: Daily returns (dates x tickers). Must cover the same tickers
                     as used in :meth:`fit`.

        Returns:
            residuals_df: DataFrame of idiosyncratic residuals, same shape as returns.
            daily_r2_s:   Series of cross-sectional R² per date (diagnostic).
        """
        if not self.factor_exposures_:
            raise RuntimeError("Call fit() before residualize().")

        residuals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
        daily_r2: dict[pd.Timestamp, float] = {}

        continuous_factors = [f for f in self.factors if f != "sector"]

        for date in returns.index:
            r_t = returns.loc[date].dropna()
            if len(r_t) < self.min_stocks:
                continue

            cols = r_t.index

            # Assemble exposure matrix for this date
            parts: list[pd.Series | pd.DataFrame] = []
            for fname in continuous_factors:
                if fname in self.factor_exposures_:
                    parts.append(self.factor_exposures_[fname].loc[date, cols].rename(fname))

            if self._sector_dummies is not None:
                parts.append(self._sector_dummies.reindex(cols).fillna(0.0))

            if not parts:
                continue

            X_t = pd.concat(parts, axis=1).reindex(cols)
            # Drop rows where any continuous factor is NaN
            valid = X_t[continuous_factors].notna().all(axis=1) if continuous_factors else X_t.index
            X_t = X_t.loc[valid]
            y_t = r_t.reindex(X_t.index).dropna()
            X_t = X_t.reindex(y_t.index)

            if len(y_t) < self.min_stocks:
                continue

            # Cross-sectionally standardize continuous factors (not sector dummies)
            if continuous_factors:
                cont_cols = [c for c in continuous_factors if c in X_t.columns]
                mu = X_t[cont_cols].mean()
                sigma = X_t[cont_cols].std().replace(0.0, np.nan)
                X_t = X_t.copy()
                X_t[cont_cols] = (X_t[cont_cols] - mu) / sigma

            X_t = X_t.fillna(0.0)

            model = LinearRegression(fit_intercept=True)
            model.fit(X_t.values, y_t.values)
            residuals.loc[date, y_t.index] = y_t.values - model.predict(X_t.values)
            daily_r2[date] = float(model.score(X_t.values, y_t.values))

        daily_r2_s = pd.Series(daily_r2, name="cross_sectional_r2")
        daily_r2_s.index.name = "date"
        return residuals, daily_r2_s

    def exposures_on(self, date: pd.Timestamp) -> pd.DataFrame:
        """Return the exposure matrix for a single date (tickers x factors).

        Args:
            date: The date to retrieve exposures for.

        Returns:
            DataFrame of shape (tickers x factors). Continuous factors are as-computed
            (not standardized); sector columns are 0/1 dummies.
        """
        if not self.factor_exposures_:
            raise RuntimeError("Call fit() before exposures_on().")

        tickers = next(iter(self.factor_exposures_.values())).columns
        parts: list[pd.Series | pd.DataFrame] = []
        for fname, df in self.factor_exposures_.items():
            parts.append(df.loc[date].rename(fname))
        if self._sector_dummies is not None:
            parts.append(self._sector_dummies.reindex(tickers).fillna(0.0))

        return pd.concat(parts, axis=1).reindex(tickers)


def cross_sectional_residualize(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    close: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    beta_window: int = 60,
    include_momentum: bool = False,
    momentum_window: int = 20,
    include_vol: bool = False,
    vol_window: int = 20,
    include_size: bool = False,
    min_stocks: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run daily cross-sectional OLS to residualize returns against Barra-style factor exposures.

    Thin functional wrapper around :class:`BarraLiteFactorModel`. See that class for full details.

    Args:
        returns:           Daily returns (dates x tickers).
        benchmark_returns: Market returns Series (dates,), e.g. SPY.
        close:             Adjusted close prices (dates x tickers).
        sector_map:        Dict of ticker -> GICS sector string. When None, sector dummies
                           are omitted and only market beta is used.
        beta_window:       Rolling window for beta estimation (days).
        include_momentum:  If True, add rolling momentum as a factor.
        momentum_window:   Lookback for momentum exposure.
        include_vol:       If True, add realized vol as a factor.
        vol_window:        Lookback for vol exposure.
        include_size:      If True, add log close price as a size proxy.
        min_stocks:        Minimum stocks with valid data required to run regression on a date.

    Returns:
        residuals_df: DataFrame of daily idiosyncratic residuals, same shape as returns.
        daily_r2_s:   Series of cross-sectional R² per date (diagnostic).
    """
    factors: list[str] = ["market"]
    if sector_map is not None:
        factors.append("sector")
    if include_momentum:
        factors.append("momentum")
    if include_vol:
        factors.append("volatility")
    if include_size:
        factors.append("size")

    model = BarraLiteFactorModel(
        factors=factors,
        beta_window=beta_window,
        momentum_window=momentum_window,
        vol_window=vol_window,
        sector_map=sector_map,
        min_stocks=min_stocks,
    )
    model.fit(returns, benchmark_returns, close)
    return model.residualize(returns)
