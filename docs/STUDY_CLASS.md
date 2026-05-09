# Add Study Class to `qstudy`

I want to add a `Study` class to the src/qstudy library. This new class will be an abstraction that enables users to construct pipelines to quickly do unconstrainted backtests. It basically should bring together the functionality already exposed by the `qstudy` lib under one abstraction.

- Should work like a pipeline passing the outputs of one function as inputs to the next function.
  - This will generally be a signal dataframe and a positions dataframe that are manipulated before getting the final returns of the study.
- Should handle custom constraints/filters
- It should cache data, results and intermediate steps on the study object
- Should have methods for writing study returns to a csv file
- It should have methods of pickling a study after it has been run so that it can be saved and reloaded later.

Basically the pipeline should work like this: base signal -> signal processing -> build positions -> positions processing -> returns -> report
- underneath the hood when the study object gets created it should call the `qs.download(SP500, start_date, end_date)` and cache the universe on the object.
- Optionally the residualize_returns function can be called before creating the base signal. We can only call residualize if a benchmark or factors are passed to the study. Otherwise we should raise error if residualize_returns is called and there's no benchmark or factors. If both benchmark and factors are passed, we should use factors. If only benchmark is passed, then use the benchmark
- Then a base signal method must be called. Right now we will have 2 built in base signal generators: `mean_reversion` and `momentum`. And then we will have a function for passing in a custom base signal: `base_signal()`
- Then a chain of zero or more signal processing and filtering methods can be called. Basically there are two types of here that I can think of right now:
  - First, filters can be applied to the base signal such as the existing `liquidity_filter`, `vol_filter`, `volume_zscore_filter`
  - Second, mutations can be applied to the base signal such as `signal = signal.sub(signal.mean(axis=1), axis=0)`
  - Some these easy filters we can make available as pipeline functions, e.g. `liquidity_filter`
  - But then for many of the signal filters we just need a way to pass a custom function in order to process the signal
- After signal processing, we apply the signals to the returns / residualized_returns dataframe and then build the positions. The build positions methods should both create the positions dataframe and the raw_returns dataframe and store it in the cache.
- Then we need to do any processing of the positions / returns. basically this part of hte pipeline should be a series of zero or more manipulations of the positinos dataframe. Again we should be able to pass in custom functions to handle this scaling.
- We should have some built in weighting schemes that make it easy to set the positions. These are described below in the weighting examples. Basically we'll have 4 default schemes:
  - base equal dollar weights
  - equal volatility weights
  - equal sharpe weights
  - optimal weights
- If `.run()`  is called immediately after the build positions is called, then we should just default to equal weights
- After signal processing and position processing we can generate the returns when the `.run()` is called on the pipeline. This should be the very last method called and it should be the method that kicks off the engine to run through the whole pipeline and generate the results of the study.
  - If it's easy let's have tqdm show the progress as the engine steps through the pipeline.
- After the Study has been run, then we can call `.report()` to generate and show the charts and summary given `qs.metrics.summary` and `qs.summary_plot`

## Examples

### Pipeline Examples

```python
from qstudy import Study
from qstudy.constants import SP500
import qstudy as qs

benchmark = qs.download("SPY", start_date, end_date) # note that qs.download returns a dictionary and not a dataframe
factors = qs.download(["SPY", "XLK"], start_date, end_date)
study = Study(universe=SP500, benchmark=benchmark, factors=factors)

def custom_filter(signal, **cache):
    # NOTE: should only mutate the signal and the Study object should be smart enough
    # to take the retuend signal dataframe and update it in the cache. Somehow it
    # would be good to raise an error if this function attempts to mutate the cache
    # directly.
    residuals_df = cache['residual_returns']
    med_mom = residuals_df.rolling(60).mean()
    signal = signal.where(
        med_mom.abs().lt(
            med_mom.quantile(0.7, axis=1),
            axis=0
        )
    )
    return signal

def custom_return_scale(positions, **cache):
    # NOTE: should only mutate the positions and the Study object should be smart enough
    # to take the returned positions dataframe and update it in the cache. Somehow it
    # would be good to raise an error if this function attempts to mutate the cache
    # directly.
    raw_returns = cache['returns'] # this should always be the latest state of the returns.
    # NOTE: there should also be a history of the returns mutations in the cache. Something like
    # cache['returns_mutations'] or maybe there's a better name but this should contain an
    # array that has each state of the returns in order.
    scale = # Do something to scale the positions
    scaled_positions = positions.mul(
        scale.shift(1),
        axis=0
    )
    return scaled_positions

study
  .residualize_returns()
  .mean_reversion(window=5)
  .add_filter(custom_filter)
  .add_liquidity_filter(top_n=250)
  .build_long_short(n_long=25, n_long=25)
  .scale_returns(custom_return_scale)
  .run()

study.report()
```

```python
from qstudy import Study
from qstudy.constants import SP500
import qstudy as qs

benchmark = qs.download("SPY", start_date, end_date)

Study(universe=SP500, benchmark=benchmark['returns'])
  .momentum(window=90)
  .add_liquidity_filter(top_n=250)
  .build_long_only(n=10)
  .run()
  .report()
```


```python
from qstudy import Study
from qstudy.constants import SP500
import qstudy as qs

benchmark = qs.download("SPY", start_date, end_date)

def base_signal(**cache):
    ret = cache['returns'] # or cache['residual_returns'] if residualize_returns was called.

Study(universe=SP500, benchmark=benchmark['returns'])
  .base_signal(base_signal)
  .add_liquidity_filter(top_n=250)
  .build_long_only(n=10)
  .returns()
  .summary()
```

### Weighting Examples

These are examples from my quant class. They are just meant as a guide and not intended to be directly copied into the qstudy the library. The main thing totake away here is that we want these different weighting strategies:
- base equal dollar weights
- equal volatility weights
- equal sharpe weights
- optimal weights

```python
def optimal_weights(sigma,mu):
    wgt = np.linalg.inv(sigma) @ mu
    wgt = wgt / np.abs(wgt).sum()
    return wgt

def eqvol_weights(sigma):
    wgt = 1/np.sqrt(np.diag(sigma))
    wgt = wgt / np.abs(wgt).sum()
    return wgt

def sr_weights(sigma,mu):
    wgt = mu / np.diag(sigma)
    wgt = wgt / np.abs(wgt).sum()
    return wgt

def gen_strat_returns():
    np.random.seed(5)

    corr = [[1, 0.3, 0],
            [0.3, 1, 0],
            [0,   0, 1]]

    corr = np.array(corr)

    vols = np.diag(np.array([0.1, 0.06, 0.02])) / np.sqrt(252)

    sigma = vols @ corr @ vols

    mu = np.array([0.1,0.12,0.04]) / 252

    dates = pd.date_range('20100101','20191231',freq='B')

    rets = np.random.multivariate_normal(mu, sigma, size = len(dates))
    rets = pd.DataFrame(rets,columns = ['A','B','C'], index = dates)
    return rets

rets = gen_strat_returns()

sigma = rets.cov()
corr = rets.corr()
mu = rets.mean()

weights = {}
weights['opt'] = optimal_weights(sigma,mu)
weights['eqvol'] = eqvol_weights(sigma)
weights['sr'] = sr_weights(sigma,mu)
weights = pd.DataFrame(weights)
weights.round(2)

combo_rets={}
combo_rets['opt'] = (rets*weights['opt']).sum(1)
combo_rets['eqvol'] = (rets*weights['eqvol']).sum(1)
combo_rets['sr'] = (rets*weights['sr']).sum(1)
combo_rets = pd.DataFrame(combo_rets)
combo_sr = combo_rets.mean() / combo_rets.std() * np.sqrt(252)
combo_sr
```

## Future Development

### Portofolio weighting

After we get this Study class working we will want to be able to combine different studies in order to test portfolios of multiple strategies. So we will want to be able to take Study1, Study2, Study3 and wrap them together in portfolio and then test different weighting schemes to see which produces the highest sharpes. So maybe something like we can have a study take an array of studies and work on them or maybe we need a new class like `MultiStrategyStudy`
```python
study = Study(studies=[study1, Study2, Study3], benchmark=...)

mss = MultiStrategyStudy(studies=[study1, Study2, Study3], benchmark=...)
```

### Other Base signals.

We will eventually want to add other base signals:
- event driven models
- time series models
- cross sectional models