# How Each Chart Uses Gamma

This document explains the gamma calculations used by each GEX chart in `src/trade_lab/charts/`.

---

## General GEX **Calculation**

Most charts use the `row_gross_gex()` function from `src/trade_lab/utils/gex.py`:

```
GEX = gamma × open_interest × spot² × multiplier × gamma_scale
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | — | Option's gamma from data |
| `open_interest` | — | Number of contracts |
| `spot` | — | Current underlying price |
| `multiplier` | 100 | Contract multiplier (SPX = 100) |
| `gamma_scale` | 0.01 | Unit scaling factor |

The `spot²` term converts gamma (per $1 move) into dollar-weighted exposure. This is "gross" GEX—dealer-agnostic and unsigned. For directional (net) GEX, charts apply a sign convention: calls positive, puts negative.

---

## DirectionalGammaImbalance

**Purpose**: Measures the imbalance of gamma exposure above vs below the current spot price to indicate directional bias.

**Calculation**:
1. Filter strikes within `± strike_width` of spot
2. Compute gross GEX per row: `row_gross_gex(df, spot, multiplier, gamma_scale)`
3. Sum gamma above spot (`gamma_above`) and below spot (`gamma_below`)
4. DGI = `(gamma_above - gamma_below) / (|gamma_above| + |gamma_below|)`

The result is normalized to [-1, +1]. Positive values indicate more gamma mass below spot (bullish hedging pressure); negative values indicate more gamma mass above spot (bearish pressure).

---

## GrossGEX

**Purpose**: Tracks total gross gamma exposure over time to identify market regime (mean reversion vs breakout).

**Calculation**:
1. Filter strikes within ±`strike_width` of spot
2. Compute gross GEX per row: `row_gross_gex(df, spot, multiplier, gamma_scale)`
3. Sum all GEX values: `gross_gex = sum(row_gross_gex)`

This is dealer-agnostic (unsigned) gamma. Higher values suggest stronger mean-reversion pressure; lower values suggest potential for larger moves.

---

## GEXPrice

**Purpose**: Shows net gamma exposure across a range of hypothetical prices to identify the Zero Gamma Level (ZGL).

**Calculation**:
1. For each price `p` in a grid (spot ± 300):
   - Compute Black-Scholes gamma: `bs_gamma(s=p, k=strike, t=T, sigma=iv)`
   - GEX per contract: `gamma * OI * p²`
   - Net GEX: `sum(calls) - sum(puts)`
2. Find ZGL via linear interpolation where net GEX crosses zero

The chart reveals where gamma flips sign. Above ZGL, dealers hedge by buying dips; below ZGL, they sell into weakness.

---

## GEXStrike

**Purpose**: Displays net gamma exposure aggregated by strike price as a bar chart.

**Calculation**:
1. Use pre-computed `gamma` column from option chain data
2. GEX per contract: `gamma * OI * spot²`
3. Aggregate by strike: `net_gex = sum(call_gex) - sum(put_gex)`
4. Filter to strikes within ±300 of spot

This shows which strikes have the highest gamma concentration, highlighting key support/resistance levels where dealer hedging is most active.

---

## Summary of Gamma Formulas

| Chart | Gamma Source | GEX Formula | Aggregation |
|-------|--------------|-------------|-------------|
| DirectionalGammaImbalance | `row_gross_gex()` | gross GEX | above vs below spot ratio |
| GrossGEX | `row_gross_gex()` | gross GEX | sum within strike window |
| GEXPrice | `bs_gamma()` | `gamma × OI × S²` | calls − puts per price level |
| GEXStrike | data column | `gamma × OI × S²` | calls − puts per strike |
