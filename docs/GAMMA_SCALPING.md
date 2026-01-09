# Gamma Scalping Strategy

## Definition

Gamma scalping is an options trading strategy that continuously adjusts a delta-neutral position to profit from the gamma of an option. The strategy exploits option convexity to capture profits from underlying price fluctuations while maintaining minimal directional exposure.

## Core Mechanism

1. Establish delta-neutral position by combining long options with underlying stock
2. As underlying price moves, delta changes due to gamma exposure
3. Periodically rebalance by trading underlying stock to restore delta neutrality
4. Capture profit from rebalancing trades (buy low after drops, sell high after rises)
5. Rebalancing profit must exceed theta decay on long options positions

## Mathematical Framework

### Black-Scholes Greeks

**d₁ and d₂:**
```
d₁ = [ln(S/K) + (r + 0.5σ²)T] / (σ√T)
d₂ = d₁ - σ√T
```

**Delta:**
```
Δ_call = N(d₁)
Δ_put = N(d₁) - 1
```

**Gamma (same for calls and puts):**
```
Γ = N'(d₁) / (S × σ × √T)
```

Where:
- S = Spot price
- K = Strike price
- r = Risk-free rate
- σ = Implied volatility (derived via Brent's optimization)
- T = Time to expiration (years)
- N = Cumulative normal distribution
- N' = Normal probability density function

### Portfolio Delta
```
Δ_portfolio = Σ(Δ_option × quantity × 100) + stock_shares
```

## Execution

### Position Setup

**Options:**
- Buy call options (typically 3-5 contracts)
- Strikes: 1-5% above current price
- Expiration: 14-60 days (avoid 0DTE)
- ATM options have maximum gamma

**Initial Neutralization:**
- Calculate total portfolio delta
- Offset with underlying stock position to achieve delta ≈ 0

### Rebalancing Logic

**Monitoring:**
- Check portfolio delta at regular intervals (e.g., every 120 seconds)
- Update Greeks using current option prices and underlying price

**Triggers:**
- Set notional delta threshold (e.g., $500)
- If |Δ_portfolio × S| > threshold:
  - Positive delta → sell underlying stock
  - Negative delta → buy underlying stock

**Adjustment:**
```python
shares_to_trade = abs(current_delta)
if current_delta > max_notional_delta / underlying_price:
    action = "sell"
elif current_delta < -max_notional_delta / underlying_price:
    action = "buy"
```

## Profit Dynamics

**When Strategy Profits:**
- Realized volatility > implied volatility
- Frequent price movements enable profitable rebalancing
- Gamma P&L from rebalancing > theta decay

**Key Trade-off:**
- Long options = negative theta (time decay cost)
- Gamma exposure = convexity profit from rebalancing
- Net P&L = Gamma profit - Theta cost - Transaction costs

## Implementation Parameters

**Critical Thresholds:**
- Maximum notional delta: $500 (adjust for portfolio size)
- Rebalancing frequency: 30-120 seconds
- Risk-free rate: Current Treasury rate (e.g., 4.5%)

**Strategy Variations:**
- **Adjustment frequency:** 30s (high-frequency) to 10min (low-frequency)
- **Delta bands:** ±$100 (tight) to ±$1,000 (wide)
- **Option selection:** ATM (max gamma), OTM (lower cost), multiple strikes (distributed gamma)

## Limitations

**Model Constraints:**
- Assumes flat interest rate and no dividends
- Black-Scholes framework (log-normal distribution, constant volatility)
- Cannot rebalance outside market hours

**Practical Risks:**
- Transaction costs from frequent rebalancing
- Slippage on market orders
- Early assignment risk on American options
- Liquidity constraints on wide spreads
