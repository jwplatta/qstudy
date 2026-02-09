# region imports
from AlgorithmImports import *
# endregion

def calculate_straddle_price(contracts, spx_price):
    """Calculate ATM straddle price for expected move"""
    # Find ATM contracts
    call_atm = min([c for c in contracts if c.right == OptionRight.CALL],
                    key=lambda x: abs(x.strike - spx_price))
    put_atm = min([c for c in contracts if c.right == OptionRight.PUT],
                    key=lambda x: abs(x.strike - spx_price))
    
    return round(call_atm.ask_price + put_atm.ask_price)

def find_initial_spread(contracts, spx_price, straddle_price, side, spread_width):
    """Find initial spread at 2-sigma strike"""
    # Calculate 2-sigma strike
    if side == 'CALL':
        target_strike = spx_price + (2 * straddle_price)
    else:
        target_strike = spx_price - (2 * straddle_price)
    
    # Round to nearest $5
    target_strike = round(target_strike / 5) * 5
    
    # Find short leg at target strike
    short_leg = next((c for c in contracts if c.strike == target_strike), None)
    if not short_leg:
        # Find closest strike
        short_leg = min(contracts, key=lambda x: abs(x.strike - target_strike))
    
    # Find long leg (spread_width away)
    if side == 'CALL':
        long_strike = short_leg.strike + spread_width
    else:
        long_strike = short_leg.strike - spread_width
    
    long_leg = next((c for c in contracts if c.strike == long_strike), None)
    
    if not long_leg:
        return None
    
    return {
        'short_leg': short_leg,
        'long_leg': long_leg,
        'price': round(short_leg.bid_price - long_leg.ask_price, 2),
        'delta': abs(short_leg.greeks.delta),
        'side': side
    }


def is_credit_balanced(call_credit, put_credit, credit_balance_ratio):
        """Check if credits are balanced"""
        smaller = min(call_credit, put_credit)
        larger = max(call_credit, put_credit)
        return (smaller / larger) >= credit_balance_ratio
    
def is_delta_balanced(call_delta, put_delta, delta_ratio):
    """Check if deltas are balanced"""
    smaller = min(call_delta, put_delta)
    larger = max(call_delta, put_delta)
    return (smaller / larger) >= delta_ratio

def calculate_pnl(trade, securities, call_side_closed, put_side_closed):
    """Calculate current P&L"""
    call_pnl = (
        calculate_call_side_pnl(trade, securities, call_side_closed)
        if not call_side_closed
        else trade["call_credit"]
    )
    put_pnl = (
        calculate_put_side_pnl(trade, securities, put_side_closed)
        if not put_side_closed
        else trade["put_credit"]
    )
    return call_pnl + put_pnl

def calculate_put_side_pnl(trade, securities, put_side_closed):
    """Calculate current P&L for put spread only"""
    if put_side_closed:
        return trade["put_credit"]

    short_put_price = securities[trade["short_put"]].price
    long_put_price = securities[trade["long_put"]].price
    put_exit_cost = short_put_price - long_put_price
    put_pnl = trade["put_credit"] - put_exit_cost
    return put_pnl

def calculate_call_side_pnl(trade, securities, call_side_closed):
    """Calculate current P&L for call spread only"""
    if call_side_closed:
        return trade["call_credit"]

    short_call_price = securities[trade["short_call"]].price
    long_call_price = securities[trade["long_call"]].price
    call_exit_cost = short_call_price - long_call_price
    call_pnl = trade["call_credit"] - call_exit_cost
    return call_pnl

def find_spread_with_target_delta(contracts, max_delta, spread_width, side):
    """Find a spread where short leg delta is <= max_delta"""
    for contract in contracts:
        delta = abs(contract.greeks.delta)
        
        if delta > max_delta:
            continue

        if side == "call":
            long_strike = contract.strike + spread_width
        else:
            long_strike = contract.strike - spread_width

        long_leg = next((c for c in contracts if c.strike == long_strike), None)
        
        if not long_leg:
            continue

        if side == "call" and long_leg.strike <= contract.strike:
            continue
        if side == "put" and long_leg.strike >= contract.strike:
            continue

        return {
            "short_leg": contract,
            "long_leg": long_leg,
            "price": round(contract.bid_price - long_leg.ask_price, 2),
            "delta": delta,
            "side": side.upper(),
        }

    return None

def determine_tested_side(spx_price, short_call_strike, short_put_strike):
    """Determine which side is tested (closer to being ITM)"""
    call_distance = short_call_strike - spx_price
    put_distance = spx_price - short_put_strike
    
    if call_distance < put_distance:
        return "call", call_distance, put_distance
    else:
        return "put", call_distance, put_distance
