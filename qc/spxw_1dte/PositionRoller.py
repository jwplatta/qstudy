# region imports
from AlgorithmImports import *
# endregion

# Your New Python File
from AlgorithmImports import *
from util import find_spread_with_target_delta, determine_tested_side

class PositionRoller:
    """Handles all position rolling logic"""
    
    def __init__(self, algorithm, iron_condor_finder):
        self.algorithm = algorithm
        self.iron_condor_finder = iron_condor_finder
    
    def attempt_roll_on_max_loss(self, trade):
        """Attempt to roll tested side on max loss instead of exiting"""
        spx_price = self.algorithm.securities[self.algorithm.spx].price
        short_call_strike = self.algorithm.securities[trade["short_call"]].strike
        short_put_strike = self.algorithm.securities[trade["short_put"]].strike
        
        tested_side, _, _ = determine_tested_side(spx_price, short_call_strike, short_put_strike)

        return self.roll_for_side(trade, tested_side)
    
    def attempt_late_exit_roll(self, trade):
        """Exit untested side and roll tested side on 0DTE"""
        spx_price = self.algorithm.securities[self.algorithm.spx].price
        short_call_strike = self.algorithm.securities[trade["short_call"]].strike
        short_put_strike = self.algorithm.securities[trade["short_put"]].strike
        tested_side, _, _ = determine_tested_side(spx_price, short_call_strike, short_put_strike)

        return self.roll_for_side(trade, tested_side)
    
    def roll_for_side(self, trade, side):
        """Roll tested side to best expiry within next week for credit"""
        chain = self.algorithm.current_slice.option_chains.get(self.algorithm.spxw)
        if not chain:
            return None
        
        current_expiry = trade["expiry"].date()
        today = self.algorithm.time.date()
        one_week_out = today + timedelta(days=7)
        available_expiries = sorted(set([c.expiry.date() for c in chain if current_expiry < c.expiry.date() <= one_week_out]))
        
        if not available_expiries:
            return None
        
        best_expiry = None
        best_roll_credit = float('-inf')
        best_spreads = None
        spx_price = self.algorithm.securities[self.algorithm.spx].price
        
        for expiry_date in available_expiries:
            expiry_contracts = [c for c in chain if c.expiry.date() == expiry_date]
            if len(expiry_contracts) < 4:
                continue
            
            calls = sorted([c for c in expiry_contracts if c.right == OptionRight.CALL], key=lambda x: x.strike)
            puts = sorted([c for c in expiry_contracts if c.right == OptionRight.PUT], key=lambda x: x.strike, reverse=True)
            if len(calls) < 2 or len(puts) < 2:
                continue
            
            spread_width = self.iron_condor_finder.spread_width
            if side == "call":
                tested_spread = find_spread_with_target_delta(calls, 0.20, spread_width, "call")
                untested_spread = self.iron_condor_finder.find_initial_spread(puts, spx_price, self.iron_condor_finder.calculate_straddle_price(expiry_contracts, spx_price), "PUT")
            else:
                tested_spread = find_spread_with_target_delta(puts, 0.20, spread_width, "put")
                untested_spread = self.iron_condor_finder.find_initial_spread(calls, spx_price, self.iron_condor_finder.calculate_straddle_price(expiry_contracts, spx_price), "CALL")
            
            if not tested_spread or not untested_spread:
                continue
            
            current_tested_short_price = self.algorithm.securities[trade["short_call" if side == "call" else "short_put"]].ask_price
            current_tested_long_price = self.algorithm.securities[trade["long_call" if side == "call" else "long_put"]].bid_price
            current_untested_short_price = self.algorithm.securities[trade["short_put" if side == "call" else "short_call"]].ask_price
            current_untested_long_price = self.algorithm.securities[trade["long_put" if side == "call" else "long_call"]].bid_price
            
            close_cost = (current_tested_short_price - current_tested_long_price) + (current_untested_short_price - current_untested_long_price)
            new_credit = tested_spread["price"] + untested_spread["price"]
            roll_credit = new_credit - close_cost
            
            if roll_credit > best_roll_credit:
                best_roll_credit = roll_credit
                best_expiry = expiry_date
                best_spreads = {"tested": tested_spread, "untested": untested_spread, "roll_credit": roll_credit}
        
        if not best_spreads:
            return None
        
        tested_spread = best_spreads["tested"]
        untested_spread = best_spreads["untested"]
        
        current_tested_short = trade["short_call"] if side == "call" else trade["short_put"]
        current_tested_long = trade["long_call"] if side == "call" else trade["long_put"]
        current_untested_short = trade["short_put"] if side == "call" else trade["short_call"]
        current_untested_long = trade["long_put"] if side == "call" else trade["long_call"]
        
        legs = [
            Leg.create(current_tested_short, 1), Leg.create(current_tested_long, -1),
            Leg.create(current_untested_short, 1), Leg.create(current_untested_long, -1),
            Leg.create(tested_spread["short_leg"].symbol, -1), Leg.create(tested_spread["long_leg"].symbol, 1),
            Leg.create(untested_spread["short_leg"].symbol, -1), Leg.create(untested_spread["long_leg"].symbol, 1),
        ]
        
        self.algorithm.combo_market_order(legs, 1)
        
        new_total_credit = tested_spread["price"] + untested_spread["price"]
        if side == "call":
            trade["short_call"] = tested_spread["short_leg"].symbol
            trade["long_call"] = tested_spread["long_leg"].symbol
            trade["short_put"] = untested_spread["short_leg"].symbol
            trade["long_put"] = untested_spread["long_leg"].symbol
            trade["call_credit"] = round(tested_spread["price"], 2)
            trade["put_credit"] = round(untested_spread["price"], 2)
        else:
            trade["short_put"] = tested_spread["short_leg"].symbol
            trade["long_put"] = tested_spread["long_leg"].symbol
            trade["short_call"] = untested_spread["short_leg"].symbol
            trade["long_call"] = untested_spread["long_leg"].symbol
            trade["put_credit"] = round(tested_spread["price"], 2)
            trade["call_credit"] = round(untested_spread["price"], 2)
        
        trade["expiry"] = tested_spread["short_leg"].expiry
        trade["entry_credit"] = round(new_total_credit, 2)
        trade["cumulative_credit"] += round(new_total_credit, 2)
        trade["profit_target"] = trade["cumulative_credit"] * 0.6
        trade["max_loss"] = trade["cumulative_credit"] * -3.5
        
        self.algorithm.debug(f"ROLLED to {best_expiry}: TOTAL=${new_total_credit:.2f}")
        
        return trade
