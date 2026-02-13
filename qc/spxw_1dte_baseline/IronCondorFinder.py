from AlgorithmImports import *


class IronCondorFinder:
    """
    Finds and constructs iron condor strategies based on configurable parameters.
    Uses iterative search and tweaking to meet credit, delta, and balance requirements.
    """

    def __init__(
        self,
        spread_width=20,
        min_credit=1.05,
        max_credit=1.45,
        max_call_delta=0.08,
        min_call_delta=0.03,
        max_put_delta=0.1,
        min_put_delta=0.03,
        max_total_delta=0.18,
        credit_balance_ratio=0.6,
        delta_ratio=0.6,
        max_tweak_attempts=100,
    ):
        self.spread_width = spread_width
        self.min_credit = min_credit
        self.max_credit = max_credit
        self.max_call_delta = max_call_delta
        self.min_call_delta = min_call_delta
        self.max_put_delta = max_put_delta
        self.min_put_delta = min_put_delta
        self.max_total_delta = max_total_delta
        self.credit_balance_ratio = credit_balance_ratio
        self.delta_ratio = delta_ratio
        self.max_tweak_attempts = max_tweak_attempts

    def find_iron_condor(self, contracts, spx_price):
        """
        Find a valid iron condor strategy from the given option chain.

        Args:
            contracts: List of option contracts
            spx_price: Current SPX price

        Returns:
            Tuple of (call_spread, put_spread, tweak_count) if valid strategy found, else None
        """
        puts = sorted(
            [x for x in contracts if x.right == OptionRight.PUT],
            key=lambda x: x.strike,
            reverse=True,
        )
        calls = sorted(
            [x for x in contracts if x.right == OptionRight.CALL], key=lambda x: x.strike
        )

        if len(puts) < 2 or len(calls) < 2:
            return None

        straddle_price = self.calculate_straddle_price(contracts, spx_price)

        call_spread = self.find_initial_spread(calls, spx_price, straddle_price, "CALL")
        if call_spread == None:
            raise Exception("No valid call spread found")

        put_spread = self.find_initial_spread(puts, spx_price, straddle_price, "PUT")
        if put_spread == None:
            raise Exception("No valid put spread found")

        if not call_spread or not put_spread:
            return None

        result = self.tweak_strategy(call_spread, put_spread, calls, puts)

        return result

    def calculate_straddle_price(self, contracts, spx_price):
        """Calculate ATM straddle price for expected move"""
        call_atm = min(
            [c for c in contracts if c.right == OptionRight.CALL],
            key=lambda x: abs(x.strike - spx_price),
        )
        put_atm = min(
            [c for c in contracts if c.right == OptionRight.PUT],
            key=lambda x: abs(x.strike - spx_price),
        )

        return round(call_atm.ask_price + put_atm.ask_price)

    def find_initial_spread(self, contracts, spx_price, straddle_price, side):
        """Find initial spread at 2-sigma strike"""
        if side == "CALL":
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
        if side == "CALL":
            long_strike = short_leg.strike + self.spread_width
        else:
            long_strike = short_leg.strike - self.spread_width

        long_leg = next((c for c in contracts if c.strike == long_strike), None)

        if not long_leg:
            return None

        return {
            "short_leg": short_leg,
            "long_leg": long_leg,
            "price": round(short_leg.bid_price - long_leg.ask_price, 2),
            "delta": abs(short_leg.greeks.delta),
            "side": side,
        }

    def tweak_strategy(self, call_spread, put_spread, calls, puts):
        """Iteratively tweak strategy until it meets all criteria"""
        tweak_attempts = 0

        while tweak_attempts < self.max_tweak_attempts:
            tweak_attempts += 1

            # Validate spreads are not None before accessing
            if not call_spread or not put_spread:
                return None

            strategy_price = call_spread["price"] + put_spread["price"]

            # Check 1: Minimum credit
            if strategy_price < self.min_credit:
                if call_spread["price"] < put_spread["price"]:
                    call_spread = self.move_spread_up(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_up(put_spread, puts, 5)
                continue

            # Check 2: Maximum credit
            if strategy_price > self.max_credit:
                if call_spread["price"] > put_spread["price"]:
                    call_spread = self.move_spread_away(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_away(put_spread, puts, 5)
                continue

            # Check 3: Call delta too high
            if call_spread["delta"] > self.max_call_delta:
                call_spread = self.move_spread_away(call_spread, calls, 5)
                continue

            # Check 4: Put delta too high
            if put_spread["delta"] > self.max_put_delta:
                put_spread = self.move_spread_away(put_spread, puts, 5)
                continue

            # Check 5: Total delta too high
            total_delta = call_spread["delta"] + put_spread["delta"]
            if total_delta > self.max_total_delta:
                if call_spread["delta"] > put_spread["delta"]:
                    call_spread = self.move_spread_away(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_away(put_spread, puts, 5)
                continue

            # Check 6: Credit balance
            if not self.is_credit_balanced(call_spread["price"], put_spread["price"]):
                if call_spread["price"] < put_spread["price"]:
                    call_spread = self.move_spread_up(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_up(put_spread, puts, 5)
                continue

            # Check 7: Delta balance
            if not self.is_delta_balanced(call_spread["delta"], put_spread["delta"]):
                if call_spread["delta"] < put_spread["delta"]:
                    call_spread = self.move_spread_up(call_spread, calls, 5)
                    put_spread = self.move_spread_away(put_spread, puts, 5)
                else:
                    put_spread = self.move_spread_up(put_spread, puts, 5)
                    call_spread = self.move_spread_away(call_spread, calls, 5)
                continue

            return (call_spread, put_spread, tweak_attempts)

        return None

    def move_spread_up(self, spread, contracts, points):
        """Move spread toward ATM (increases credit, increases delta)"""
        if spread["side"] == "CALL":
            new_short_strike = spread["short_leg"].strike - points
        else:
            new_short_strike = spread["short_leg"].strike + points

        return self.build_spread(contracts, new_short_strike, spread["side"])

    def move_spread_away(self, spread, contracts, points):
        """Move spread away from ATM (decreases credit, decreases delta)"""
        if spread["side"] == "CALL":
            new_short_strike = spread["short_leg"].strike + points
        else:
            new_short_strike = spread["short_leg"].strike - points

        return self.build_spread(contracts, new_short_strike, spread["side"])

    def build_spread(self, contracts, short_strike, side):
        """Find spread with given short strike (or nearest available)"""
        if not contracts:
            return None

        # Find short leg at exact strike, or nearest if not available
        short_leg = next((c for c in contracts if c.strike == short_strike), None)
        if not short_leg:
            short_leg = min(contracts, key=lambda x: abs(x.strike - short_strike))

        # Find valid long leg candidates (must be on correct side of short leg)
        if side == "CALL":
            # For calls, long leg must be higher strike (further OTM)
            valid_longs = [c for c in contracts if c.strike > short_leg.strike]
            target_long_strike = short_leg.strike + self.spread_width
        else:
            # For puts, long leg must be lower strike (further OTM)
            valid_longs = [c for c in contracts if c.strike < short_leg.strike]
            target_long_strike = short_leg.strike - self.spread_width

        if not valid_longs:
            return None

        # Try to find exact target strike first, otherwise find nearest valid option
        long_leg = next((c for c in valid_longs if c.strike == target_long_strike), None)
        if not long_leg:
            # Find the closest valid long leg to our target
            long_leg = min(valid_longs, key=lambda x: abs(x.strike - target_long_strike))

        # Get delta from greeks
        delta = abs(short_leg.greeks.delta)

        return {
            "short_leg": short_leg,
            "long_leg": long_leg,
            "price": round(short_leg.bid_price - long_leg.ask_price, 2),
            "delta": delta,
            "side": side,
        }

    def is_credit_balanced(self, call_credit, put_credit):
        """Check if credits are balanced"""
        smaller = min(call_credit, put_credit)
        larger = max(call_credit, put_credit)
        if larger == 0:
            return False
        return (smaller / larger) >= self.credit_balance_ratio

    def is_delta_balanced(self, call_delta, put_delta):
        """Check if deltas are balanced"""
        smaller = min(call_delta, put_delta)
        larger = max(call_delta, put_delta)
        if larger == 0:
            return False  # Avoid division by zero
        return (smaller / larger) >= self.delta_ratio
