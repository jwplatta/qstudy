from AlgorithmImports import *


class SPXW1DTEIronCondor(QCAlgorithm):
    """
    1DTE SPX Iron Condor Strategy with iterative search and tweaking
    """

    def initialize(self):
        # Set dates and cash
        self.set_start_date(2023, 10, 1)
        self.set_end_date(2024, 3, 31)
        self.set_cash(100000)
        self.set_benchmark("SPX")
        self.set_brokerage_model(BrokerageName.CHARLES_SCHWAB, AccountType.MARGIN)

        self.settings.seed_initial_prices = True
        self.set_warmup(10, Resolution.MINUTE)
        self.spx = self.add_index("SPX", Resolution.MINUTE).symbol

        # Add SPXW index options (weeklys)
        self.option = self.add_index_option(self.spx, "SPXW", Resolution.MINUTE)
        self.option.set_filter(lambda x: x.include_weeklys().expiration(0, 1).strikes(-50, 50))
        self.spxw = self.option.symbol

        # Add VIX for expected move calculation
        self.vix = self.add_index("VIX", Resolution.HOUR).symbol
        self.vix1d = self.add_index("VIX1D", Resolution.HOUR).symbol
        self.vix9d = self.add_index("VIX9D", Resolution.HOUR).symbol

        # Trade tracking
        self.trade = None
        self.position_entered = False
        self.call_side_tested = False
        self.put_side_tested = False
        self.call_side_closed = False  # Track if call side already closed
        self.put_side_closed = False  # Track if put side already closed

        # Search parameters (from your Ruby class)
        self.spread_width = 20
        self.min_credit = 1.05  # $105
        self.max_credit = 1.45  # $150
        self.max_call_delta = 0.08
        self.min_call_delta = 0.03
        self.max_put_delta = 0.1
        self.min_put_delta = 0.03
        self.max_total_delta = 0.18
        self.credit_balance_ratio = 0.6
        self.delta_ratio = 0.6
        self.max_tweak_attempts = 100
        self.max_search_attempts = 5
        self.require_vix_contango = True
        self.min_contango_spread = 0.1

        # Schedule entry window (3:30pm EST day before expiration)
        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.at(15, 50, TimeZones.NEW_YORK),
            self.check_entry,
        )

        # Schedule 0DTE late exit
        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.at(13, 30, TimeZones.NEW_YORK),
            self.late_exit,
        )

        # Monitor positions every 5 minutes
        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.every(timedelta(minutes=5)),
            self.monitor_positions,
        )

    def check_entry(self):
        """Check if we should enter a new 1DTE iron condor"""
        if self.is_warming_up:
            return

        if self.position_entered:
            return

        if not self.is_market_open(self.spxw):
            self.debug("Market not open, skipping entry check")
            return

        tomorrow = self.time + timedelta(days=1)
        if tomorrow.weekday() >= 5:
            self.debug(f"Tomorrow is {tomorrow.strftime('%A')}, skipping entry")
            return

        if self.require_vix_contango and not self.is_vix_in_contango():
            return

        self.search_and_enter_iron_condor()

    def is_vix_in_contango(self):
        """
        Check if VIX term structure is in contango
        For 1DTE trading, we want near-term vol LOW and longer-term vol HIGHER
        Contango: VIX1D < VIX9D < VIX (30-day)
        This means implied volatility increases with time to expiration
        """
        # Get current VIX prices
        vix_price = self.securities[self.vix].price  # 30-day implied vol
        vix1d_price = self.securities[self.vix1d].price  # 1-day implied vol
        vix9d_price = self.securities[self.vix9d].price  # 9-day implied vol

        if not all([vix_price, vix1d_price, vix9d_price]):
            self.debug(
                f"Invalid VIX prices: VIX1D={vix1d_price}, VIX9D={vix9d_price}, VIX={vix_price}"
            )
            return False

        is_contango = vix1d_price < vix9d_price and vix9d_price < vix_price

        if is_contango:
            self.debug(
                f"VIX CONTANGO: VIX1D={vix1d_price:.2f} < VIX9D={vix9d_price:.2f} < VIX={vix_price:.2f} "
            )
        else:
            self.debug(
                f"VIX BACKWARDATION: VIX1D={vix1d_price:.2f}, VIX9D={vix9d_price:.2f}, VIX={vix_price:.2f}"
            )

        return is_contango

    def search_and_enter_iron_condor(self, attempt=0):
        """Iteratively search for valid iron condor strategy"""
        if attempt >= self.max_search_attempts:
            self.log(f"Failed to find valid strategy after {attempt} attempts")
            return

        # Get option chain
        chain = self.current_slice.option_chains.get(self.spxw)
        if not chain:
            self.log("No option chain available")
            return

        # Filter for 1DTE options
        tomorrow = self.time + timedelta(days=1)
        contracts = [x for x in chain if x.expiry.date() == tomorrow.date()]

        if not contracts:
            self.log("No 1DTE contracts found")
            return

        puts = sorted(
            [x for x in contracts if x.right == OptionRight.PUT],
            key=lambda x: x.strike,
            reverse=True,
        )
        calls = sorted(
            [x for x in contracts if x.right == OptionRight.CALL], key=lambda x: x.strike
        )

        if len(puts) < 2 or len(calls) < 2:
            self.log("Insufficient contracts")
            return

        # Get SPX price and calculate straddle
        spx_price = self.securities[self.spx].price
        straddle_price = self.calculate_straddle_price(contracts, spx_price)

        # Find initial strategy using 2-sigma strikes
        call_spread = self.find_initial_spread(calls, spx_price, straddle_price, "CALL")
        put_spread = self.find_initial_spread(puts, spx_price, straddle_price, "PUT")

        if not call_spread or not put_spread:
            self.log(f"Could not find initial spreads, retrying in 5s (attempt {attempt + 1})")
            self.schedule.on(
                self.date_rules.today,
                self.time_rules.after_market_open(self.spxw, 0),
                lambda: self.search_and_enter_iron_condor(attempt + 1),
            )
            return

        # Tweak strategy until valid
        result = self.tweak_strategy(call_spread, put_spread, calls, puts)

        if result:
            call_spread, put_spread, tweak_count = result
            self.enter_position(call_spread, put_spread, tweak_count, spx_price, straddle_price)
        else:
            self.log(f"Strategy tweaking failed, retrying in 5s (attempt {attempt + 1})")
            self.schedule.on(
                self.date_rules.today,
                self.time_rules.after_market_open(self.spxw, 0),
                lambda: self.search_and_enter_iron_condor(attempt + 1),
            )

    def calculate_straddle_price(self, contracts, spx_price):
        """Calculate ATM straddle price for expected move"""
        # Find ATM contracts
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
        # Calculate 2-sigma strike
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
            strategy_price = call_spread["price"] + put_spread["price"]

            # Check 1: Minimum credit
            if strategy_price < self.min_credit:
                # self.debug(f"Attempt #{tweak_attempts}: Credit too low {strategy_price:.2f} < {self.min_credit}")
                if call_spread["price"] < put_spread["price"]:
                    call_spread = self.move_spread_toward_atm(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_toward_atm(put_spread, puts, 5)
                continue

            # Check 2: Maximum credit
            if strategy_price > self.max_credit:
                # self.debug(f"Attempt #{tweak_attempts}: Credit too high {strategy_price:.2f} > {self.max_credit}")
                if call_spread["price"] > put_spread["price"]:
                    call_spread = self.move_spread_away_from_atm(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_away_from_atm(put_spread, puts, 5)
                continue

            # Check 3: Call delta too high
            if call_spread["delta"] > self.max_call_delta:
                # self.debug(f"Attempt #{tweak_attempts}: Call delta high {call_spread['delta']:.3f} > {self.max_call_delta}")
                call_spread = self.move_spread_away_from_atm(call_spread, calls, 5)
                continue

            # Check 4: Put delta too high
            if put_spread["delta"] > self.max_put_delta:
                # self.debug(f"Attempt #{tweak_attempts}: Put delta high {put_spread['delta']:.3f} > {self.max_put_delta}")
                put_spread = self.move_spread_away_from_atm(put_spread, puts, 5)
                continue

            # Check 5: Total delta too high
            total_delta = call_spread["delta"] + put_spread["delta"]
            if total_delta > self.max_total_delta:
                # self.debug(f"Attempt #{tweak_attempts}: Total delta high {total_delta:.3f} > {self.max_total_delta}")
                if call_spread["delta"] > put_spread["delta"]:
                    call_spread = self.move_spread_away_from_atm(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_away_from_atm(put_spread, puts, 5)
                continue

            # Check 6: Credit balance
            if not self.is_credit_balanced(call_spread["price"], put_spread["price"]):
                # self.debug(f"Attempt #{tweak_attempts}: Credit lopsided C:{call_spread['price']:.2f} P:{put_spread['price']:.2f}")

                if call_spread["price"] < put_spread["price"]:
                    call_spread = self.move_spread_toward_atm(call_spread, calls, 5)
                else:
                    put_spread = self.move_spread_toward_atm(put_spread, puts, 5)
                continue

            # Check 7: Delta balance
            if not self.is_delta_balanced(call_spread["delta"], put_spread["delta"]):
                # self.debug(f"Attempt #{tweak_attempts}: Delta lopsided C:{call_spread['delta']:.3f} P:{put_spread['delta']:.3f}")
                if call_spread["delta"] < put_spread["delta"]:
                    call_spread = self.move_spread_toward_atm(call_spread, calls, 5)
                    put_spread = self.move_spread_away_from_atm(put_spread, puts, 5)
                else:
                    put_spread = self.move_spread_toward_atm(put_spread, puts, 5)
                    call_spread = self.move_spread_away_from_atm(call_spread, calls, 5)
                continue

            # All checks passed!
            return (call_spread, put_spread, tweak_attempts)

        # Failed to find valid strategy
        self.debug(f"Could not find valid strategy after {tweak_attempts} tweak attempts")
        return None

    def move_spread_toward_atm(self, spread, contracts, points):
        """Move spread toward ATM (increases credit, increases delta)"""
        if spread["side"] == "CALL":
            new_short_strike = spread["short_leg"].strike - points
        else:
            new_short_strike = spread["short_leg"].strike + points

        return self.find_spread_at_strike(contracts, new_short_strike, spread["side"])

    def move_spread_away_from_atm(self, spread, contracts, points):
        """Move spread away from ATM (decreases credit, decreases delta)"""
        if spread["side"] == "CALL":
            new_short_strike = spread["short_leg"].strike + points
        else:
            new_short_strike = spread["short_leg"].strike - points

        return self.find_spread_at_strike(contracts, new_short_strike, spread["side"])

    def find_spread_at_strike(self, contracts, short_strike, side):
        """Find spread with given short strike (or nearest available)"""
        if not contracts:
            return None

        # Find short leg at exact strike, or nearest if not available
        short_leg = next((c for c in contracts if c.strike == short_strike), None)
        if not short_leg:
            short_leg = min(contracts, key=lambda x: abs(x.strike - short_strike))

        # Calculate target long strike
        if side == "CALL":
            long_strike = short_leg.strike + self.spread_width
        else:
            long_strike = short_leg.strike - self.spread_width

        # Find long leg at exact strike, or nearest if not available
        long_leg = next((c for c in contracts if c.strike == long_strike), None)
        if not long_leg:
            long_leg = min(contracts, key=lambda x: abs(x.strike - long_strike))

        # Validate spread: long leg should be further from ATM than short leg
        if side == "CALL" and long_leg.strike <= short_leg.strike:
            return None
        if side == "PUT" and long_leg.strike >= short_leg.strike:
            return None

        return {
            "short_leg": short_leg,
            "long_leg": long_leg,
            "price": round(short_leg.bid_price - long_leg.ask_price, 2),
            "delta": abs(short_leg.greeks.delta),
            "side": side,
        }

    def is_credit_balanced(self, call_credit, put_credit):
        """Check if credits are balanced"""
        smaller = min(call_credit, put_credit)
        larger = max(call_credit, put_credit)
        return (smaller / larger) >= self.credit_balance_ratio

    def is_delta_balanced(self, call_delta, put_delta):
        """Check if deltas are balanced"""
        smaller = min(call_delta, put_delta)
        larger = max(call_delta, put_delta)
        return (smaller / larger) >= self.delta_ratio

    def enter_position(self, call_spread, put_spread, tweak_count, spx_price, straddle_price):
        """Enter the iron condor position"""
        total_credit = call_spread["price"] + put_spread["price"]

        self.debug(
            f"FinalStrategy(tweaks={tweak_count}) SPX={spx_price:.2f} "
            + f"CALL={call_spread['short_leg'].strike}/{call_spread['long_leg'].strike} "
            + f"credit=${call_spread['price']:.2f} delta={call_spread['delta']:.3f} | "
            + f"PUT={put_spread['short_leg'].strike}/{put_spread['long_leg'].strike} "
            + f"credit=${put_spread['price']:.2f} delta={put_spread['delta']:.3f} | "
            + f"TOTAL=${total_credit:.2f}"
        )

        legs = [
            Leg.create(put_spread["long_leg"].symbol, 1),
            Leg.create(put_spread["short_leg"].symbol, -1),
            Leg.create(call_spread["short_leg"].symbol, -1),
            Leg.create(call_spread["long_leg"].symbol, 1),
        ]

        self.combo_market_order(legs, 1)

        self.trade = {
            "entry_time": self.time,
            "entry_credit": round(total_credit, 2),
            "call_credit": round(call_spread["price"], 2),
            "put_credit": round(put_spread["price"], 2),
            "profit_target": total_credit * 0.6,
            "max_loss": total_credit * -3.5,
            "long_put": put_spread["long_leg"].symbol,
            "short_put": put_spread["short_leg"].symbol,
            "short_call": call_spread["short_leg"].symbol,
            "long_call": call_spread["long_leg"].symbol,
            "expiry": call_spread["short_leg"].expiry,
        }

        self.position_entered = True
        self.call_side_tested = False
        self.put_side_tested = False
        self.call_side_closed = False
        self.put_side_closed = False

    def monitor_positions(self):
        """Monitor positions for exit conditions"""
        if self.is_warming_up or not self.position_entered or not self.trade:
            return

        # Skip first hour on 0DTE
        if self.is_0dte() and self.time.hour == 9 and self.time.minute < 30:
            return

        current_pnl = self.calculate_pnl()

        # Profit target
        if current_pnl >= self.trade["profit_target"]:
            self.exit_position("Profit target reached")
            return

        # Max loss
        if current_pnl <= self.trade["max_loss"]:
            self.exit_position("Max loss hit")
            return

    def late_exit(self):
        """Exit by late on 0DTE"""
        if self.is_warming_up:
            return

        if self.is_0dte() and self.position_entered:
            self.exit_position("0DTE late exit")

    def is_0dte(self):
        """Check if 0DTE"""
        if not self.trade:
            return False
        return self.trade["expiry"].date() == self.time.date()

    def check_individual_side_exits(self):
        """Check if either call or put side should be closed independently"""
        # Calculate P&L for each side
        call_pnl = self.calculate_call_side_pnl()
        put_pnl = self.calculate_put_side_pnl()

        # Calculate individual profit targets (50% of each side's credit)
        call_profit_target = self.trade["call_credit"] * 0.5
        put_profit_target = self.trade["put_credit"] * 0.5

        # Check call side
        if not self.call_side_closed and call_pnl >= call_profit_target:
            self.exit_call_side("Call side 50% profit target reached")

        # Check put side
        if not self.put_side_closed and put_pnl >= put_profit_target:
            self.exit_put_side("Put side 50% profit target reached")

    def calculate_pnl(self):
        """Calculate current P&L"""
        call_pnl = (
            self.calculate_call_side_pnl()
            if not self.call_side_closed
            else self.trade["call_credit"]
        )
        put_pnl = (
            self.calculate_put_side_pnl() if not self.put_side_closed else self.trade["put_credit"]
        )

        return call_pnl + put_pnl

    def calculate_put_side_pnl(self):
        """Calculate current P&L for put spread only"""
        if self.put_side_closed:
            return self.trade["put_credit"]  # Already captured full profit

        short_put_price = self.securities[self.trade["short_put"]].price
        long_put_price = self.securities[self.trade["long_put"]].price

        # Exit cost for put spread (negative = credit, positive = debit)
        put_exit_cost = (long_put_price - short_put_price) * -1

        # P&L = entry credit - exit cost
        put_pnl = self.trade["put_credit"] - put_exit_cost
        return put_pnl

    def calculate_call_side_pnl(self):
        """Calculate current P&L for call spread only"""
        if self.call_side_closed:
            return self.trade["call_credit"]  # Already captured full profit

        short_call_price = self.securities[self.trade["short_call"]].price
        long_call_price = self.securities[self.trade["long_call"]].price

        # Exit cost for call spread (negative = credit, positive = debit)
        call_exit_cost = (long_call_price - short_call_price) * -1

        # P&L = entry credit - exit cost
        call_pnl = self.trade["call_credit"] - call_exit_cost
        return call_pnl

    def exit_call_side(self, reason):
        """Close only the call spread"""
        if self.call_side_closed:
            return

        # Calculate exit price for call spread
        short_call_price = self.securities[self.trade["short_call"]].bid_price
        long_call_price = self.securities[self.trade["long_call"]].ask_price
        call_exit_price = -(short_call_price - long_call_price)

        # Close call spread
        legs = [
            Leg.create(self.trade["short_call"], 1),  # Buy back short call
            Leg.create(self.trade["long_call"], -1),  # Sell long call
        ]

        # ticket = self.combo_limit_order(legs, 1, call_exit_price)
        ticket = self.combo_market_order(legs, 1)

        call_pnl = self.calculate_call_side_pnl()
        pct = (call_pnl / self.trade["call_credit"]) * 100 if self.trade["call_credit"] > 0 else 0

        self.debug(f"CALL EXIT: {reason} | P&L=${call_pnl:.2f} ({pct:.1f}% of call credit)")

        self.call_side_closed = True
        return ticket

    def exit_put_side(self, reason):
        """Close only the put spread"""
        if self.put_side_closed:
            return

        # Calculate exit price for put spread
        short_put_price = self.securities[self.trade["short_put"]].bid_price
        long_put_price = self.securities[self.trade["long_put"]].ask_price
        put_exit_price = -(short_put_price - long_put_price)

        # Close put spread
        legs = [
            Leg.create(self.trade["short_put"], 1),  # Buy back short put
            Leg.create(self.trade["long_put"], -1),  # Sell long put
        ]

        # ticket = self.combo_limit_order(legs, 1, put_exit_price)
        ticket = self.combo_market_order(legs, 1)

        put_pnl = self.calculate_put_side_pnl()
        pct = (put_pnl / self.trade["put_credit"]) * 100 if self.trade["put_credit"] > 0 else 0

        self.debug(f"PUT EXIT: {reason} | P&L=${put_pnl:.2f} ({pct:.1f}% of put credit)")

        self.put_side_closed = True
        return ticket

    def exit_position(self, reason, exit_price=None):
        """Exit position with limit order at specified price"""
        if not self.position_entered:
            return

        # Build legs only for sides that are still open
        legs = []

        if not self.put_side_closed:
            legs.extend(
                [
                    Leg.create(self.trade["long_put"], -1),  # Sell long put
                    Leg.create(self.trade["short_put"], 1),  # Buy back short put
                ]
            )

        if not self.call_side_closed:
            legs.extend(
                [
                    Leg.create(self.trade["short_call"], 1),  # Buy back short call
                    Leg.create(self.trade["long_call"], -1),  # Sell long call
                ]
            )

        if not legs:
            # Both sides already closed
            self.debug("Both sides already closed")
            self.position_entered = False
            self.trade = None
            return

        if exit_price is None:
            exit_price = self.calculate_exit_price()

        # Place combo limit order at the exit price
        # ticket = self.combo_limit_order(legs, 1, exit_price)
        ticket = self.combo_market_order(legs, 1)

        # Calculate expected P&L based on exit price
        expected_pnl = self.calculate_pnl()
        pct = (
            (expected_pnl / self.trade["entry_credit"]) * 100
            if self.trade["entry_credit"] > 0
            else 0
        )

        self.debug(f"EXIT ORDER: {reason} | Expected P&L=${expected_pnl:.2f} ({pct:.1f}%)")

        self.position_entered = False
        self.trade = None
        self.adjustment_count = 0
        self.call_side_tested = False
        self.put_side_tested = False
        self.call_side_closed = False
        self.put_side_closed = False

        return ticket

    def calculate_exit_price(self):
        """Calculate current market price to exit the iron condor"""
        exit_price = 0

        if not self.put_side_closed:
            long_put_price = self.securities[self.trade["long_put"]].ask_price
            short_put_price = self.securities[self.trade["short_put"]].bid_price
            put_spread_cost = short_put_price - long_put_price
            exit_price += -put_spread_cost

        if not self.call_side_closed:
            short_call_price = self.securities[self.trade["short_call"]].bid_price
            long_call_price = self.securities[self.trade["long_call"]].ask_price
            call_spread_cost = short_call_price - long_call_price
            exit_price += -call_spread_cost

        return exit_price
