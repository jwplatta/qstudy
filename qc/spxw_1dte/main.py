from AlgorithmImports import *
from IronCondorFinder import IronCondorFinder
from PositionRoller import PositionRoller
from VIXContango import VIXContango
from util import calculate_pnl, determine_tested_side

class SPXW1DTEIronCondor(QCAlgorithm):
    """1DTE SPX Iron Condor Strategy with rolling on max loss"""
    def initialize(self):
        self.set_start_date(2024, 2, 1)
        self.set_end_date(2024, 5, 31)
        self.set_cash(20000)
        self.set_brokerage_model(BrokerageName.CHARLES_SCHWAB, AccountType.MARGIN)

        self.settings.seed_initial_prices = True
        self.set_warmup(10, Resolution.MINUTE)
        self.spx = self.add_index("SPX", Resolution.MINUTE).symbol

        # Add SPXW index options (weeklys)
        self.option = self.add_index_option(self.spx, "SPXW", Resolution.MINUTE)
        self.option.set_filter(lambda x: x.include_weeklys().expiration(0, 10).strikes(-60, 60))
        self.spxw = self.option.symbol

        # Add VIX for expected move calculation
        self.vix = self.add_index("VIX", Resolution.HOUR).symbol
        self.vix1d = self.add_index("VIX1D", Resolution.HOUR).symbol
        self.vix9d = self.add_index("VIX9D", Resolution.HOUR).symbol

        # Trade tracking
        self.trade = None
        self.position_entered = False
        self.call_side_closed = False
        self.put_side_closed = False
        self._pending_call_side_close = False
        self._pending_put_side_close = False

        # Iron Condor Finder
        self.iron_condor_finder = IronCondorFinder(
            spread_width=20,
            min_credit=1.05,
            max_credit=1.45,
            max_call_delta=0.08,
            min_call_delta=0.03,
            max_put_delta=0.1,
            min_put_delta=0.03,
            max_total_delta=0.18,
            credit_balance_ratio=0.5,
            delta_ratio=0.5,
            max_tweak_attempts=100,
        )

        # Position roller and VIX contango checker
        self.position_roller = PositionRoller(self, self.iron_condor_finder)
        self.vix_contango = VIXContango(self)

        # Search parameters
        self.max_search_attempts = 5
        self.require_vix_contango = True

        # Schedule entry window
        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.at(15, 55, TimeZones.NEW_YORK),
            self.check_entry
        )

        # Monitor positions every 5 minutes
        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.every(timedelta(minutes=5)),
            self.monitor_positions,
        )

    def check_entry(self):
        if self.is_warming_up or self.position_entered:
            return

        if self.require_vix_contango and not self.vix_contango.is_in_contango():
            return
        
        self.search_and_enter_iron_condor()
    
    def search_and_enter_iron_condor(self, attempt=0):
        if attempt >= self.max_search_attempts:
            return
        chain = self.current_slice.option_chains.get(self.spxw)
        if not chain:
            return
        tomorrow = self.time + timedelta(days=1)
        contracts = [x for x in chain if x.expiry.date() == tomorrow.date()]
        if not contracts:
            return
        spx_price = self.securities[self.spx].price
        result = self.iron_condor_finder.find_iron_condor(contracts, spx_price)

        if result:
            call_spread, put_spread, tweak_count = result
            self.enter_position(call_spread, put_spread, spx_price)
        else:
            self.debug("No trade found. Skipping entry.")

    def enter_position(self, call_spread, put_spread, spx_price):
        total_credit = call_spread["price"] + put_spread["price"]
        self.debug(
            f"ENTRY: SPX={spx_price:.2f} "
            f"CALL={call_spread['short_leg'].strike}/{call_spread['long_leg'].strike} "
            f"PUT={put_spread['short_leg'].strike}/{put_spread['long_leg'].strike} "
            f"TOTAL=${total_credit:.2f}"
        )

        legs = [
            Leg.create(put_spread["long_leg"].symbol, 1),
            Leg.create(put_spread["short_leg"].symbol, -1),
            Leg.create(call_spread["short_leg"].symbol, -1),
            Leg.create(call_spread["long_leg"].symbol, 1),
        ]
        self.combo_market_order(legs, 1)

        self.trade = {
            "entry_credit": round(total_credit, 2),
            "call_credit": round(call_spread["price"], 2),
            "put_credit": round(put_spread["price"], 2),
            "cumulative_credit": round(total_credit, 2),
            "profit_target": total_credit * 0.6,
            "max_loss": total_credit * -3.5,
            "long_put": put_spread["long_leg"].symbol,
            "short_put": put_spread["short_leg"].symbol,
            "short_call": call_spread["short_leg"].symbol,
            "long_call": call_spread["long_leg"].symbol,
            "expiry": call_spread["short_leg"].expiry,
            "entry_spx_price": round(spx_price, 2),
        }
        self.position_entered = True
        self.call_side_closed = False
        self.put_side_closed = False
        self._pending_call_side_close = False
        self._pending_put_side_close = False

    def monitor_positions(self):
        if self.is_warming_up or not self.position_entered or not self.trade:
            return

        if self.is_0dte() and self.time.hour < 9:
            return

        current_pnl = calculate_pnl(self.trade, self.securities, self.call_side_closed, self.put_side_closed)
        if current_pnl >= self.trade["profit_target"]:
            self.exit_position("Profit target reached")
            return

        if current_pnl <= self.trade["max_loss"]:
            self.attempt_roll_on_max_loss()
            return
        
        if self.is_0dte():
            chain = self.current_slice.option_chains.get(self.spxw)
            if not chain:
                return
            
            short_call_contract = next((c for c in chain if c.symbol == self.trade["short_call"]), None)
            short_put_contract = next((c for c in chain if c.symbol == self.trade["short_put"]), None)
            
            short_call_delta = abs(short_call_contract.greeks.delta) if short_call_contract else 0
            short_put_delta = abs(short_put_contract.greeks.delta) if short_put_contract else 0
            
            # Only check prices for sides that exist in current securities and haven't been closed
            call_buyback_cost = float('inf')
            if not self.call_side_closed and self.trade["short_call"] in self.securities and self.trade["long_call"] in self.securities:
                short_call_price = self.securities[self.trade["short_call"]].bid_price
                long_call_price = self.securities[self.trade["long_call"]].ask_price
                call_buyback_cost = short_call_price - long_call_price
            
            put_buyback_cost = float('inf')
            if not self.put_side_closed and self.trade["short_put"] in self.securities and self.trade["long_put"] in self.securities:
                short_put_price = self.securities[self.trade["short_put"]].bid_price
                long_put_price = self.securities[self.trade["long_put"]].ask_price
                put_buyback_cost = short_put_price - long_put_price
            
            if call_buyback_cost <= 0.20 and call_buyback_cost != float('inf') and not self.call_side_closed:
                self.debug(f"Call spread buyback cost ${call_buyback_cost:.2f} <= 0.20, exiting call side")
                self.exit_call_side("Call spread cheap buyback")
            
            if put_buyback_cost <= 0.20 and put_buyback_cost != float('inf') and not self.put_side_closed:
                self.debug(f"Put spread buyback cost ${put_buyback_cost:.2f} <= 0.20, exiting put side")
                self.exit_put_side("Put spread cheap buyback")
            
            if short_call_delta > 0.30 and not self.call_side_closed:
                self.debug(f"Call short delta {short_call_delta:.3f} > 0.30, rolling call side")
                if not self.put_side_closed:
                    self.exit_put_side("Rolling call side - exit put side")
                rolled = self.position_roller.roll_for_side(self.trade, "call")
                if rolled:
                    self.trade = rolled
                    self._pending_call_side_close = False
                    self._pending_put_side_close = False
            
            if short_put_delta > 0.30 and not self.put_side_closed:
                self.debug(f"Put short delta {short_put_delta:.3f} > 0.30, rolling put side")
                if not self.call_side_closed:
                    self.exit_call_side("Rolling put side - exit call side")
                rolled = self.position_roller.roll_for_side(self.trade, "put")
                if rolled:
                    self.trade = rolled
                    self._pending_call_side_close = False
                    self._pending_put_side_close = False

    def attempt_roll_on_max_loss(self):
        new_trade = self.position_roller.attempt_roll_on_max_loss(self.trade)

        if new_trade:
            self.trade = new_trade
            self.call_side_closed = False
            self.put_side_closed = False
            self._pending_call_side_close = False
            self._pending_put_side_close = False
        else:
            self.exit_position("Max loss - roll failed")

    def is_0dte(self):
        return self.trade and self.trade["expiry"].date() == self.time.date()
    
    def exit_call_side(self, reason):
        if self.call_side_closed or self._pending_call_side_close:
            return
            
        legs = [Leg.create(self.trade["short_call"], 1), Leg.create(self.trade["long_call"], -1)]
        
        self.debug(f"CALL EXIT ORDERED: {reason}")
        self.combo_market_order(legs, 1)
        
        self._pending_call_side_close = True
    
    def exit_put_side(self, reason):
        if self.put_side_closed or self._pending_put_side_close:
            return
        legs = [Leg.create(self.trade["short_put"], 1), Leg.create(self.trade["long_put"], -1)]
        
        self.debug(f"PUT EXIT ORDERED: {reason}")
        self.combo_market_order(legs, 1)
        
        self._pending_put_side_close = True
    
    def exit_position(self, reason):
        if not self.position_entered:
            return
        legs = []

        self.debug(f"EXIT ORDERED: {reason}")

        if not self.put_side_closed:
            legs.extend([Leg.create(self.trade["long_put"], -1), Leg.create(self.trade["short_put"], 1)])

        if not self.call_side_closed:
            legs.extend([Leg.create(self.trade["short_call"], 1), Leg.create(self.trade["long_call"], -1)])
        
        if legs:
            self.combo_market_order(legs, 1)
            self._pending_call_side_close = not self.call_side_closed
            self._pending_put_side_close = not self.put_side_closed

        self.position_entered = False
        self.trade = None
        self.call_side_closed = False
        self.put_side_closed = False
        self._pending_call_side_close = False
        self._pending_put_side_close = False
    
    def on_order_event(self, order_event):
        """Detect when side closures fill and set flags"""
        if order_event.status == OrderStatus.FILLED:
            fill_price = order_event.fill_price
            quantity = order_event.quantity
            symbol = order_event.symbol
            self.debug(f"ORDER FILLED: {symbol} | Qty: {quantity} | Fill Price: ${fill_price:.4f}")

            if not self.trade:
                return

            # Check if this is a call side closure
            if self._pending_call_side_close and symbol in [self.trade["short_call"], self.trade["long_call"]]:
                self.call_side_closed = True
                self.debug(f"CALL SIDE CLOSED (order filled)")

            # Check if this is a put side closure
            if self._pending_put_side_close and symbol in [self.trade["short_put"], self.trade["long_put"]]:
                self.put_side_closed = True
                self.debug(f"PUT SIDE CLOSED (order filled)")

            # Reset pending flags if both legs of a side are filled
            if self.call_side_closed:
                self._pending_call_side_close = False
            if self.put_side_closed:
                self._pending_put_side_close = False

