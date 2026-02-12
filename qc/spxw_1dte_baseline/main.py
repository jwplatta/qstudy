from datetime import timedelta

from AlgorithmImports import *
from event_dates import get_event_dates
from IronCondorFinder import IronCondorFinder


class Spxw1dteBaseline(QCAlgorithm):
    """
    Baseline SPXW 1DTE Iron Condor Strategy

    - Enters 1DTE iron condors at 3:55pm ET
    - Skips entries when next trading day is an event date (FOMC, CPI, employment)
    - Exits at 60% profit, -3x loss, or on 0DTE day (12pm if profitable, 1pm forced)
    - No rolling or adjustments (simple baseline)
    """

    def initialize(self):
        self.set_start_date(2022, 4, 1)
        self.set_end_date(2025, 12, 31)
        self.set_cash(50000)

        # Brokerage model for realistic fills
        self.set_brokerage_model(BrokerageName.CHARLES_SCHWAB, AccountType.MARGIN)

        # Enable price seeding for options
        self.settings.seed_initial_prices = True

        # Add SPX index
        self.spx = self.add_index("SPX", Resolution.MINUTE).symbol

        # Add SPXW options (weekly)
        self.option = self.add_index_option(self.spx, "SPXW", Resolution.MINUTE)
        self.option.set_filter(lambda x: x.include_weeklys().expiration(0, 2).strikes(-100, 100))
        self.spxw = self.option.symbol

        # Load event dates (dates we don't want positions to expire on)
        self.event_dates = self.load_event_dates()

        # Configure iron condor finder
        self.iron_condor_finder = IronCondorFinder(
            spread_width=20,
            min_credit=1.05,
            max_credit=1.45,
            max_call_delta=0.08,
            min_call_delta=0.03,
            max_put_delta=0.10,
            min_put_delta=0.03,
            max_total_delta=0.18,
            credit_balance_ratio=0.6,
            delta_ratio=0.6,
            max_tweak_attempts=100,
        )

        # Position tracking
        self.trade = None
        self.position_entered = False

        # Schedule entry at 3:55pm ET
        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.at(15, 55, TimeZones.NEW_YORK),
            self.check_entry,
        )

        # Monitor every 5 minutes
        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.every(timedelta(minutes=5)),
            self.monitor_positions,
        )

    def load_event_dates(self):
        """Load event dates from event_dates.py"""
        try:
            event_dates = get_event_dates()
            self.debug(f"Loaded {len(event_dates)} event dates")
            return event_dates
        except Exception as e:
            self.debug(f"Warning: Could not load event dates: {e}")
            return set()

    def is_expiration_on_event_date(self, expiry_date):
        """Check if expiry_date is an event date"""
        return expiry_date in self.event_dates

    def check_entry(self):
        """
        Called at 3:55pm ET to check if we should enter a new position.
        Skips entry if:
        - Already in a position
        - Next available expiration is an event date (avoid expiring on events)
        - No valid iron condor found
        """
        if self.is_warming_up:
            return

        if self.position_entered:
            return

        # Get option chain
        chain = self.current_slice.option_chains.get(self.spxw)
        if not chain:
            self.debug("No option chain available")
            return

        # Find the nearest expiration date (SPXW expires Mon/Wed/Fri, not every day)
        # Group by expiration and find the soonest one
        expiries = sorted(set(x.expiry.date() for x in chain))
        if not expiries:
            self.debug("No expiries found in option chain")
            return

        # Get the nearest expiration date
        nearest_expiry = expiries[0]

        # Skip if nearest expiration is an event date
        if self.is_expiration_on_event_date(nearest_expiry):
            self.debug(f"Skipping entry - nearest expiry {nearest_expiry} is event date")
            return

        # Filter for options expiring on nearest date
        contracts = [x for x in chain if x.expiry.date() == nearest_expiry]

        if not contracts:
            self.debug(f"No contracts expiring on {nearest_expiry}")
            return

        # Get current SPX price
        spx_price = self.securities[self.spx].price

        # Use IronCondorFinder to find valid iron condor
        result = self.iron_condor_finder.find_iron_condor(contracts, spx_price)

        if result:
            call_spread, put_spread, tweak_count = result
            self.enter_position(call_spread, put_spread, spx_price, nearest_expiry)
        else:
            self.debug("No valid iron condor found - skipping entry")

    def enter_position(self, call_spread, put_spread, spx_price, expiry_date):
        """Enter iron condor position"""
        total_credit = call_spread["price"] + put_spread["price"]

        # Log entry details
        self.debug(
            f"ENTRY: SPX={spx_price:.2f} | "
            f"PUT={put_spread['short_leg'].strike}/{put_spread['long_leg'].strike} @ ${put_spread['price']:.2f} | "
            f"CALL={call_spread['short_leg'].strike}/{call_spread['long_leg'].strike} @ ${call_spread['price']:.2f} | "
            f"TOTAL CREDIT=${total_credit:.2f} | "
            f"EXPIRY={expiry_date}"
        )

        legs = [
            Leg.create(put_spread["long_leg"].symbol, 1),  # Buy long put
            Leg.create(put_spread["short_leg"].symbol, -1),  # Sell short put
            Leg.create(call_spread["short_leg"].symbol, -1),  # Sell short call
            Leg.create(call_spread["long_leg"].symbol, 1),  # Buy long call
        ]

        # Execute as combo market order
        self.combo_market_order(legs, 1)

        # Store trade details
        self.trade = {
            "entry_credit": round(total_credit, 2),
            "call_credit": round(call_spread["price"], 2),
            "put_credit": round(put_spread["price"], 2),
            "profit_target": round(total_credit * 0.95, 2),
            "max_loss": round(total_credit * -3.0, 2),
            "long_put": put_spread["long_leg"].symbol,
            "short_put": put_spread["short_leg"].symbol,
            "short_call": call_spread["short_leg"].symbol,
            "long_call": call_spread["long_leg"].symbol,
            "expiry": call_spread["short_leg"].expiry,
            "entry_spx_price": round(spx_price, 2),
            "entry_time": self.time,
        }

        self.position_entered = True

    def monitor_positions(self):
        """
        Monitor position every 5 minutes and check exit conditions.
        Exit priority:
        1. Profit target (60%)
        2. Max loss (-3x)
        3. 0DTE 12pm if profitable
        4. 0DTE 1pm forced exit
        """
        if not self.position_entered or not self.trade:
            return

        if self.is_warming_up:
            return

        current_pnl = self.calculate_pnl()

        # Exit 1: Profit target
        if current_pnl >= self.trade["profit_target"]:
            self.exit_position(f"Profit target reached: ${current_pnl:.2f}")
            return

        # Exit 2: Max loss
        if current_pnl <= self.trade["max_loss"]:
            self.exit_position(f"Max loss reached: ${current_pnl:.2f}")
            return

        # Exit 3 & 4: 0DTE special rules
        if self.is_0dte():
            current_hour = self.time.hour

            # Exit after 12pm if profitable
            if current_hour >= 12 and current_pnl > 0:
                self.exit_position(f"0DTE 12pm+ profitable exit: ${current_pnl:.2f}")
                return

            # Exit after 1pm regardless
            if current_hour >= 13:
                self.exit_position(f"0DTE 1pm+ forced exit: ${current_pnl:.2f}")
                return

    def calculate_pnl(self):
        """Calculate current P&L of iron condor"""
        # Get current prices for all legs
        short_put_price = self.securities[self.trade["short_put"]].price
        long_put_price = self.securities[self.trade["long_put"]].price
        short_call_price = self.securities[self.trade["short_call"]].price
        long_call_price = self.securities[self.trade["long_call"]].price

        # Calculate exit cost (what we'd pay to close)
        put_exit_cost = short_put_price - long_put_price
        call_exit_cost = short_call_price - long_call_price
        total_exit_cost = put_exit_cost + call_exit_cost

        # P&L = credit received - cost to exit
        pnl = self.trade["entry_credit"] - total_exit_cost

        return pnl

    def exit_position(self, reason):
        """Exit entire iron condor position"""
        if not self.position_entered:
            return

        current_pnl = self.calculate_pnl()

        self.debug(
            f"EXIT: {reason} | Entry Credit: ${self.trade['entry_credit']:.2f} | P&L: ${current_pnl:.2f}"
        )

        # Create all 4 legs to close (reverse of entry)
        legs = [
            Leg.create(self.trade["short_put"], 1),  # Buy back short put
            Leg.create(self.trade["long_put"], -1),  # Sell long put
            Leg.create(self.trade["short_call"], 1),  # Buy back short call
            Leg.create(self.trade["long_call"], -1),  # Sell long call
        ]

        # Execute as combo market order
        self.combo_market_order(legs, 1)

        # Reset state
        self.position_entered = False
        self.trade = None

    def is_0dte(self):
        """Check if position is 0DTE (expires today)"""
        return self.trade and self.trade["expiry"].date() == self.time.date()
