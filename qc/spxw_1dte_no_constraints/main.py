# ruff: noqa: F403, F405

from datetime import timedelta

from AlgorithmImports import *
from event_dates import get_event_dates
from IronCondorFinder import IronCondorFinder


class Spxw1dteNoConstraints(QCAlgorithm):
    """
    SPXW 1DTE iron condor strategy without exit constraints.

    - Enters a 1DTE iron condor near the close and retries until 4:00pm ET
    - Skips entries when the next trading day is an event date
    - Does not use stop losses, profit targets, or forced exits
    - Holds each position until the contracts expire
    """

    START_DATE = (2022, 4, 1)
    END_DATE = (2026, 3, 26)
    INITIAL_CASH = 50000

    def initialize(self):
        self.set_start_date(*self.START_DATE)
        self.set_end_date(*self.END_DATE)
        self.set_cash(self.INITIAL_CASH)

        self.set_brokerage_model(BrokerageName.CHARLES_SCHWAB, AccountType.MARGIN)
        self.settings.seed_initial_prices = True
        self.spx = self.add_index("SPX", Resolution.HOUR).symbol

        self.option = self.add_index_option(
            self.spx, "SPXW", resolution=Resolution.HOUR, fill_forward=True
        )
        self.option.set_filter(lambda universe: universe.expiration(0, 7).weeklys_only())
        self.spxw = self.option.symbol
        self.event_dates = self.load_event_dates()

        self.iron_condor_finder = IronCondorFinder(
            spread_width=20,
            min_credit=1.05,
            max_credit=1.45,
            max_call_delta=0.08,
            min_call_delta=0.02,
            max_put_delta=0.10,
            min_put_delta=0.02,
            max_total_delta=0.18,
            credit_balance_ratio=0.6,
            delta_ratio=0.6,
            max_tweak_attempts=100,
        )

        self.trade = None
        self.position_entered = False
        self.entry_retry_seconds = 20

        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.at(15, 50, TimeZones.NEW_YORK),
            self.check_entry,
        )

    def load_event_dates(self):
        try:
            return get_event_dates()
        except Exception as e:
            raise Exception(f"Error loading event dates: {e}")

    def is_expiration_on_event_date(self, expiry_date):
        return expiry_date in self.event_dates

    def next_valid_expiry(self, from_date):
        """Return the next trading-day expiry after the current session."""
        max_days_ahead = 10
        candidate_date = from_date + timedelta(days=1)

        for _ in range(max_days_ahead):
            if candidate_date.weekday() >= 5:
                candidate_date += timedelta(days=1)
                continue

            if not self.securities[self.spx].exchange.date_is_open(candidate_date):
                candidate_date += timedelta(days=1)
                continue

            if self.is_expiration_on_event_date(candidate_date):
                return None

            return candidate_date

        return None

    def check_entry(self):
        current_date = self.time.date()

        if self.is_warming_up:
            return

        self.reset_after_expiry()
        if self.position_entered:
            return

        if self.time.hour >= 16:
            return

        target_expiry = self.next_valid_expiry(current_date)
        if not target_expiry:
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - "
                "No valid expiry found in next 10 days"
            )
            return

        chain = self.current_slice.option_chains.get(self.spxw)
        if not chain:
            self.debug(f"{current_date} {self.time.strftime('%H:%M')} - No option chain available")
            self.schedule_retry()
            return

        contracts = [x for x in chain if x.expiry.date() == target_expiry]
        if not contracts:
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - "
                f"No contracts expiring on {target_expiry}"
            )
            self.schedule_retry()
            return

        spx_price = self.securities[self.spx].price
        result = self.iron_condor_finder.find_iron_condor(contracts, spx_price)

        if result:
            call_spread, put_spread, tweak_count = result
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - "
                f"Found valid iron condor after {tweak_count} tweaks"
            )
            self.enter_position(call_spread, put_spread, spx_price, target_expiry)
            return

        self.debug(
            f"{current_date} {self.time.strftime('%H:%M')} - "
            "No valid iron condor found, "
            f"will retry in {self.entry_retry_seconds}s"
        )
        self.schedule_retry()

    def schedule_retry(self):
        if self.time.hour < 16:
            retry_time = self.time + timedelta(seconds=self.entry_retry_seconds)
            self.schedule.on(
                self.date_rules.on(retry_time.year, retry_time.month, retry_time.day),
                self.time_rules.at(retry_time.hour, retry_time.minute, TimeZones.NEW_YORK),
                self.check_entry,
            )

    def enter_position(self, call_spread, put_spread, spx_price, expiry_date):
        total_credit = call_spread["price"] + put_spread["price"]

        self.debug(
            f"ENTRY: SPX={spx_price:.2f} | "
            f"PUT={put_spread['short_leg'].strike}/{put_spread['long_leg'].strike} "
            f"@ ${put_spread['price']:.2f} | "
            f"CALL={call_spread['short_leg'].strike}/{call_spread['long_leg'].strike} "
            f"@ ${call_spread['price']:.2f} | "
            f"TOTAL CREDIT=${total_credit:.2f} | "
            f"EXPIRY={expiry_date}"
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
            "long_put": put_spread["long_leg"].symbol,
            "short_put": put_spread["short_leg"].symbol,
            "short_call": call_spread["short_leg"].symbol,
            "long_call": call_spread["long_leg"].symbol,
            "expiry": call_spread["short_leg"].expiry,
            "entry_spx_price": round(spx_price, 2),
            "entry_time": self.time,
        }
        self.position_entered = True

    def reset_after_expiry(self):
        """
        Clear state after the prior trade has naturally expired and the portfolio is flat.
        """
        if not self.trade or not self.position_entered:
            return

        if self.time.date() <= self.trade["expiry"].date():
            return

        tracked_symbols = [
            self.trade["long_put"],
            self.trade["short_put"],
            self.trade["short_call"],
            self.trade["long_call"],
        ]
        still_invested = any(self.portfolio[symbol].invested for symbol in tracked_symbols)
        if still_invested or self.portfolio.invested:
            return

        self.debug(
            "EXPIRED: clearing state for position entered "
            f"{self.trade['entry_time']} and expired {self.trade['expiry'].date()}"
        )
        self.trade = None
        self.position_entered = False
