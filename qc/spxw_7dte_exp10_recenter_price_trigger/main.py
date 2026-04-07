# ruff: noqa: F403, F405

from datetime import timedelta

from AlgorithmImports import *
from event_dates import get_event_dates
from IronCondorEntryOrderManager import IronCondorEntryOrderManager
from IronCondorExitOrderManager import IronCondorExitOrderManager
from IronCondorFinder import IronCondorFinder
from IronCondorRepairManager import IronCondorRepairManager

"""
Recenter in these experiments means:
close the current condor, then re-enter a new condor on the same expiry.
This is not an atomic adjustment and does not enforce a small net credit/debit.
"""


class Spxw7dteRepairExperiment(QCAlgorithm):
    """
    SPXW 7DTE iron condor baseline with one isolated repair rule.

    The baseline entry logic, pricing, and iron condor selection stay fixed.
    Each project folder changes only the repair trigger bucket and repair action.
    """

    EXPERIMENT_ID = 10
    REFERENCE_EXPERIMENT = 0
    EXPERIMENT_NAME = "exp10_recenter_price_trigger"
    ACTION = "recenter"
    TRIGGER_MODE = "price_multiple"
    DELTA_THRESHOLD = None
    MARK_MULTIPLE = 2.5
    DTE_BUCKET = (3, 5)
    WINDOW_START = (9, 30)
    WINDOW_END = (16, 0)

    START_DATE = (2022, 1, 1)
    END_DATE = (2025, 12, 31)
    INITIAL_CASH = 50000
    MIN_DTE = 6
    MAX_DTE = 10
    ENTRY_LIMIT_OFFSET = 0.05
    ORDER_REFRESH_MINUTES = 1
    WALK_AFTER_UNCHANGED_REFRESHES = 2
    WALK_INCREMENT = 0.05

    EXIT_LIMIT_OFFSET = 0.15
    EXIT_WALK_AFTER_UNCHANGED_REFRESHES = 1
    EXIT_WALK_INCREMENT = 0.10
    MONITOR_MIN_DTE = 0
    MONITOR_MAX_DTE = 14

    def initialize(self):
        self.set_start_date(*self.START_DATE)
        self.set_end_date(*self.END_DATE)
        self.set_cash(self.INITIAL_CASH)

        self.set_brokerage_model(BrokerageName.CHARLES_SCHWAB, AccountType.MARGIN)
        self.settings.seed_initial_prices = True
        self.spx = self.add_index("SPX", Resolution.MINUTE).symbol

        self.option = self.add_index_option(
            self.spx, "SPXW", resolution=Resolution.MINUTE, fill_forward=True
        )
        self.option.set_filter(
            lambda universe: universe.expiration(
                self.MONITOR_MIN_DTE, self.MONITOR_MAX_DTE
            ).weeklys_only()
        )
        self.spxw = self.option.symbol
        self.event_dates = self.load_event_dates()

        self.iron_condor_finder = IronCondorFinder(
            spread_width=20,
            min_credit=1.2,
            max_credit=1.8,
            max_call_delta=0.08,
            min_call_delta=0.02,
            max_put_delta=0.10,
            min_put_delta=0.02,
            max_total_delta=0.18,
            credit_balance_ratio=0.7,
            delta_ratio=0.7,
            max_tweak_attempts=100,
        )

        self.trade = None
        self.position_entered = False
        self.blocked_entry_date = None
        self.entry_retry_seconds = 20

        self.entry_order_manager = IronCondorEntryOrderManager(
            self,
            entry_limit_offset=self.ENTRY_LIMIT_OFFSET,
            order_refresh_minutes=self.ORDER_REFRESH_MINUTES,
            walk_after_unchanged_refreshes=self.WALK_AFTER_UNCHANGED_REFRESHES,
            walk_increment=self.WALK_INCREMENT,
        )
        self.exit_order_manager = IronCondorExitOrderManager(
            self,
            exit_limit_offset=self.EXIT_LIMIT_OFFSET,
            order_refresh_minutes=self.ORDER_REFRESH_MINUTES,
            walk_after_unchanged_refreshes=self.EXIT_WALK_AFTER_UNCHANGED_REFRESHES,
            walk_increment=self.EXIT_WALK_INCREMENT,
        )
        self.repair_manager = IronCondorRepairManager(self, action=self.ACTION)

        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.at(15, 0, TimeZones.NEW_YORK),
            self.check_entry,
        )

        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.every(timedelta(minutes=1)),
            self.manage_pending_orders,
        )

        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.every(timedelta(minutes=1)),
            self.monitor_position,
        )

    def load_event_dates(self):
        try:
            return get_event_dates()
        except Exception as e:
            raise Exception(f"Error loading event dates: {e}")

    def is_expiration_on_event_date(self, expiry_date):
        return expiry_date in self.event_dates

    def is_day_after_event_date(self, expiry_date):
        return (expiry_date - timedelta(days=1)) in self.event_dates

    def is_valid_expiry_candidate(self, expiry_date):
        if expiry_date.weekday() >= 5:
            return False

        if not self.securities[self.spx].exchange.date_is_open(expiry_date):
            return False

        if self.is_expiration_on_event_date(expiry_date):
            return False

        if self.is_day_after_event_date(expiry_date):
            return False

        return True

    def is_regular_market_hours(self):
        if not self.securities[self.spx].exchange.date_is_open(self.time.date()):
            return False

        minutes_since_midnight = (self.time.hour * 60) + self.time.minute
        market_open_minutes = 9 * 60 + 30
        market_close_minutes = 16 * 60
        return market_open_minutes <= minutes_since_midnight < market_close_minutes

    def next_valid_expiry(self, from_date):
        six_dte = from_date + timedelta(days=6)
        seven_dte = from_date + timedelta(days=7)
        eight_dte = from_date + timedelta(days=8)

        preferred_expiries = [seven_dte]
        if self.is_expiration_on_event_date(seven_dte):
            preferred_expiries = [six_dte, eight_dte, seven_dte]
        elif self.is_day_after_event_date(seven_dte):
            preferred_expiries = [eight_dte, six_dte, seven_dte]
        else:
            preferred_expiries = [seven_dte, six_dte, eight_dte]

        for expiry_date in preferred_expiries:
            days_out = (expiry_date - from_date).days
            if not self.MIN_DTE <= days_out <= self.MAX_DTE:
                continue

            if self.is_valid_expiry_candidate(expiry_date):
                return expiry_date

        return None

    def next_available_chain_expiry(self, after_expiry, chain):
        available_expiries = sorted({contract.expiry.date() for contract in chain})
        for expiry_date in available_expiries:
            if expiry_date <= after_expiry:
                continue
            if self.is_valid_expiry_candidate(expiry_date):
                return expiry_date
        return None

    def check_entry(self):
        current_date = self.time.date()

        if self.is_warming_up:
            return

        self.reset_after_expiry()
        if (
            self.position_entered
            or self.entry_order_manager.is_pending
            or self.exit_order_manager.is_pending
            or self.repair_manager.is_pending
        ):
            return

        if self.blocked_entry_date == current_date:
            return

        if self.time.hour >= 16:
            return

        target_expiry = self.next_valid_expiry(current_date)
        if not target_expiry:
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - "
                f"No valid expiry found between {self.MIN_DTE} and {self.MAX_DTE} DTE"
            )
            return

        self.submit_entry_for_expiry(target_expiry, "baseline entry")

    def submit_entry_for_expiry(self, target_expiry, reason):
        chain = self.current_slice.option_chains.get(self.spxw)
        if not chain:
            if reason == "baseline entry":
                self.schedule_retry()
            return False

        contracts = [x for x in chain if x.expiry.date() == target_expiry]
        if not contracts:
            next_expiry = self.next_available_chain_expiry(target_expiry, chain)
            if next_expiry and reason == "baseline entry":
                target_expiry = next_expiry
                contracts = [x for x in chain if x.expiry.date() == target_expiry]

        if not contracts:
            if reason == "baseline entry":
                self.schedule_retry()
            return False

        spx_price = self.securities[self.spx].price
        result = self.iron_condor_finder.find_iron_condor(contracts, spx_price)
        if not result:
            if reason == "baseline entry":
                self.schedule_retry()
            return False

        call_spread, put_spread, tweak_count = result
        self.entry_order_manager.submit_entry_order(
            call_spread, put_spread, spx_price, target_expiry
        )
        return True

    def schedule_retry(self):
        if self.time.hour < 16:
            retry_time = self.time + timedelta(seconds=self.entry_retry_seconds)
            self.schedule.on(
                self.date_rules.on(retry_time.year, retry_time.month, retry_time.day),
                self.time_rules.at(retry_time.hour, retry_time.minute, TimeZones.NEW_YORK),
                self.check_entry,
            )

    def manage_pending_orders(self):
        self.entry_order_manager.manage()
        self.exit_order_manager.manage()

    def monitor_position(self):
        if self.is_warming_up:
            return

        if self.repair_manager.manage():
            return

        if not self.position_entered or not self.trade:
            return

        if self.entry_order_manager.is_pending or self.exit_order_manager.is_pending:
            return

        if not self.is_regular_market_hours():
            return

        trigger_reason = self.repair_trigger_reason()
        if not trigger_reason:
            return

        self.repair_manager.start(trigger_reason)

    def repair_trigger_reason(self):
        days_to_expiry = (self.trade["expiry"].date() - self.time.date()).days
        if not self.DTE_BUCKET[0] <= days_to_expiry <= self.DTE_BUCKET[1]:
            return None

        if not self.in_time_window():
            return None

        current_debit = self.exit_order_manager.current_market_debit()
        if current_debit is None:
            return None

        threshold = self.trade["entry_credit"] * self.MARK_MULTIPLE
        if current_debit >= threshold:
            return (
                f"Condor mark ${current_debit:.2f} >= "
                f"{self.MARK_MULTIPLE:.1f}x entry credit ${self.trade['entry_credit']:.2f}"
            )

        return None

    def in_time_window(self):
        current_minutes = (self.time.hour * 60) + self.time.minute
        start_minutes = (self.WINDOW_START[0] * 60) + self.WINDOW_START[1]
        end_minutes = (self.WINDOW_END[0] * 60) + self.WINDOW_END[1]
        return start_minutes <= current_minutes < end_minutes

    def on_order_event(self, order_event):
        self.repair_manager.handle_order_event(order_event)

    def reset_after_expiry(self):
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
        self.repair_manager.clear()
