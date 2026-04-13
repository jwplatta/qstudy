# ruff: noqa: F403, F405

from datetime import timedelta

from AlgorithmImports import *
from event_dates import get_event_dates
from IronCondorEntryOrderManager import IronCondorEntryOrderManager
from IronCondorFinder import IronCondorFinder


class Spxw7dteBaseline(QCAlgorithm):
    STRATEGY_NAME = "baseline"
    DELTA_THRESHOLD = 0.25

    START_DATE = (2022, 1, 1)
    END_DATE = (2025, 12, 31)
    INITIAL_CASH = 50000
    MIN_DTE = 6
    MAX_DTE = 10
    MONITOR_MIN_DTE = 0
    MONITOR_MAX_DTE = 14

    ENTRY_LIMIT_OFFSET = 0.05
    ORDER_REFRESH_MINUTES = 1
    WALK_AFTER_UNCHANGED_REFRESHES = 2
    WALK_INCREMENT = 0.05

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
        except Exception as error:
            raise Exception(f"Error loading event dates: {error}")

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
        if self.position_entered or self.entry_order_manager.is_pending:
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

        contracts = [contract for contract in chain if contract.expiry.date() == target_expiry]
        if not contracts:
            next_expiry = self.next_available_chain_expiry(target_expiry, chain)
            if next_expiry and reason == "baseline entry":
                target_expiry = next_expiry
                contracts = [
                    contract for contract in chain if contract.expiry.date() == target_expiry
                ]

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

        call_spread, put_spread, _ = result
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

    def monitor_position(self):
        if self.is_warming_up:
            return
        if not self.position_entered or not self.trade:
            return
        if self.entry_order_manager.is_pending:
            return
        if not self.is_regular_market_hours():
            return

        trigger = self.evaluate_trigger()
        if not trigger or self.trade.get("trigger_logged"):
            return

        self.log_trigger_event(trigger)
        self.trade["trigger_logged"] = True

    def evaluate_trigger(self):
        chain = self.current_slice.option_chains.get(self.spxw)
        if not chain:
            return None

        call_contract = self.contract_by_symbol(chain, self.trade["short_call"])
        put_contract = self.contract_by_symbol(chain, self.trade["short_put"])
        if call_contract is None or put_contract is None:
            return None

        spx_price = self.securities[self.spx].price
        call_delta = self.contract_delta(call_contract)
        put_delta = self.contract_delta(put_contract)
        call_breach = spx_price >= call_contract.strike
        put_breach = spx_price <= put_contract.strike

        call_trigger = call_breach or (
            call_delta is not None and call_delta >= self.DELTA_THRESHOLD
        )
        put_trigger = put_breach or (put_delta is not None and put_delta >= self.DELTA_THRESHOLD)
        if not call_trigger and not put_trigger:
            return None

        if call_trigger and not put_trigger:
            tested_side = "call"
        elif put_trigger and not call_trigger:
            tested_side = "put"
        else:
            tested_side = "call" if (call_delta or 0.0) >= (put_delta or 0.0) else "put"

        side_delta = call_delta if tested_side == "call" else put_delta
        side_breach = call_breach if tested_side == "call" else put_breach
        distance_to_strike = (
            round(call_contract.strike - spx_price, 2)
            if tested_side == "call"
            else round(spx_price - put_contract.strike, 2)
        )

        reasons = []
        if side_breach:
            reasons.append(f"{tested_side} short strike breached")
        if side_delta is not None and side_delta >= self.DELTA_THRESHOLD:
            reasons.append(f"{tested_side} delta {side_delta:.3f} >= {self.DELTA_THRESHOLD:.2f}")

        return {
            "tested_side": tested_side,
            "reason": " | ".join(reasons),
            "spx_price": round(spx_price, 2),
            "call_delta": call_delta,
            "put_delta": put_delta,
            "distance_to_strike": distance_to_strike,
        }

    def log_trigger_event(self, trigger):
        days_since_entry = round(
            (self.time - self.trade["entry_time"]).total_seconds() / 86400.0, 4
        )
        dte_at_trigger = (self.trade["expiry"].date() - self.time.date()).days
        unrealized_pnl = self.current_unrealized_pnl()
        movement_type = self.classify_price_movement(
            trigger["tested_side"], trigger["spx_price"]
        )

        self.debug(
            "TRIGGER EVENT | "
            f"strategy={self.STRATEGY_NAME} | "
            f"time={self.time.isoformat()} | "
            f"days_since_entry={days_since_entry:.4f} | "
            f"dte={dte_at_trigger} | "
            f"tested_side={trigger['tested_side']} | "
            f"distance_to_strike={trigger['distance_to_strike']:.2f} | "
            f"call_delta={self.format_optional(trigger['call_delta'])} | "
            f"put_delta={self.format_optional(trigger['put_delta'])} | "
            f"unrealized_pnl={self.format_optional(unrealized_pnl, money=True)} | "
            f"movement={movement_type} | "
            f"reason={trigger['reason']}"
        )

    def current_unrealized_pnl(self):
        market_debit = self.current_trade_debit()
        if market_debit is None:
            return None
        return round(self.trade["entry_credit"] - market_debit, 2)

    def current_trade_debit(self):
        put_debit = self.vertical_debit(self.trade["short_put"], self.trade["long_put"])
        call_debit = self.vertical_debit(self.trade["short_call"], self.trade["long_call"])
        if put_debit is None or call_debit is None:
            return None
        return round(put_debit + call_debit, 2)

    def vertical_debit(self, short_symbol, long_symbol):
        short_ask = self.securities[short_symbol].ask_price
        long_bid = self.securities[long_symbol].bid_price
        if short_ask is None or long_bid is None:
            return None
        return short_ask - long_bid

    def classify_price_movement(self, tested_side, spx_price):
        move = round(spx_price - self.trade["entry_spx_price"], 2)
        if abs(move) < 5:
            return "flat"
        if tested_side == "call":
            return "up_move_into_call_side" if move > 0 else "reversal_after_drop"
        return "down_move_into_put_side" if move < 0 else "reversal_after_rally"

    def contract_by_symbol(self, chain, symbol):
        return next((contract for contract in chain if contract.symbol == symbol), None)

    def contract_delta(self, contract):
        greeks = getattr(contract, "greeks", None)
        if greeks is None or greeks.delta is None:
            return None
        try:
            return abs(float(greeks.delta))
        except (TypeError, ValueError):
            return None

    def format_optional(self, value, money=False):
        if value is None:
            return "n/a"
        return f"${value:.2f}" if money else f"{value:.3f}"

    def on_order_event(self, order_event):
        filled_trade = self.entry_order_manager.handle_order_event(order_event)
        if not filled_trade:
            return

        self.trade = filled_trade
        self.position_entered = True
        self.initialize_trade_state()

    def initialize_trade_state(self):
        if not self.trade:
            return

        self.trade.setdefault(
            "active_legs",
            [
                self.trade["long_put"],
                self.trade["short_put"],
                self.trade["short_call"],
                self.trade["long_call"],
            ],
        )
        self.trade["trigger_logged"] = False

    def reset_after_expiry(self):
        if not self.trade or not self.position_entered:
            return
        if self.time.date() <= self.trade["expiry"].date():
            return

        still_invested = any(
            self.portfolio[symbol].invested
            for symbol in self.trade.get("active_legs", [])
            if symbol is not None
        )
        if still_invested or self.portfolio.invested:
            return

        self.debug(
            "EXPIRED: clearing state for position entered "
            f"{self.trade['entry_time']} and expired {self.trade['expiry'].date()}"
        )
        self.trade = None
        self.position_entered = False
