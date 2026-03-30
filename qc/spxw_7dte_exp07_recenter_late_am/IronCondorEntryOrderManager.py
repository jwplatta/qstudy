# ruff: noqa: F403, F405

from datetime import timedelta

from AlgorithmImports import *


class IronCondorEntryOrderManager:
    """Owns the pending entry order lifecycle for the strategy."""

    def __init__(
        self,
        algorithm,
        entry_limit_offset=0.05,
        order_refresh_minutes=1,
        walk_after_unchanged_refreshes=2,
        walk_increment=0.05,
    ):
        self.algorithm = algorithm
        self.entry_limit_offset = entry_limit_offset
        self.order_refresh_minutes = order_refresh_minutes
        self.walk_after_unchanged_refreshes = walk_after_unchanged_refreshes
        self.walk_increment = walk_increment
        self.pending_entry = None

    @property
    def is_pending(self):
        return self.pending_entry is not None

    def submit_entry_order(self, call_spread, put_spread, spx_price, expiry_date):
        total_credit = call_spread["price"] + put_spread["price"]
        limit_price = self.credit_limit_price(total_credit)

        self.algorithm.debug(
            f"ENTRY: SPX={spx_price:.2f} | "
            f"PUT={put_spread['short_leg'].strike}/{put_spread['long_leg'].strike} "
            f"@ ${put_spread['price']:.2f} | "
            f"CALL={call_spread['short_leg'].strike}/{call_spread['long_leg'].strike} "
            f"@ ${call_spread['price']:.2f} | "
            f"TOTAL CREDIT=${total_credit:.2f} | LIMIT=${limit_price:.2f} | "
            f"EXPIRY={expiry_date}"
        )

        legs = [
            Leg.create(put_spread["long_leg"].symbol, 1),
            Leg.create(put_spread["short_leg"].symbol, -1),
            Leg.create(call_spread["short_leg"].symbol, -1),
            Leg.create(call_spread["long_leg"].symbol, 1),
        ]

        tickets = self.algorithm.combo_limit_order(legs, 1, limit_price)
        trade = {
            "entry_credit": round(total_credit, 2),
            "call_credit": round(call_spread["price"], 2),
            "put_credit": round(put_spread["price"], 2),
            "long_put": put_spread["long_leg"].symbol,
            "short_put": put_spread["short_leg"].symbol,
            "short_call": call_spread["short_leg"].symbol,
            "long_call": call_spread["long_leg"].symbol,
            "expiry": call_spread["short_leg"].expiry,
            "entry_spx_price": round(spx_price, 2),
            "entry_time": self.algorithm.time,
        }
        self.pending_entry = {
            "tickets": self.normalize_tickets(tickets),
            "trade": trade,
            "put_symbols": {
                "short": put_spread["short_leg"].symbol,
                "long": put_spread["long_leg"].symbol,
            },
            "call_symbols": {
                "short": call_spread["short_leg"].symbol,
                "long": call_spread["long_leg"].symbol,
            },
            "last_market_credit": round(total_credit, 2),
            "limit_price": limit_price,
            "offset": self.entry_limit_offset,
            "unchanged_refreshes": 0,
            "submitted_at": self.algorithm.time,
            "expiry_date": expiry_date,
            "entry_spx_price": round(spx_price, 2),
        }

    def manage(self):
        if not self.pending_entry:
            return

        if self.algorithm.position_entered:
            self.pending_entry = None
            return

        if not self.algorithm.is_regular_market_hours():
            self.cancel("outside market hours")
            return

        if self.algorithm.time - self.pending_entry["submitted_at"] < timedelta(
            minutes=self.order_refresh_minutes
        ):
            return

        if self.has_partial_fill():
            return

        market_credit = self.current_market_credit()
        if market_credit is None:
            self.cancel("missing price data")
            return

        market_credit = round(market_credit, 2)
        if market_credit == self.pending_entry["last_market_credit"]:
            self.pending_entry["unchanged_refreshes"] += 1
            if (
                self.pending_entry["unchanged_refreshes"]
                < self.walk_after_unchanged_refreshes
            ):
                self.pending_entry["submitted_at"] = self.algorithm.time
                return

            new_offset = max(0.0, self.pending_entry["offset"] - self.walk_increment)
            self.replace(market_credit, new_offset, "walking entry limit")
            return

        self.replace(market_credit, self.entry_limit_offset, "repricing entry")

    def handle_order_event(self, order_event):
        if not self.pending_entry:
            return None

        pending_order_ids = {ticket.order_id for ticket in self.pending_entry["tickets"]}
        if order_event.order_id not in pending_order_ids:
            return None

        if all(ticket.status == OrderStatus.FILLED for ticket in self.pending_entry["tickets"]):
            trade = self.pending_entry["trade"]
            self.pending_entry = None
            self.algorithm.debug(
                f"ENTRY FILLED: credit=${trade['entry_credit']:.2f} | "
                f"expiry={trade['expiry'].date()}"
            )
            return trade

        return None

    def replace(self, market_credit, offset, reason):
        existing_trade = self.pending_entry["trade"]
        put_symbols = self.pending_entry["put_symbols"]
        call_symbols = self.pending_entry["call_symbols"]
        expiry_date = self.pending_entry["expiry_date"]
        entry_spx_price = self.pending_entry["entry_spx_price"]

        self.cancel(reason)

        refreshed_trade = dict(existing_trade)
        refreshed_trade["entry_credit"] = market_credit
        refreshed_trade["call_credit"] = round(
            self.vertical_credit(call_symbols["short"], call_symbols["long"]), 2
        )
        refreshed_trade["put_credit"] = round(
            self.vertical_credit(put_symbols["short"], put_symbols["long"]), 2
        )
        refreshed_trade["entry_spx_price"] = entry_spx_price
        refreshed_trade["entry_time"] = self.algorithm.time

        limit_price = self.credit_limit_price(market_credit, offset)
        legs = [
            Leg.create(put_symbols["long"], 1),
            Leg.create(put_symbols["short"], -1),
            Leg.create(call_symbols["short"], -1),
            Leg.create(call_symbols["long"], 1),
        ]
        tickets = self.algorithm.combo_limit_order(legs, 1, limit_price)
        self.pending_entry = {
            "tickets": self.normalize_tickets(tickets),
            "trade": refreshed_trade,
            "put_symbols": put_symbols,
            "call_symbols": call_symbols,
            "last_market_credit": market_credit,
            "limit_price": limit_price,
            "offset": offset,
            "unchanged_refreshes": 0,
            "submitted_at": self.algorithm.time,
            "expiry_date": expiry_date,
            "entry_spx_price": entry_spx_price,
        }

    def cancel(self, reason):
        if not self.pending_entry:
            return

        for ticket in self.pending_entry["tickets"]:
            if ticket.status in (
                OrderStatus.NEW,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
            ):
                self.algorithm.transactions.cancel_order(ticket.order_id, reason)

        self.pending_entry = None

    def has_partial_fill(self):
        if not self.pending_entry:
            return False
        for ticket in self.pending_entry["tickets"]:
            if ticket.status == OrderStatus.PARTIALLY_FILLED:
                return True
        return False

    def current_market_credit(self):
        if not self.pending_entry:
            return None

        put_symbols = self.pending_entry["put_symbols"]
        call_symbols = self.pending_entry["call_symbols"]
        put_credit = self.vertical_credit(put_symbols["short"], put_symbols["long"])
        call_credit = self.vertical_credit(call_symbols["short"], call_symbols["long"])
        if put_credit is None or call_credit is None:
            return None
        return put_credit + call_credit

    def vertical_credit(self, short_symbol, long_symbol):
        short_bid = self.algorithm.securities[short_symbol].bid_price
        long_ask = self.algorithm.securities[long_symbol].ask_price
        if short_bid is None or long_ask is None:
            return None
        return short_bid - long_ask

    def credit_limit_price(self, market_credit, offset=None):
        if offset is None:
            offset = self.entry_limit_offset
        return round(max(0.05, market_credit - offset), 2)

    def normalize_tickets(self, tickets):
        if isinstance(tickets, list):
            return tickets
        return [tickets]
