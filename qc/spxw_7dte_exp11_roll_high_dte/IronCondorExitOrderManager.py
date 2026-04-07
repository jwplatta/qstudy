# ruff: noqa: F403, F405

from datetime import timedelta

from AlgorithmImports import *


class IronCondorExitOrderManager:
    """Owns the pending exit order lifecycle for the strategy."""

    def __init__(
        self,
        algorithm,
        exit_limit_offset=0.05,
        order_refresh_minutes=1,
        walk_after_unchanged_refreshes=2,
        walk_increment=0.05,
    ):
        self.algorithm = algorithm
        self.exit_limit_offset = exit_limit_offset
        self.order_refresh_minutes = order_refresh_minutes
        self.walk_after_unchanged_refreshes = walk_after_unchanged_refreshes
        self.walk_increment = walk_increment
        self.pending_exit = None

    @property
    def is_pending(self):
        return self.pending_exit is not None

    def submit_exit_order(self, trade, market_debit, reason, limit_offset=None):
        if limit_offset is None:
            limit_offset = self.exit_limit_offset

        limit_price = self.debit_limit_price(market_debit, limit_offset)
        current_pnl = round(trade["entry_credit"] - market_debit, 2)

        self.algorithm.debug(
            f"EXIT SUBMITTED: {reason} | "
            f"entry_credit=${trade['entry_credit']:.2f} | "
            f"exit_debit=${market_debit:.2f} | "
            f"limit=${limit_price:.2f} | "
            f"pnl=${current_pnl:.2f}"
        )

        legs = self.exit_legs(trade)
        tickets = self.algorithm.combo_limit_order(legs, 1, limit_price)
        self.pending_exit = {
            "tickets": self.normalize_tickets(tickets),
            "trade": dict(trade),
            "reason": reason,
            "last_market_debit": round(market_debit, 2),
            "filled_net_debit": 0.0,
            "limit_price": limit_price,
            "offset": limit_offset,
            "unchanged_refreshes": 0,
            "submitted_at": self.algorithm.time,
        }

    def manage(self):
        if not self.pending_exit:
            return

        if not self.algorithm.is_regular_market_hours():
            self.cancel("outside market hours")
            return

        if self.algorithm.time - self.pending_exit["submitted_at"] < timedelta(
            minutes=self.order_refresh_minutes
        ):
            return

        if self.has_partial_fill():
            return

        market_debit = self.current_market_debit()
        if market_debit is None:
            self.cancel("missing price data")
            return

        market_debit = round(market_debit, 2)
        if market_debit == self.pending_exit["last_market_debit"]:
            self.pending_exit["unchanged_refreshes"] += 1
            if (
                self.pending_exit["unchanged_refreshes"]
                < self.walk_after_unchanged_refreshes
            ):
                self.pending_exit["submitted_at"] = self.algorithm.time
                return

            new_offset = self.pending_exit["offset"] + self.walk_increment
            self.replace(market_debit, new_offset, "walking exit limit")
            return

        self.replace(market_debit, self.pending_exit["offset"], "repricing exit")

    def handle_order_event(self, order_event):
        if not self.pending_exit:
            return False

        pending_order_ids = {ticket.order_id for ticket in self.pending_exit["tickets"]}
        if order_event.order_id not in pending_order_ids:
            return False

        if order_event.status == OrderStatus.FILLED:
            signed_fill = self.signed_fill_amount(order_event)
            self.pending_exit["filled_net_debit"] = round(
                self.pending_exit["filled_net_debit"] + signed_fill, 2
            )

        if all(ticket.status == OrderStatus.FILLED for ticket in self.pending_exit["tickets"]):
            trade = self.pending_exit["trade"]
            exit_debit = round(self.pending_exit["filled_net_debit"], 2)
            current_pnl = round(trade["entry_credit"] - exit_debit, 2)
            self.algorithm.debug(
                f"EXIT FILLED: {self.pending_exit['reason']} | "
                f"entry_credit=${trade['entry_credit']:.2f} | "
                f"exit_debit=${exit_debit:.2f} | pnl=${current_pnl:.2f}"
            )
            self.pending_exit = None
            return True

        return False

    def replace(self, market_debit, offset, reason):
        trade = self.pending_exit["trade"]
        exit_reason = self.pending_exit["reason"]
        self.cancel(reason)

        limit_price = self.debit_limit_price(market_debit, offset)
        tickets = self.algorithm.combo_limit_order(self.exit_legs(trade), 1, limit_price)
        self.pending_exit = {
            "tickets": self.normalize_tickets(tickets),
            "trade": trade,
            "reason": exit_reason,
            "last_market_debit": market_debit,
            "filled_net_debit": 0.0,
            "limit_price": limit_price,
            "offset": offset,
            "unchanged_refreshes": 0,
            "submitted_at": self.algorithm.time,
        }

    def cancel(self, reason):
        if not self.pending_exit:
            return

        for ticket in self.pending_exit["tickets"]:
            if ticket.status in (
                OrderStatus.NEW,
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
            ):
                self.algorithm.transactions.cancel_order(ticket.order_id, reason)

        self.pending_exit = None

    def has_partial_fill(self):
        if not self.pending_exit:
            return False
        for ticket in self.pending_exit["tickets"]:
            if ticket.status == OrderStatus.PARTIALLY_FILLED:
                return True
        return False

    def current_market_debit(self):
        trade = self.pending_exit["trade"] if self.pending_exit else self.algorithm.trade
        if not trade:
            return None

        put_debit = self.vertical_debit(trade["short_put"], trade["long_put"])
        call_debit = self.vertical_debit(trade["short_call"], trade["long_call"])
        if put_debit is None or call_debit is None:
            return None
        return put_debit + call_debit

    def vertical_debit(self, short_symbol, long_symbol):
        short_ask = self.algorithm.securities[short_symbol].ask_price
        long_bid = self.algorithm.securities[long_symbol].bid_price
        if short_ask is None or long_bid is None:
            return None
        return short_ask - long_bid

    def debit_limit_price(self, market_debit, offset=None):
        if offset is None:
            offset = self.exit_limit_offset
        return round(max(0.05, market_debit + offset), 2)

    def exit_legs(self, trade):
        return [
            Leg.create(trade["short_put"], 1),
            Leg.create(trade["long_put"], -1),
            Leg.create(trade["short_call"], 1),
            Leg.create(trade["long_call"], -1),
        ]

    def normalize_tickets(self, tickets):
        if isinstance(tickets, list):
            return tickets
        return [tickets]

    def signed_fill_amount(self, order_event):
        quantity = abs(order_event.fill_quantity)
        gross_fill = order_event.fill_price * quantity
        if order_event.direction == OrderDirection.BUY:
            return gross_fill
        if order_event.direction == OrderDirection.SELL:
            return -gross_fill
        return 0.0
