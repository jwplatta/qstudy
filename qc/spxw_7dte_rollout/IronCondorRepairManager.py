class IronCondorRepairManager:
    """Executes close, recenter, and roll-forward repair flows."""

    def __init__(self, algorithm, action):
        self.algorithm = algorithm
        self.action = action
        self.pending_repair = None

    @property
    def is_pending(self):
        return self.pending_repair is not None

    def manage(self):
        if self.pending_repair and self.pending_repair["stage"] == "awaiting_reentry":
            self.attempt_repair_reentry()
            return True
        return False

    def start(self, trigger_reason):
        repair_plan = self.build_repair_plan(trigger_reason)
        if repair_plan is None:
            return False

        market_debit = self.algorithm.exit_order_manager.current_market_debit()
        if market_debit is None:
            return False

        self.pending_repair = repair_plan
        self.algorithm.exit_order_manager.submit_exit_order(
            self.algorithm.trade,
            market_debit,
            repair_plan["exit_reason"],
        )
        return True

    def build_repair_plan(self, trigger_reason):
        if self.action == "close":
            self.algorithm.debug(f"REPAIR TRIGGERED: {trigger_reason} | action=close")
            return {
                "action": "close",
                "stage": "exiting",
                "original_expiry": self.algorithm.trade["expiry"].date(),
                "exit_reason": f"Repair close: {trigger_reason}",
            }

        target_expiry = self.algorithm.trade["expiry"].date()
        chain = self.algorithm.current_slice.option_chains.get(self.algorithm.spxw)
        if not chain:
            self.algorithm.debug("REPAIR SKIPPED: no chain available to validate repair reentry")
            return None

        if self.action == "roll_forward":
            target_expiry = self.algorithm.next_valid_expiry(self.algorithm.time.date())
            if target_expiry == self.algorithm.trade["expiry"].date():
                target_expiry = self.algorithm.next_available_chain_expiry(target_expiry, chain)
            if target_expiry is None:
                self.algorithm.debug(
                    "REPAIR SKIPPED: no valid 7DTE replacement expiry available for roll-forward"
                )
                return None

        if not self.can_find_repair_entry(target_expiry):
            self.algorithm.debug(
                f"REPAIR SKIPPED: no replacement condor found for {self.action} "
                f"at expiry {target_expiry}"
            )
            return None

        self.algorithm.debug(
            f"REPAIR TRIGGERED: {trigger_reason} | action={self.action} | "
            f"target_expiry={target_expiry}"
        )
        return {
            "action": self.action,
            "stage": "exiting",
            "original_expiry": self.algorithm.trade["expiry"].date(),
            "target_expiry": target_expiry,
            "exit_reason": (
                f"Repair {self.action}: {trigger_reason} | "
                f"target_expiry={target_expiry}"
            ),
        }

    def can_find_repair_entry(self, target_expiry):
        chain = self.algorithm.current_slice.option_chains.get(self.algorithm.spxw)
        if not chain:
            return False

        contracts = [contract for contract in chain if contract.expiry.date() == target_expiry]
        if not contracts:
            return False

        spx_price = self.algorithm.securities[self.algorithm.spx].price
        return self.algorithm.iron_condor_finder.find_iron_condor(contracts, spx_price) is not None

    def attempt_repair_reentry(self):
        if (
            self.algorithm.position_entered
            or self.algorithm.entry_order_manager.is_pending
            or self.algorithm.exit_order_manager.is_pending
        ):
            return

        if not self.algorithm.is_regular_market_hours():
            self.pending_repair = None
            return

        target_expiry = self.pending_repair.get("target_expiry")
        if target_expiry is None:
            return

        submitted = self.algorithm.submit_entry_for_expiry(
            target_expiry,
            f"repair reentry ({self.pending_repair['action']})",
        )
        if submitted:
            self.pending_repair["stage"] = "entry_submitted"

    def handle_order_event(self, order_event):
        filled_trade = self.algorithm.entry_order_manager.handle_order_event(order_event)
        if filled_trade:
            if self.pending_repair and self.pending_repair["stage"] == "entry_submitted":
                filled_trade["repair_entry"] = self.pending_repair["action"]
                filled_trade["original_expiry"] = self.pending_repair["original_expiry"]
                self.pending_repair = None
            self.algorithm.trade = filled_trade
            self.algorithm.position_entered = True
            return True

        if self.algorithm.exit_order_manager.handle_order_event(order_event):
            prior_repair = self.pending_repair
            self.algorithm.trade = None
            self.algorithm.position_entered = False
            self.algorithm.blocked_entry_date = self.algorithm.time.date()

            if prior_repair and prior_repair["action"] in ("recenter", "roll_forward"):
                prior_repair["stage"] = "awaiting_reentry"
                self.pending_repair = prior_repair
            else:
                self.pending_repair = None
            return True

        return False

    def clear(self):
        self.pending_repair = None
