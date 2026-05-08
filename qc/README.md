## Refactor Ideas

### `spxw_7dte_no_constraints`

1. Extract an `ExpirySelector` service.
   - Responsibility: event-date checks, day-after-event checks, 6/7/8-DTE preference rules, and chain-based fallback expiry selection.
   - Why: this logic is coherent on its own and currently mixed into the algorithm lifecycle.

- [x] Extract an `IronCondorEntryOrderManager`.
   - Responsibility: submit combo limit orders, track pending tickets, reprice, walk limits, cancel outside market hours, and handle fills.
   - Why: this is already a state machine and is the densest part of the algorithm.
   - Status: done for iron condors only.
   - Follow-up: add a separate `VerticalEntryOrderManager` when vertical entry/exit workflows are needed.

3. Extract a `TradeLifecycle` or `PositionState` object.
   - Responsibility: represent flat vs pending-entry vs live-position state, hold trade metadata, and own expiration/reset behavior.
   - Why: it would replace the current spread of `self.trade`, `self.position_entered`, and `self.pending_entry` with one explicit state model.
