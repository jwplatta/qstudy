from datetime import timedelta
import math

from AlgorithmImports import *
from event_dates import get_event_dates
from IronCondorFinder import IronCondorFinder


class Spxw1dteRegimeForecast(QCAlgorithm):
    """
    SPXW 1DTE baseline iron condor strategy with a regime forecast filter.

    - Enters 1DTE iron condors starting at 3:50pm ET, retries every 20s until 4:00pm
    - Skips entries when next trading day is an event date (FOMC, CPI, employment)
    - Skips entries when the regime model implies a high probability of a 51+ point SPX range
    - Exits at 60% profit, -3x loss, or on 0DTE day (12pm if profitable, 1pm forced)
    """

    PROFIT_TARGET = 0.60
    MAX_LOSS_MULTIPLIER = -3.0
    START_DATE = (2022, 4, 1)
    END_DATE = (2025, 12, 31)
    INITIAL_CASH = 50000

    RANGE_THRESHOLD = 51.698
    REGIME_SKIP_PROBABILITY = 0.50

    # Offline fit using the feature set selected in `qc/spxw_1dte_baseline/research.ipynb`,
    # with the target shifted one trading day forward to match the 1DTE entry timing.
    REGIME_LOGIT_INTERCEPT = -3.9225339071705148
    REGIME_LOGIT_WEIGHTS = {
        "prior_slope": 0.15572728210666714,
        "5d_avg_range": 0.06168979627510342,
        "prior_abs_ret": 36.499736265948194,
        "gap_mag": 40.380107779126185,
    }

    def initialize(self):
        self.set_start_date(*self.START_DATE)
        self.set_end_date(*self.END_DATE)
        self.set_cash(self.INITIAL_CASH)

        self.set_brokerage_model(BrokerageName.CHARLES_SCHWAB, AccountType.MARGIN)
        self.settings.seed_initial_prices = True

        self.spx = self.add_index("SPX", Resolution.MINUTE).symbol
        self.vix = self.add_index("VIX", Resolution.DAILY).symbol
        self.vix9d = self.add_index("VIX9D", Resolution.DAILY).symbol

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
        self.current_day_open = None
        self.current_day_open_date = None

        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.after_market_open(self.spx, 1),
            self.capture_market_open,
        )

        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.at(15, 50, TimeZones.NEW_YORK),
            self.check_entry,
        )

        self.schedule.on(
            self.date_rules.every_day(self.spx),
            self.time_rules.every(timedelta(minutes=5)),
            self.monitor_positions,
        )

    def load_event_dates(self):
        try:
            return get_event_dates()
        except Exception as error:
            raise Exception(f"Error loading event dates: {error}")

    def is_expiration_on_event_date(self, expiry_date):
        return expiry_date in self.event_dates

    def next_valid_expiry(self, from_date):
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

    def capture_market_open(self):
        current_date = self.time.date()
        if self.current_day_open_date == current_date:
            return

        minute_history = self.history(self.spx, 2, Resolution.MINUTE)
        if minute_history.empty:
            self.debug(f"{current_date} - Unable to capture SPX open")
            return

        minute_history = minute_history.reset_index(level=0, drop=True)
        minute_history = minute_history[minute_history.index.date == current_date]
        if minute_history.empty:
            self.debug(f"{current_date} - SPX minute history missing current session open")
            return

        self.current_day_open = float(minute_history.iloc[0]["open"])
        self.current_day_open_date = current_date

    def check_entry(self):
        current_date = self.time.date()

        if self.is_warming_up or self.position_entered or self.time.hour >= 16:
            return

        target_expiry = self.next_valid_expiry(current_date)
        if not target_expiry:
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - No valid expiry found in next 10 days"
            )
            return

        regime_probability, features = self.forecast_next_day_regime()
        if regime_probability is not None:
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - Regime forecast prob={regime_probability:.3f} "
                f"features={features}"
            )
            if regime_probability >= self.REGIME_SKIP_PROBABILITY:
                self.debug(
                    f"{current_date} {self.time.strftime('%H:%M')} - Skipping entry: "
                    f"high-regime probability {regime_probability:.3f} >= {self.REGIME_SKIP_PROBABILITY:.2f}"
                )
                return
        else:
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - Regime forecast unavailable, using baseline rules"
            )

        chain = self.current_slice.option_chains.get(self.spxw)
        if not chain:
            self.debug(f"{current_date} {self.time.strftime('%H:%M')} - No option chain available")
            self.schedule_retry()
            return

        chain_size = len(chain)
        contracts = [contract for contract in chain if contract.expiry.date() == target_expiry]

        self.debug(
            f"{current_date} {self.time.strftime('%H:%M')} - Option chain has {chain_size} contracts / "
            f"{len(contracts)} contracts expiring on {target_expiry}"
        )

        if not contracts:
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - No contracts expiring on {target_expiry}"
            )
            self.schedule_retry()
            return

        spx_price = self.securities[self.spx].price
        self.debug(
            f"{current_date} {self.time.strftime('%H:%M')} - Searching for iron condor, "
            f"SPX={spx_price:.2f}, target expiry={target_expiry}"
        )
        result = self.iron_condor_finder.find_iron_condor(contracts, spx_price)

        if result:
            call_spread, put_spread, tweak_count = result
            self.debug(
                f"{current_date} {self.time.strftime('%H:%M')} - Found valid iron condor after {tweak_count} tweaks"
            )
            self.enter_position(
                call_spread, put_spread, spx_price, target_expiry, regime_probability
            )
            return

        self.debug(
            f"{current_date} {self.time.strftime('%H:%M')} - No valid iron condor found, "
            f"will retry in {self.entry_retry_seconds}s"
        )
        self.schedule_retry()

    def forecast_next_day_regime(self):
        current_date = self.time.date()
        if self.current_day_open_date != current_date or self.current_day_open is None:
            self.capture_market_open()

        if self.current_day_open_date != current_date or self.current_day_open is None:
            return (None, None)

        spx_history = self.daily_history(self.spx, 6)
        vix_history = self.daily_history(self.vix, 1)
        vix9d_history = self.daily_history(self.vix9d, 1)

        if spx_history is None or vix_history is None or vix9d_history is None:
            return (None, None)

        if len(spx_history) < 6 or len(vix_history) < 1 or len(vix9d_history) < 1:
            return (None, None)

        daily_ranges = spx_history["high"] - spx_history["low"]
        yesterday_close = float(spx_history.iloc[-1]["close"])
        day_before_close = float(spx_history.iloc[-2]["close"])

        features = {
            "prior_slope": round(
                float(vix9d_history.iloc[-1]["close"] - vix_history.iloc[-1]["close"]), 6
            ),
            "5d_avg_range": round(float(daily_ranges.tail(5).mean()), 6),
            "prior_abs_ret": round(abs(math.log(yesterday_close / day_before_close)), 8),
            "gap_mag": round(abs((self.current_day_open - yesterday_close) / yesterday_close), 8),
        }

        logit = self.REGIME_LOGIT_INTERCEPT
        for name, weight in self.REGIME_LOGIT_WEIGHTS.items():
            logit += weight * features[name]

        probability = 1.0 / (1.0 + math.exp(-logit))
        return (probability, features)

    def daily_history(self, symbol, periods):
        history = self.history(symbol, periods + 1, Resolution.DAILY)
        if history.empty:
            return None

        history = history.reset_index(level=0, drop=True)
        history = history[history.index.date < self.time.date()]
        if history.empty:
            return None

        return history.tail(periods)

    def schedule_retry(self):
        if self.time.hour < 16:
            retry_time = self.time + timedelta(seconds=self.entry_retry_seconds)
            self.schedule.on(
                self.date_rules.on(retry_time.year, retry_time.month, retry_time.day),
                self.time_rules.at(retry_time.hour, retry_time.minute, TimeZones.NEW_YORK),
                self.check_entry,
            )

    def enter_position(self, call_spread, put_spread, spx_price, expiry_date, regime_probability):
        total_credit = call_spread["price"] + put_spread["price"]
        regime_text = (
            f"REGIME_PROB={regime_probability:.3f}"
            if regime_probability is not None
            else "REGIME_PROB=n/a"
        )

        self.debug(
            f"ENTRY: SPX={spx_price:.2f} | "
            f"PUT={put_spread['short_leg'].strike}/{put_spread['long_leg'].strike} @ ${put_spread['price']:.2f} | "
            f"CALL={call_spread['short_leg'].strike}/{call_spread['long_leg'].strike} @ ${call_spread['price']:.2f} | "
            f"TOTAL CREDIT=${total_credit:.2f} | EXPIRY={expiry_date} | "
            f"{regime_text}"
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
            "profit_target": round(total_credit * self.PROFIT_TARGET, 2),
            "max_loss": round(total_credit * self.MAX_LOSS_MULTIPLIER, 2),
            "long_put": put_spread["long_leg"].symbol,
            "short_put": put_spread["short_leg"].symbol,
            "short_call": call_spread["short_leg"].symbol,
            "long_call": call_spread["long_leg"].symbol,
            "expiry": call_spread["short_leg"].expiry,
            "entry_spx_price": round(spx_price, 2),
            "entry_time": self.time,
            "regime_probability": None
            if regime_probability is None
            else round(regime_probability, 4),
        }

        self.position_entered = True

    def monitor_positions(self):
        if not self.position_entered or not self.trade or self.is_warming_up:
            return

        current_pnl = self.calculate_pnl()

        if current_pnl >= self.trade["profit_target"]:
            self.exit_position(f"Profit target reached: ${current_pnl:.2f}")
            return

        if current_pnl <= self.trade["max_loss"]:
            self.exit_position(f"Max loss reached: ${current_pnl:.2f}")
            return

        if self.is_0dte():
            current_hour = self.time.hour
            if current_hour >= 12 and current_pnl > 0:
                self.exit_position(f"0DTE 12pm+ profitable exit: ${current_pnl:.2f}")
                return

            if current_hour >= 13:
                self.exit_position(f"0DTE 1pm+ forced exit: ${current_pnl:.2f}")
                return

    def calculate_pnl(self):
        short_put_price = self.securities[self.trade["short_put"]].price
        long_put_price = self.securities[self.trade["long_put"]].price
        short_call_price = self.securities[self.trade["short_call"]].price
        long_call_price = self.securities[self.trade["long_call"]].price

        put_exit_cost = short_put_price - long_put_price
        call_exit_cost = short_call_price - long_call_price
        total_exit_cost = put_exit_cost + call_exit_cost
        return self.trade["entry_credit"] - total_exit_cost

    def exit_position(self, reason):
        if not self.position_entered:
            return

        current_pnl = self.calculate_pnl()
        self.debug(
            f"EXIT: {reason} | Entry Credit: ${self.trade['entry_credit']:.2f} | P&L: ${current_pnl:.2f}"
        )

        legs = [
            Leg.create(self.trade["short_put"], 1),
            Leg.create(self.trade["long_put"], -1),
            Leg.create(self.trade["short_call"], 1),
            Leg.create(self.trade["long_call"], -1),
        ]
        self.combo_market_order(legs, 1)

        self.position_entered = False
        self.trade = None

    def is_0dte(self):
        return self.trade and self.trade["expiry"].date() == self.time.date()
