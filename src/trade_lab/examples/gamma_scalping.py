"""
Gamma Scalping Strategy Implementation

This module demonstrates a gamma scalping strategy that maintains delta neutrality
by continuously rebalancing an options portfolio. The strategy profits from the
convexity (gamma) of long options positions while managing directional risk.

Based on: https://alpaca.markets/learn/gamma-scalping

Key Components:
1. Options selection and Greeks calculation using Black-Scholes
2. Delta-neutral position establishment
3. Continuous portfolio monitoring and rebalancing
4. Risk management with notional delta thresholds

WARNING: This is for educational purposes only. Options trading involves substantial
risk and requires active monitoring. Not suitable for all investors.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import math
from scipy.stats import norm
from scipy.optimize import brentq

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.stream import TradingStream


class GammaScalper:
    """
    Implements a gamma scalping strategy that maintains delta neutrality.

    Attributes:
        trading_client: Alpaca trading API client
        data_client: Alpaca data API client
        stream: Real-time trade update stream
        symbol: Underlying asset ticker symbol
        risk_free_rate: Annual risk-free interest rate (default 4.5%)
        max_notional_delta: Maximum acceptable notional delta exposure
        rebalance_interval: Seconds between delta checks
        initial_delay: Seconds to wait before first rebalance
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        risk_free_rate: float = 0.045,
        max_notional_delta: float = 500.0,
        rebalance_interval: int = 120,
        initial_delay: int = 30
    ):
        """
        Initialize the gamma scalper with API credentials and parameters.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            symbol: Underlying ticker symbol (e.g., 'JPM', 'SPY')
            risk_free_rate: Annual risk-free rate (default 0.045 = 4.5%)
            max_notional_delta: Max notional delta before rebalancing (default $500)
            rebalance_interval: Seconds between rebalance checks (default 120)
            initial_delay: Seconds before first rebalance (default 30)
        """
        self.trading_client = TradingClient(api_key, api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        self.stream = TradingStream(api_key, api_secret, paper=True)

        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        self.max_notional_delta = max_notional_delta
        self.rebalance_interval = rebalance_interval
        self.initial_delay = initial_delay

        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.underlying_price: float = 0.0

    def calculate_implied_volatility(
        self,
        option_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str
    ) -> float:
        """
        Calculate implied volatility using Brent's optimization method.

        Solves the Black-Scholes equation backward to find the volatility that
        produces the observed market price.

        Args:
            option_price: Market price of the option
            S: Current underlying price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free interest rate
            option_type: 'call' or 'put'

        Returns:
            Implied volatility as a decimal (e.g., 0.25 = 25%)
        """
        def black_scholes_price(sigma: float) -> float:
            """Black-Scholes option pricing formula."""
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            if option_type.lower() == 'call':
                return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        def objective(sigma: float) -> float:
            """Objective function for optimization."""
            return black_scholes_price(sigma) - option_price

        try:
            # Search for IV in range [0.01, 5.0]
            implied_vol = brentq(objective, 0.01, 5.0)
            return implied_vol
        except ValueError:
            # If no solution found, return a default estimate
            return 0.25

    def calculate_greeks(
        self,
        option_price: float,
        strike_price: float,
        expiry: datetime,
        underlying_price: float,
        option_type: str = 'call'
    ) -> Tuple[float, float, float]:
        """
        Calculate option Greeks using Black-Scholes model.

        Args:
            option_price: Current option market price
            strike_price: Option strike price
            expiry: Option expiration datetime
            underlying_price: Current underlying asset price
            option_type: 'call' or 'put'

        Returns:
            Tuple of (delta, gamma, implied_volatility)
        """
        # Calculate time to expiration in years
        T = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)

        if T <= 0:
            return (0.0, 0.0, 0.0)

        # Calculate implied volatility
        iv = self.calculate_implied_volatility(
            option_price,
            underlying_price,
            strike_price,
            T,
            self.risk_free_rate,
            option_type
        )

        # Calculate d1 for Greeks
        d1 = (
            math.log(underlying_price / strike_price) +
            (self.risk_free_rate + 0.5 * iv**2) * T
        ) / (iv * math.sqrt(T))

        # Calculate delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Calculate gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (underlying_price * iv * math.sqrt(T))

        return (delta, gamma, iv)

    async def liquidate_existing_positions(self) -> None:
        """
        Close all existing positions for the underlying symbol.

        Ensures a clean slate before starting the gamma scalping strategy.
        """
        try:
            positions = self.trading_client.get_all_positions()

            for position in positions:
                if position.symbol == self.symbol or self.symbol in position.symbol:
                    qty = abs(float(position.qty))
                    side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY

                    order_data = MarketOrderRequest(
                        symbol=position.symbol,
                        qty=qty,
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )

                    self.trading_client.submit_order(order_data)
                    print(f"Liquidated position: {position.symbol} ({side.value} {qty})")

        except Exception as e:
            print(f"Error liquidating positions: {e}")

    async def get_underlying_price(self) -> float:
        """
        Fetch the latest price for the underlying asset.

        Returns:
            Current market price of the underlying
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=[self.symbol])
            quotes = self.data_client.get_stock_latest_quote(request)

            if self.symbol in quotes:
                quote = quotes[self.symbol]
                price = (quote.ask_price + quote.bid_price) / 2
                self.underlying_price = price
                return price
            else:
                raise ValueError(f"No quote data for {self.symbol}")

        except Exception as e:
            print(f"Error fetching underlying price: {e}")
            return 0.0

    async def select_options(self, num_contracts: int = 3) -> List[Dict]:
        """
        Select call options for the gamma scalping strategy.

        Selection criteria:
        - Expiration between 14-60 days from today
        - Strike price at least 1% above current underlying price
        - Active and tradable contracts

        Args:
            num_contracts: Number of different option contracts to select

        Returns:
            List of option contract details
        """
        try:
            # Get current underlying price
            current_price = await self.get_underlying_price()
            min_strike = current_price * 1.01

            # Calculate date range
            today = datetime.now().date()
            min_expiry = today + timedelta(days=14)
            max_expiry = today + timedelta(days=60)

            # Search for call options
            search_params = GetAssetsRequest(
                asset_class=AssetClass.US_OPTION
            )

            assets = self.trading_client.get_all_assets(search_params)

            # Filter options
            eligible_options = []

            for asset in assets:
                # Check if it's an option on our underlying
                if not asset.symbol.startswith(self.symbol):
                    continue

                # Parse option symbol (format: SYMBOL YYMMDD C/P STRIKE)
                try:
                    parts = asset.symbol.split()
                    if len(parts) < 3:
                        continue

                    exp_str = parts[1]
                    option_type = parts[2][0]
                    strike_str = parts[2][1:]

                    # Only select calls
                    if option_type != 'C':
                        continue

                    # Parse expiration
                    exp_date = datetime.strptime(exp_str, '%y%m%d').date()

                    # Parse strike
                    strike = float(strike_str) / 1000

                    # Apply filters
                    if (min_expiry <= exp_date <= max_expiry and
                        strike >= min_strike and
                        asset.tradable and asset.status == 'active'):

                        eligible_options.append({
                            'symbol': asset.symbol,
                            'strike': strike,
                            'expiry': datetime.combine(exp_date, datetime.min.time()),
                            'asset': asset
                        })

                except Exception:
                    continue

            # Sort by expiration, then by strike
            eligible_options.sort(key=lambda x: (x['expiry'], x['strike']))

            # Select first num_contracts
            selected = eligible_options[:num_contracts]

            print(f"Selected {len(selected)} call options:")
            for opt in selected:
                print(f"  {opt['symbol']} - Strike: ${opt['strike']:.2f}, "
                      f"Expiry: {opt['expiry'].date()}")

            return selected

        except Exception as e:
            print(f"Error selecting options: {e}")
            return []

    async def execute_initial_trades(self, options: List[Dict]) -> None:
        """
        Execute initial option purchases to establish long gamma position.

        Args:
            options: List of option contracts to purchase
        """
        try:
            for option in options:
                order_data = MarketOrderRequest(
                    symbol=option['symbol'],
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )

                order = self.trading_client.submit_order(order_data)

                # Initialize position tracking
                self.positions[option['symbol']] = {
                    'qty': 1,
                    'strike': option['strike'],
                    'expiry': option['expiry'],
                    'type': 'option',
                    'delta': 0.0,
                    'gamma': 0.0
                }

                print(f"Submitted order for {option['symbol']}: {order.id}")

            # Wait for fills
            await asyncio.sleep(5)

        except Exception as e:
            print(f"Error executing initial trades: {e}")

    async def update_greeks(self) -> None:
        """
        Update Greeks for all option positions based on current market prices.
        """
        try:
            current_price = await self.get_underlying_price()

            for symbol, position in self.positions.items():
                if position['type'] != 'option':
                    continue

                # Get current option price from position
                try:
                    pos = self.trading_client.get_open_position(symbol)
                    option_price = float(pos.current_price)

                    # Calculate Greeks
                    delta, gamma, iv = self.calculate_greeks(
                        option_price,
                        position['strike'],
                        position['expiry'],
                        current_price,
                        'call'
                    )

                    # Update position
                    position['delta'] = delta
                    position['gamma'] = gamma
                    position['iv'] = iv

                except Exception as e:
                    print(f"Error updating Greeks for {symbol}: {e}")

        except Exception as e:
            print(f"Error in update_greeks: {e}")

    def calculate_portfolio_delta(self) -> float:
        """
        Calculate the total portfolio delta.

        Returns:
            Total delta exposure (positive = long, negative = short)
        """
        total_delta = 0.0

        for symbol, position in self.positions.items():
            if position['type'] == 'option':
                # Options: delta * contracts * 100 shares per contract
                total_delta += position['delta'] * position['qty'] * 100
            else:
                # Underlying stock: 1 delta per share
                total_delta += position['qty']

        return total_delta

    async def adjust_delta(self, current_delta: float) -> None:
        """
        Adjust underlying stock position to maintain delta neutrality.

        Args:
            current_delta: Current portfolio delta exposure
        """
        try:
            current_price = await self.get_underlying_price()
            notional_delta = abs(current_delta * current_price)

            # Check if adjustment needed
            if notional_delta < self.max_notional_delta:
                return

            # Determine action
            if current_delta > 0:
                # Positive delta: sell underlying to reduce
                side = OrderSide.SELL
                qty = abs(int(current_delta))
            else:
                # Negative delta: buy underlying to increase
                side = OrderSide.BUY
                qty = abs(int(current_delta))

            if qty == 0:
                return

            # Submit adjustment order
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data)

            # Update stock position tracking
            if self.symbol not in self.positions:
                self.positions[self.symbol] = {'qty': 0, 'type': 'stock'}

            if side == OrderSide.BUY:
                self.positions[self.symbol]['qty'] += qty
            else:
                self.positions[self.symbol]['qty'] -= qty

            print(f"Delta adjustment: {side.value} {qty} shares of {self.symbol} "
                  f"(delta: {current_delta:.2f}, notional: ${notional_delta:.2f})")

        except Exception as e:
            print(f"Error adjusting delta: {e}")

    async def maintain_delta_neutral(self) -> None:
        """
        Continuous loop to monitor and maintain delta neutrality.

        Runs indefinitely, checking portfolio delta at regular intervals and
        rebalancing when necessary.
        """
        # Initial delay to allow positions to establish
        await asyncio.sleep(self.initial_delay)

        while True:
            try:
                # Update Greeks with current market data
                await self.update_greeks()

                # Calculate current portfolio delta
                portfolio_delta = self.calculate_portfolio_delta()

                current_price = await self.get_underlying_price()
                notional_delta = portfolio_delta * current_price

                print(f"\nPortfolio Status:")
                print(f"  Underlying: ${current_price:.2f}")
                print(f"  Delta: {portfolio_delta:.2f}")
                print(f"  Notional Delta: ${notional_delta:.2f}")

                # Adjust if needed
                await self.adjust_delta(portfolio_delta)

                # Wait before next check
                await asyncio.sleep(self.rebalance_interval)

            except Exception as e:
                print(f"Error in maintain_delta_neutral loop: {e}")
                await asyncio.sleep(self.rebalance_interval)

    async def handle_trade_updates(self, data):
        """
        Handle real-time trade update events from the stream.

        Args:
            data: Trade update event data
        """
        event = data.event
        order = data.order

        print(f"Trade update: {event} for {order.symbol}")

        if event == 'fill':
            print(f"  Filled: {order.filled_qty} @ ${order.filled_avg_price}")

    async def run(self) -> None:
        """
        Main execution loop for the gamma scalping strategy.

        Steps:
        1. Liquidate existing positions
        2. Select suitable options contracts
        3. Execute initial option purchases
        4. Start trade update stream
        5. Begin continuous delta-neutral maintenance
        """
        try:
            print("Starting Gamma Scalping Strategy")
            print(f"Symbol: {self.symbol}")
            print(f"Risk-free rate: {self.risk_free_rate:.2%}")
            print(f"Max notional delta: ${self.max_notional_delta:.2f}\n")

            # Step 1: Clean slate
            print("Liquidating existing positions...")
            await self.liquidate_existing_positions()
            await asyncio.sleep(2)

            # Step 2: Get current price and select options
            print("\nFetching underlying price...")
            current_price = await self.get_underlying_price()
            print(f"Current {self.symbol} price: ${current_price:.2f}\n")

            print("Selecting options contracts...")
            options = await self.select_options(num_contracts=3)

            if not options:
                print("No suitable options found. Exiting.")
                return

            # Step 3: Execute initial trades
            print("\nExecuting initial option purchases...")
            await self.execute_initial_trades(options)

            # Step 4: Subscribe to trade updates
            self.stream.subscribe_trade_updates(self.handle_trade_updates)

            # Step 5: Run strategy
            print("\nStarting delta-neutral maintenance loop...\n")

            # Run stream and maintenance concurrently
            await asyncio.gather(
                self.stream._run_forever(),
                self.maintain_delta_neutral()
            )

        except KeyboardInterrupt:
            print("\nStrategy stopped by user")
        except Exception as e:
            print(f"Error in run loop: {e}")
        finally:
            print("Closing connections...")


async def main():
    """
    Example usage of the GammaScalper class.

    Configure with your Alpaca API credentials and desired parameters.
    """
    # Configuration
    API_KEY = "your_api_key_here"
    API_SECRET = "your_api_secret_here"
    SYMBOL = "SPY"  # Underlying ticker

    # Initialize scalper
    scalper = GammaScalper(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol=SYMBOL,
        risk_free_rate=0.045,  # 4.5% annual rate
        max_notional_delta=500.0,  # $500 max delta exposure
        rebalance_interval=120,  # Check every 2 minutes
        initial_delay=30  # Wait 30 seconds before first rebalance
    )

    # Run strategy
    await scalper.run()


if __name__ == "__main__":
    asyncio.run(main())
