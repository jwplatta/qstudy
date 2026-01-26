#!/usr/bin/env python
"""Generate vanna exposure chart for SPXW options."""

import argparse
from datetime import datetime
from pathlib import Path

from trade_lab.charts import GreekExposure


def main():
    parser = argparse.ArgumentParser(description="Generate vanna exposure chart")
    parser.add_argument("--symbol", default="SPXW", help="Ticker symbol (default: SPXW)")
    parser.add_argument(
        "--start-date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Start date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--days-out", type=int, default=5, help="Days out for expirations (default: 5)"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing option chain CSV files (default: data)",
    )
    parser.add_argument(
        "--output-dir", default="/tmp", help="Directory to save charts (default: /tmp)"
    )
    parser.add_argument(
        "--greek",
        default="vanna",
        choices=["gamma", "vanna", "charm"],
        help="Greek to plot (default: vanna)",
    )
    parser.add_argument(
        "--dealer-sign",
        type=float,
        default=1.0,
        help="Dealer sign: 1.0 for raw, -1.0 for dealer flow (default: 1.0)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chart = GreekExposure(
        symbol=args.symbol,
        start_date=args.start_date,
        days_out=args.days_out,
        data_dir=args.data_dir,
        greek=args.greek,
        dealer_sign=args.dealer_sign,
        debug=args.debug,
    )

    output_file = output_dir / f"{args.symbol}_{args.greek}_exposure_{args.start_date}.png"
    fig, ax = chart.plot(save_path=str(output_file))

    print(f"Chart saved to: {output_file}")


if __name__ == "__main__":
    main()
