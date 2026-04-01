#!/usr/bin/env python3
"""
Script to analyze 1DTE IronCondor trades from Trading-1DTE.csv

This script loops through the CSV file and:
- Identifies each trade (marked by "1DTE IronCondor" in the first column)
- Extracts the profit percentage from each trade
- Counts positive and negative profits
- Provides summary statistics
"""

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def analyze_trades(csv_path: str):
    """
    Analyze trades from the CSV file.

    Args:
        csv_path: Path to the Trading-1DTE.csv file
    """
    trades = []
    current_trade = None
    current_month = None

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row

        for row in reader:
            # Check if this row is a month marker (first column has a date, rest is empty)
            if row[0] and row[0][0].isdigit() and all(not cell for cell in row[1:]):
                try:
                    current_month = datetime.strptime(row[0], "%m/%d/%Y")
                except ValueError:
                    pass
                continue

            # Check if this is the start of a new trade
            if row[0] == "1DTE IronCondor":
                # If we were tracking a previous trade, save it
                if current_trade is not None:
                    trades.append(current_trade)

                # Start a new trade
                current_trade = {
                    "start_row": row,
                    "profit_pct": None,
                    "total": None,
                    "month": current_month,
                }

            # Check if this row contains the Profit percentage
            # The Profit row has "Profit" in the 12th column (index 11)
            elif len(row) > 11 and row[11] == "Profit":
                if current_trade is not None and len(row) > 12:
                    profit_str = row[12].strip()
                    # Remove the % sign and convert to float
                    if profit_str and profit_str != "":
                        profit_str = profit_str.rstrip("%")
                        try:
                            current_trade["profit_pct"] = float(profit_str)
                        except ValueError:
                            print(f"Warning: Could not parse profit value: {profit_str}")

            # Check if this row contains the Total
            elif len(row) > 11 and row[11] == "Total":
                if current_trade is not None and len(row) > 12:
                    total_str = row[12].strip()
                    if total_str and total_str != "":
                        try:
                            current_trade["total"] = float(total_str)
                        except ValueError:
                            print(f"Warning: Could not parse total value: {total_str}")

        # Don't forget the last trade
        if current_trade is not None:
            trades.append(current_trade)

    # Group trades by month
    monthly_trades = defaultdict(list)
    for trade in trades:
        if trade["month"] is not None and trade["profit_pct"] is not None:
            month_key = trade["month"].strftime("%Y-%m")
            monthly_trades[month_key].append(trade)

    print(f"\n{'=' * 80}")
    print("1DTE IronCondor Trade Analysis - Monthly Breakdown")
    print(f"{'=' * 80}\n")

    # Overall totals
    overall_wins = 0
    overall_losses = 0
    overall_win_pnl = 0.0
    overall_loss_pnl = 0.0

    # Process each month
    for month in sorted(monthly_trades.keys()):
        month_trades = monthly_trades[month]
        month_name = datetime.strptime(month, "%Y-%m").strftime("%B %Y")

        wins = 0
        losses = 0
        win_pnl = 0.0
        loss_pnl = 0.0

        for trade in month_trades:
            profit = trade["profit_pct"]
            total = trade["total"] if trade["total"] is not None else 0.0

            if profit >= 0:
                wins += 1
                win_pnl += total
            else:
                losses += 1
                loss_pnl += total

        total_trades = wins + losses
        net_pnl = win_pnl + loss_pnl
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        print(f"{month_name}")
        print(f"{'-' * 80}")
        print(
            f"  Trades: {total_trades:2d}  |  Wins: {wins:2d} ({win_rate:5.1f}%)  |  Losses: {losses:2d}"
        )
        print(f"  Wins:   ${win_pnl:9,.2f}")
        print(f"  Losses: ${loss_pnl:9,.2f}")
        print(f"  Net:    ${net_pnl:9,.2f}")
        print()

        # Add to overall totals
        overall_wins += wins
        overall_losses += losses
        overall_win_pnl += win_pnl
        overall_loss_pnl += loss_pnl

    # Print overall totals
    print(f"{'=' * 80}")
    print("OVERALL TOTALS")
    print(f"{'=' * 80}")

    total_trades = overall_wins + overall_losses
    overall_win_rate = (overall_wins / total_trades * 100) if total_trades > 0 else 0
    overall_net_pnl = overall_win_pnl + overall_loss_pnl

    mean_profit_per_trade = overall_net_pnl / total_trades if total_trades > 0 else 0
    profit_factor = (
        overall_win_pnl / abs(overall_loss_pnl) if overall_loss_pnl != 0 else float("inf")
    )

    print(f"Total Trades:    {total_trades:3d}")
    print(f"Winning Trades:  {overall_wins:3d} ({overall_win_rate:.1f}%)")
    print(f"Losing Trades:   {overall_losses:3d}")
    print()
    print(f"Wins:            ${overall_win_pnl:,.2f}")
    print(f"Losses:          ${overall_loss_pnl:,.2f}")
    print(f"Net:             ${overall_net_pnl:,.2f}")
    print(f"Mean PnL:        ${mean_profit_per_trade:,.2f}")
    print(f"Profit Factor:   {profit_factor:.2f}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    csv_path = Path("./data/trades/Trading-1DTE.csv")

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        exit(1)

    analyze_trades(str(csv_path))
