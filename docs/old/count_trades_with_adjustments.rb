#!/usr/bin/env ruby
# frozen_string_literal: true

require 'csv'

# Path to the CSV file
csv_file = File.join(__dir__, '..', 'tmp', 'trades.csv', 'Trading-1DTE-Table 1.csv')

# Actions that are not adjustments
normal_actions = ['Enter', 'Exit', 'Expired']

# Track trades and their adjustments
trades = []
current_trade = nil

CSV.foreach(csv_file, headers: true) do |row|
  trade_column = row['Trade']
  action = row['Action']

  # Check if this row starts a new trade (Trade column contains "1DTE IronCondor")
  if trade_column && trade_column.include?('IronCondor')
    # Save the previous trade if it exists
    trades << current_trade if current_trade

    # Start a new trade
    current_trade = {
      name: trade_column,
      date: row['Date'],
      has_adjustment: false,
      actions: []
    }
  end

  # Skip rows without actions or rows that are just date groupings
  next if action.nil? || action.strip.empty?

  # Track the action for the current trade
  if current_trade
    current_trade[:actions] << action

    # Check if this action is an adjustment (not Enter, Exit, or Expired)
    unless normal_actions.include?(action)
      current_trade[:has_adjustment] = true
    end

    # Check if Enter comes after Exit (which indicates an adjustment)
    if action == 'Enter' && current_trade[:actions].include?('Exit')
      current_trade[:has_adjustment] = true
    end
  end
end

# Add the last trade
trades << current_trade if current_trade

# Count trades with adjustments
trades_with_adjustments = trades.select { |trade| trade[:has_adjustment] }

# Print results
puts "Total trades: #{trades.count}"
puts "Trades with adjustments: #{trades_with_adjustments.count}"
puts "Trades without adjustments: #{trades.count - trades_with_adjustments.count}"
puts "\nPercentage with adjustments: #{(trades_with_adjustments.count.to_f / trades.count * 100).round(2)}%"

puts "\n--- Trades with adjustments ---"
trades_with_adjustments.each_with_index do |trade, index|
  puts "#{index + 1}. #{trade[:name]}"
  adjustment_actions = trade[:actions].reject { |a| normal_actions.include?(a) }
  puts "   Adjustments: #{adjustment_actions.uniq.join(', ')}"
end
