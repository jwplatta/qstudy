#!/usr/bin/env ruby
# frozen_string_literal: true

require "pry"
require "bundler/setup"
require "schwab_rb"
require "dotenv"
require "date"
require "csv"
require "fileutils"

Dotenv.load

SchwabRb.configure do |config|
  config.log_file = "./tmp/schwab_rb.log"
  config.log_level = "DEBUG"
  config.silence_output = false
end

def create_client
  token_path = ENV["SCHWAB_TOKEN_PATH"] || File.join(Dir.home, ".schwab_rb", "token.json")

  SchwabRb::Auth.init_client_easy(
    ENV.fetch("SCHWAB_API_KEY"),
    ENV.fetch("SCHWAB_APP_SECRET"),
    ENV.fetch("SCHWAB_APP_CALLBACK_URL"),
    token_path
  )
end

schwab = create_client
acct = schwab.get_account(account_name: "TRADING_ACCOUNT", fields: [:positions])

# spx_opt_positions = acct.positions.select { |pos| pos.instrument.asset_type == "OPTION" && pos.instrument.symbol.start_with?("SPXW") }

# spx_opt_positions.each do |pos|
#   puts pos.symbol
# end

# def get_transactions(
#   account_hash = nil,
#   account_name: nil,
#   start_date: nil,
#   end_date: nil,
#   transaction_types: nil,
#   symbol: nil,
#   return_data_objects: true
# )

def get_trasnactions(client, start_month, end_month)
  client.get_transactions(
    account_name: "TRADING_ACCOUNT",
    start_date: start_month,
    end_date: end_month,
    transaction_types: ["TRADE"]
  )
end

def contains_spxw_option?(transaction)
  # monthly_trasnactions.first.first.transfer_items.last.symbol
  transaction.transfer_items.any? do |item|
    item.instrument.asset_type == "OPTION" && item.instrument.symbol.start_with?("SPXW")
  end
end

monthly_trasnactions = (1..12).map do |month|
  if month == 12
    get_trasnactions(schwab, Date.new(2025, month, 1), Date.new(2026, 1, 1))
  else
    get_trasnactions(schwab, Date.new(2025, month, 1), Date.new(2025, month + 1, 1))
  end
end

all_spxw_transactions = monthly_trasnactions.flatten.select do |transaction|
  contains_spxw_option?(transaction)
end

monthly_sums = {}

all_spxw_transactions.each do |transaction|
  m = Date.parse(transaction.trade_date).month
  monthly_sums[m] ||= 0
  monthly_sums[m] += transaction.net_amount.to_f
end

binding.pry
