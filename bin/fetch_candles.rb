#!/usr/bin/env ruby
# frozen_string_literal: true

# Script to fetch intraday candles for a symbol at specified interval
# Usage: ruby bin/fetch_spx_5_min_candles.rb [SYMBOL] [INTERVAL]
# Example: ruby bin/fetch_spx_5_min_candles.rb $SPX 5
# SYMBOL: default is $SPX
# INTERVAL: 1, 5, or 10 (minutes), default is 5

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
  config.log_level = "INFO"
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

def interval_to_frequency(interval)
  case interval
  when 1
    SchwabRb::PriceHistory::Frequencies::EVERY_MINUTE
  when 5
    SchwabRb::PriceHistory::Frequencies::EVERY_FIVE_MINUTES
  when 10
    SchwabRb::PriceHistory::Frequencies::EVERY_TEN_MINUTES
  else
    raise ArgumentError, "Invalid interval: #{interval}. Must be 1, 5, or 10"
  end
end

def fetch_intraday_candles(symbol, interval)
  today = Date.today

  # Time objects for the start and end of the day
  start_of_day = DateTime.new(today.year, today.month, today.day, 6, 0, 0)
  end_of_day = DateTime.new(today.year, today.month, today.day, 23, 59, 59)

  client = create_client
  puts "Fetching #{interval}-minute candles for #{symbol} on #{today}..."
  puts "Start: #{start_of_day}"
  puts "End: #{end_of_day}"

  price_hist = client.get_price_history(
    symbol,
    period_type: SchwabRb::PriceHistory::PeriodTypes::DAY,
    period: SchwabRb::PriceHistory::Periods::ONE_DAY,
    frequency_type: SchwabRb::PriceHistory::FrequencyTypes::MINUTE,
    frequency: interval_to_frequency(interval),
    start_datetime: start_of_day,
    end_datetime: end_of_day,
    need_extended_hours_data: true,
    need_previous_close: false,
    return_data_objects: true
  )

  puts "Price history data collection complete!"
  puts "Symbol: #{symbol}"
  puts "Date: #{today}"
  puts "Number of candles: #{price_hist.candles.length}" if price_hist.respond_to?(:candles)

  write_candles_to_csv(symbol, interval, today, price_hist)

  price_hist
rescue StandardError => e
  puts "Error fetching price history: #{e.message}"
  puts e.backtrace.first(3)
end

def write_candles_to_csv(symbol, interval, date, price_hist)
  return unless price_hist.respond_to?(:candles) && price_hist.candles.any?

  # Create data directory if it doesn't exist
  data_dir = File.join(Dir.pwd, "data")
  FileUtils.mkdir_p(data_dir)

  # Format: symbol_interval_min_date.csv (e.g., SPX_5_min_2025-12-18.csv)
  clean_symbol = symbol.gsub("$", "")
  filename = "#{clean_symbol}_#{interval}_min_#{date}.csv"
  filepath = File.join(data_dir, filename)

  CSV.open(filepath, "w") do |csv|
    # Write header
    csv << ["datetime", "open", "high", "low", "close", "volume"]

    # Write candle data
    price_hist.candles.each do |candle|
      csv << [
        candle.datetime.strftime("%Y-%m-%d %H:%M:%S"),
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume
      ]
    end
  end

  puts "\nCandles written to: #{filepath}"
  puts "Total candles: #{price_hist.candles.length}"
rescue StandardError => e
  puts "Error writing CSV: #{e.message}"
  puts e.backtrace.first(3)
end

if __FILE__ == $PROGRAM_NAME
  symbol = ARGV[0] || "$SPX"
  interval = (ARGV[1] || "5").to_i

  unless [1, 5, 10].include?(interval)
    puts "Error: Invalid interval '#{interval}'. Must be 1, 5, or 10 minutes."
    puts "Usage: ruby bin/fetch_spx_5_min_candles.rb [SYMBOL] [INTERVAL]"
    puts "Example: ruby bin/fetch_spx_5_min_candles.rb $SPX 5"
    exit 1
  end

  fetch_intraday_candles(symbol, interval)
end
