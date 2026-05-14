#!/usr/bin/env ruby
# frozen_string_literal: true

# Script to count contango days for VIX in 2025
# A day is contango if: VIX9D_open < VIX_open
# Usage: ruby bin/vix_contango_days.rb

require "csv"
require "date"
require "pry"

def load_vix_data(filename)
  filepath = File.join(Dir.pwd, "data", filename)

  unless File.exist?(filepath)
    puts "Error: #{filepath} not found. Run bin/fetch_vix_dailys.rb first."
    exit 1
  end

  data = {}
  CSV.foreach(filepath, headers: true) do |row|
    date = Date.parse(row["date"])
    data[date] = row["open"].to_f
  end
  data
end

# Load VIX and VIX9D datasets
vix_data = load_vix_data("VIX_daily_2025.csv")
vix9d_data = load_vix_data("VIX9D_daily_2025.csv")

# Find common dates
common_dates = vix_data.keys & vix9d_data.keys
common_dates.sort!

puts "Analyzing VIX contango days for 2025..."
puts "Total trading days: #{common_dates.length}\n\n"

contango_days = []
total_days = 0

common_dates.each do |date|
  vix_open = vix_data[date]
  vix9d_open = vix9d_data[date]

  # Check contango condition: VIX9D < VIX
  if vix9d_open < vix_open
    contango_days << {
      date: date,
      vix9d: vix9d_open,
      vix: vix_open
    }
  end

  total_days += 1
end

# Display results
puts "Contango Days: #{contango_days.length}"
puts "Percentage: #{(contango_days.length.to_f / total_days * 100).round(2)}%\n\n"

if contango_days.any?
  puts "Dates with contango structure (VIX9D < VIX):"
  puts "-" * 50
  puts sprintf("%-12s  %8s  %8s", "Date", "VIX9D", "VIX")
  puts "-" * 50

  contango_days.each do |day|
    puts sprintf(
      "%-12s  %8.2f  %8.2f",
      day[:date].strftime("%Y-%m-%d"),
      day[:vix9d],
      day[:vix]
    )
  end
end
