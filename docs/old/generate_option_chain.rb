#!/usr/bin/env ruby
# frozen_string_literal: true

require "csv"
require "date"
require "optparse"
require "pathname"
require "time"
require "pry"

class GenerateOptionChain
  Candle = Struct.new(:timestamp, :open, :high, :low, :close, :volume, keyword_init: true)
  ChainRow = Struct.new(:contract_type, :strike, :mark, :delta, :actual_iv, :raw, keyword_init: true)
  OptionQuote = Struct.new(:contract_type, :strike, :vol, :price, :delta, keyword_init: true)

  DEFAULT_REFERENCE_CHAIN = File.expand_path(
    "~/.tickrake/data/options/schwab/SPXW_exp2026-04-10_2026-04-09_12-19-44.csv"
  )
  DEFAULT_OUTPUT_PATH = File.expand_path("tmp/synthetic_spxw_chain_2026-04-10.csv", Dir.pwd)
  DEFAULT_EXPIRY = Date.new(2026, 4, 10)
  DEFAULT_TIMESTAMP = Time.new(2026, 4, 9, 12, 19, 44, "-05:00")
  HISTORY_DIR = Pathname.new(File.expand_path("~/.tickrake/data/history/ibkr-paper"))
  CANDLE_FILES = {
    "SPX" => HISTORY_DIR.join("SPX_1min.csv"),
    "VIX" => HISTORY_DIR.join("VIX_1min.csv"),
    "VIX9D" => HISTORY_DIR.join("VIX9D_1min.csv"),
    "VIX1D" => HISTORY_DIR.join("VIX1D_1min.csv")
  }.freeze
  PUT_ANCHOR_DELTAS = [0.05, 0.10, 0.25].freeze
  CALL_ANCHOR_DELTAS = [0.45, 0.25, 0.2, 0.15, 0.10, 0.05, 0.02, 0.01].freeze

  def initialize(argv)
    @options = parse_options(argv)
  end

  def run
    validate_inputs!

    candles = load_market_candles
    snapshot = build_snapshot(candles)
    derived_state = compute_derived_state(candles, snapshot)
    atm_vol = compute_atm_vol(snapshot, derived_state)
    anchor_points = build_anchor_points(snapshot[:spot], derived_state, atm_vol, snapshot[:tau_years])
    reference_chain = load_reference_chain
    quotes = build_synthetic_chain(reference_chain, snapshot, anchor_points)

    enforce_no_arbitrage!(quotes, snapshot[:spot])
    write_output(quotes)
    print_summary(snapshot, derived_state, atm_vol, anchor_points, quotes)
    print_validation(reference_chain, quotes, anchor_points, snapshot[:spot]) if @options[:validate]
  end

  private

  def parse_options(argv)
    options = {
      expiry: DEFAULT_EXPIRY,
      timestamp: DEFAULT_TIMESTAMP,
      reference_chain: DEFAULT_REFERENCE_CHAIN,
      output: DEFAULT_OUTPUT_PATH,
      risk_free_rate: 0.0,
      validate: true
    }

    OptionParser.new do |parser|
      parser.banner = "Usage: ruby bin/generate_option_chain.rb [options]"

      parser.on("--expiry YYYY-MM-DD", "Target expiry (default: #{DEFAULT_EXPIRY})") do |value|
        options[:expiry] = Date.parse(value)
      end

      parser.on("--timestamp ISO8601", "Evaluation timestamp (default: #{DEFAULT_TIMESTAMP.iso8601})") do |value|
        options[:timestamp] = Time.parse(value)
      end

      parser.on("--reference-chain PATH", "Reference chain CSV used only for strike grid/validation") do |value|
        options[:reference_chain] = File.expand_path(value)
      end

      parser.on("--output PATH", "Output CSV path") do |value|
        options[:output] = File.expand_path(value)
      end

      parser.on("--rate FLOAT", Float, "Risk-free rate as annual decimal (default: 0.0)") do |value|
        options[:risk_free_rate] = value
      end

      parser.on("--[no-]validate", "Enable validation against the reference chain (default: true)") do |value|
        options[:validate] = value
      end
    end.parse!(argv)

    options
  end

  def validate_inputs!
    raise "Reference chain not found: #{@options[:reference_chain]}" unless File.exist?(@options[:reference_chain])
    raise "Missing SPX candle history: #{CANDLE_FILES.fetch("SPX")}" unless CANDLE_FILES.fetch("SPX").exist?
    raise "Missing VIX candle history: #{CANDLE_FILES.fetch("VIX")}" unless CANDLE_FILES.fetch("VIX").exist?
    raise "Missing VIX9D candle history: #{CANDLE_FILES.fetch("VIX9D")}" unless CANDLE_FILES.fetch("VIX9D").exist?
  end

  def load_market_candles
    CANDLE_FILES.each_with_object({}) do |(symbol, path), loaded|
      loaded[symbol] = path.exist? ? load_candles(path) : []
    end
  end

  def load_candles(path)
    rows = []

    CSV.foreach(path, headers: true) do |row|
      rows << Candle.new(
        timestamp: Time.parse(row.fetch("datetime")),
        open: row.fetch("open").to_f,
        high: row.fetch("high").to_f,
        low: row.fetch("low").to_f,
        close: row.fetch("close").to_f,
        volume: row.fetch("volume").to_f
      )
    end

    rows
  end

  def build_snapshot(candles)
    snapshot = {}
    target_time = @options[:timestamp]

    %w[SPX VIX VIX9D VIX1D].each do |symbol|
      next if candles.fetch(symbol).empty?

      candle = nearest_prior_candle(candles.fetch(symbol), target_time)
      snapshot[symbol] = candle
    end

    expiry_close = Time.new(@options[:expiry].year, @options[:expiry].month, @options[:expiry].day, 15, 0, 0, "-05:00")
    tau_seconds = [expiry_close - target_time, 60.0].max
    snapshot[:tau_years] = tau_seconds / (365.25 * 24 * 60 * 60)
    snapshot[:spot] = snapshot.fetch("SPX").close
    snapshot
  end

  def nearest_prior_candle(candles, target_time)
    candidate = candles.bsearch { |candle| candle.timestamp > target_time }
    if candidate.nil?
      candles.last
    else
      index = candles.index(candidate)
      raise "No candle found before #{target_time.iso8601}" if index.zero?

      candles[index - 1]
    end
  end

  def compute_derived_state(candles, snapshot)
    spx_candles = candles.fetch("SPX")
    target_time = @options[:timestamp]
    current_day = snapshot.fetch("SPX").timestamp.to_date
    current_day_candles = spx_candles.select { |c| c.timestamp.to_date == current_day && c.timestamp <= target_time }
    prior_day = current_day - 1
    prior_day_candles = spx_candles.select { |c| c.timestamp.to_date == prior_day }
    previous_days = grouped_daily_candles(spx_candles).select { |date, _| date < current_day }.to_a.last(5).to_h

    vix = snapshot.fetch("VIX").close
    vix9d = snapshot.fetch("VIX9D").close
    short_vol_proxy = snapshot["VIX1D"]&.close || vix9d

    prior_close = prior_day_candles.last&.close || snapshot.fetch("SPX").open
    overnight_open = current_day_candles.first&.open || snapshot.fetch("SPX").open
    prior_day_range = percent_range(prior_day_candles)
    rolling_ranges = previous_days.values.last(5).map { |day_candles| percent_range(day_candles) }.compact

    {
      short_vol_proxy: short_vol_proxy,
      medium_vol_proxy: vix,
      term_slope_abs: short_vol_proxy - vix,
      term_slope_ratio: vix.zero? ? 1.0 : short_vol_proxy / vix,
      prior_day_range_pct: prior_day_range || 0.0,
      rolling_range_pct: rolling_ranges.empty? ? (prior_day_range || 0.0) : rolling_ranges.sum / rolling_ranges.length,
      prior_return_pct: prior_close.zero? ? 0.0 : ((snapshot.fetch("SPX").close / prior_close) - 1.0) * 100.0,
      overnight_gap_pct: prior_close.zero? ? 0.0 : ((overnight_open / prior_close) - 1.0) * 100.0
    }
  end

  def grouped_daily_candles(candles)
    candles.group_by { |candle| candle.timestamp.to_date }.sort.to_h
  end

  def percent_range(day_candles)
    return nil if day_candles.nil? || day_candles.empty?

    low = day_candles.map(&:low).min
    high = day_candles.map(&:high).max
    close = day_candles.last.close
    return 0.0 if close.zero?

    ((high - low) / close) * 100.0
  end

  def compute_atm_vol(snapshot, derived_state)
    short = derived_state.fetch(:short_vol_proxy)
    medium = derived_state.fetch(:medium_vol_proxy)
    tau_trading_days = snapshot.fetch(:tau_years) * 252.0
    short_weight = tau_trading_days <= 1.25 ? 0.85 : 0.65
    blended = (short * short_weight) + (medium * (1.0 - short_weight))
    blended.clamp(8.0, 120.0)
  end

  def build_anchor_points(spot, derived_state, atm_vol_pct, tau_years)
    atm_vol = atm_vol_pct / 100.0
    slope_abs = derived_state.fetch(:term_slope_abs)
    slope_ratio = derived_state.fetch(:term_slope_ratio)
    realized_pressure = derived_state.fetch(:prior_day_range_pct) - derived_state.fetch(:rolling_range_pct)
    downside_intensity = 0.035 + [(-slope_abs * 0.012), 0.0].max + [realized_pressure * 0.010, 0.0].max
    upside_intensity = 0.006 + [(slope_abs * 0.003), 0.0].max
    curvature = 0.008 + [(1.0 - slope_ratio) * 0.012, 0.0].max

    points = []
    PUT_ANCHOR_DELTAS.each do |absolute_delta|
      strike = strike_for_target_delta(spot, absolute_delta, :put, atm_vol, tau_years)
      wing = (0.50 - absolute_delta) / 0.45
      vol = atm_vol_pct * (1.0 + downside_intensity * wing + curvature * (wing**2))
      points << { type: :put, abs_delta: absolute_delta, x: Math.log(strike / spot), strike: strike, vol: vol }
    end

    points << { type: :atm, abs_delta: 0.50, x: 0.0, strike: spot, vol: atm_vol_pct }

    CALL_ANCHOR_DELTAS.each do |absolute_delta|
      strike = strike_for_target_delta(spot, absolute_delta, :call, atm_vol, tau_years)
      wing = (0.50 - absolute_delta) / 0.45
      vol = atm_vol_pct * (1.0 + upside_intensity * wing + (curvature * 0.25) * (wing**2))
      vol *= 1.15 if absolute_delta < 0.05
      points << { type: :call, abs_delta: absolute_delta, x: Math.log(strike / spot), strike: strike, vol: vol }
    end

    points.sort_by { |point| point[:x] }
  end

  def strike_for_target_delta(spot, absolute_delta, contract_type, sigma, tau_years)
    sqrt_t = Math.sqrt(tau_years)
    d1 =
      if contract_type == :call
        inverse_normal_cdf(absolute_delta)
      else
        -inverse_normal_cdf(absolute_delta)
      end

    exponent = -(d1 * sigma * sqrt_t - 0.5 * sigma * sigma * tau_years)
    spot * Math.exp(exponent)
  end

  def load_reference_chain
    CSV.read(@options[:reference_chain], headers: true).map do |row|
      contract_type = row.fetch("contract_type").to_s.downcase
      actual_iv = select_actual_iv(row)

      ChainRow.new(
        contract_type: contract_type == "call" ? "call" : "put",
        strike: row.fetch("strike").to_f,
        mark: row["mark"]&.to_f,
        delta: row["delta"]&.to_f,
        actual_iv: actual_iv,
        raw: row.to_h
      )
    end
  end

  def select_actual_iv(row)
    observed = row["volatility"]&.to_f
    theoretical = row["theoretical_volatility"]&.to_f
    return observed if observed && observed.between?(1.0, 150.0)
    return theoretical if theoretical && theoretical.between?(1.0, 150.0)

    nil
  end

  def build_synthetic_chain(reference_chain, snapshot, anchor_points)
    reference_chain.map do |contract|
      vol_pct = interpolate_vol(anchor_points, snapshot[:spot], contract.strike)
      sigma = vol_pct / 100.0
      price = black_scholes_price(
        spot: snapshot[:spot],
        strike: contract.strike,
        tau_years: snapshot[:tau_years],
        sigma: sigma,
        rate: @options[:risk_free_rate],
        contract_type: contract.contract_type.to_sym
      )
      delta = black_scholes_delta(
        spot: snapshot[:spot],
        strike: contract.strike,
        tau_years: snapshot[:tau_years],
        sigma: sigma,
        rate: @options[:risk_free_rate],
        contract_type: contract.contract_type.to_sym
      )

      OptionQuote.new(
        contract_type: contract.contract_type,
        strike: contract.strike,
        vol: vol_pct,
        price: price,
        delta: delta
      )
    end.sort_by { |quote| [quote.contract_type, quote.strike] }
  end

  def interpolate_vol(anchor_points, spot, strike)
    x = Math.log(strike / spot)
    points = anchor_points.sort_by { |point| point[:x] }
    return points.first[:vol] if x <= points.first[:x]
    return points.last[:vol] if x >= points.last[:x]

    left, right = points.each_cons(2).find { |a, b| x >= a[:x] && x <= b[:x] }
    weight = (x - left[:x]) / (right[:x] - left[:x])
    vol = left[:vol] + weight * (right[:vol] - left[:vol])
    vol.clamp(5.0, 200.0)
  end

  def enforce_no_arbitrage!(quotes, spot)
    calls = quotes.select { |quote| quote.contract_type == "call" }.sort_by(&:strike)
    puts_ = quotes.select { |quote| quote.contract_type == "put" }.sort_by(&:strike)

    calls.each_with_index do |quote, index|
      intrinsic = [spot - quote.strike, 0.0].max
      quote.price = [quote.price, intrinsic].max
      next if index.zero?

      quote.price = [quote.price, calls[index - 1].price].min
    end

    puts_.each_with_index do |quote, index|
      intrinsic = [quote.strike - spot, 0.0].max
      quote.price = [quote.price, intrinsic].max
      next if index.zero?

      quote.price = [quote.price, puts_[index - 1].price].max
    end
  end

  def write_output(quotes)
    output_path = Pathname.new(@options[:output])
    output_path.dirname.mkpath

    CSV.open(output_path, "w") do |csv|
      csv << %w[contract_type strike_price vol delta price]
      quotes.each do |quote|
        csv << [
          quote.contract_type,
          format("%.4f", quote.strike),
          format("%.4f", quote.vol),
          format("%.6f", quote.delta),
          format("%.4f", quote.price)
        ]
      end
    end
  end

  def print_summary(snapshot, derived_state, atm_vol, anchor_points, quotes)
    puts "Synthetic SPXW chain generated"
    puts "  output: #{@options[:output]}"
    puts "  reference_chain: #{@options[:reference_chain]}"
    puts "  timestamp_used: #{snapshot.fetch("SPX").timestamp.iso8601}"
    puts "  expiry: #{@options[:expiry]}"
    puts "  spot: #{format("%.2f", snapshot[:spot])}"
    puts "  vix: #{format("%.2f", snapshot.fetch("VIX").close)}"
    puts "  vix9d: #{format("%.2f", snapshot.fetch("VIX9D").close)}"
    puts "  vix1d: #{snapshot["VIX1D"] ? format("%.2f", snapshot.fetch("VIX1D").close) : "n/a"}"
    puts "  tau_years: #{format("%.6f", snapshot[:tau_years])}"
    puts "  atm_vol_pct: #{format("%.2f", atm_vol)}"
    puts "  term_slope_abs: #{format("%.2f", derived_state[:term_slope_abs])}"
    puts "  prior_day_range_pct: #{format("%.2f", derived_state[:prior_day_range_pct])}"
    puts "  rolling_range_pct: #{format("%.2f", derived_state[:rolling_range_pct])}"
    puts "  prior_return_pct: #{format("%.2f", derived_state[:prior_return_pct])}"
    puts "  overnight_gap_pct: #{format("%.2f", derived_state[:overnight_gap_pct])}"
    puts "  contracts: #{quotes.length}"
    puts "Anchor surface"
    anchor_points.each do |point|
      label = point[:type] == :atm ? "atm" : "#{point[:type]}_#{(point[:abs_delta] * 100).round}"
      puts "  #{label.ljust(8)} strike=#{format("%.2f", point[:strike])} vol=#{format("%.2f", point[:vol])}"
    end
  end

  def print_validation(reference_chain, quotes, anchor_points, spot)
    quote_map = quotes.each_with_object({}) { |quote, memo| memo[[quote.contract_type, quote.strike]] = quote }
    price_errors = []

    reference_chain.each do |row|
      quote = quote_map[[row.contract_type, row.strike]]
      next unless row.mark && quote

      price_errors << (quote.price - row.mark)
    end

    atm_row = nearest_row_by_strike(reference_chain, spot)
    atm_quote = atm_row && quote_map[[atm_row.contract_type, atm_row.strike]]

    puts "Validation"
    if price_errors.empty?
      puts "  no comparable prices available"
    else
      mae = price_errors.map(&:abs).sum / price_errors.length
      rmse = Math.sqrt(price_errors.map { |error| error**2 }.sum / price_errors.length)
      puts "  price_mae: #{format("%.4f", mae)}"
      puts "  price_rmse: #{format("%.4f", rmse)}"
    end

    if atm_row && atm_quote && atm_row.mark
      puts "  atm_error: #{format("%.4f", atm_quote.price - atm_row.mark)}"
    end

    %w[put_25 put_10 call_25 call_10 call_5].each do |label|
      diagnostic = compare_wing(reference_chain, quote_map, anchor_points, label)
      next unless diagnostic

      puts "  #{label}_vol_diff: #{format("%.4f", diagnostic[:synthetic] - diagnostic[:actual])}"
    end
  end

  def nearest_row_by_strike(rows, target_strike)
    rows.min_by { |row| (row.strike - target_strike).abs }
  end

  def compare_wing(reference_chain, quote_map, anchor_points, label)
    contract_type, delta_value = label.split("_")
    absolute_delta = delta_value.to_i / 100.0
    anchor_type = contract_type == "call" ? :call : :put
    anchor = anchor_points.find { |point| point[:type] == anchor_type && (point[:abs_delta] - absolute_delta).abs < 1e-6 }
    return nil unless anchor

    actual_row = reference_chain
      .select { |row| row.contract_type == contract_type && row.actual_iv }
      .min_by { |row| ((row.delta || 0.0).abs - absolute_delta).abs }
    return nil unless actual_row

    quote = quote_map[[actual_row.contract_type, actual_row.strike]]
    return nil unless quote

    { synthetic: quote.vol, actual: actual_row.actual_iv }
  end

  def black_scholes_price(spot:, strike:, tau_years:, sigma:, rate:, contract_type:)
    intrinsic =
      if contract_type == :call
        [spot - strike, 0.0].max
      else
        [strike - spot, 0.0].max
      end
    return intrinsic if tau_years <= 0.0 || sigma <= 0.0

    d1 = d1(spot, strike, tau_years, sigma, rate)
    d2 = d1 - sigma * Math.sqrt(tau_years)
    discount = Math.exp(-rate * tau_years)

    if contract_type == :call
      (spot * normal_cdf(d1)) - (strike * discount * normal_cdf(d2))
    else
      (strike * discount * normal_cdf(-d2)) - (spot * normal_cdf(-d1))
    end
  end

  def black_scholes_delta(spot:, strike:, tau_years:, sigma:, rate:, contract_type:)
    return(contract_type == :call ? (spot > strike ? 1.0 : 0.0) : (spot < strike ? -1.0 : 0.0)) if tau_years <= 0.0 || sigma <= 0.0

    d1_value = d1(spot, strike, tau_years, sigma, rate)
    contract_type == :call ? normal_cdf(d1_value) : (normal_cdf(d1_value) - 1.0)
  end

  def d1(spot, strike, tau_years, sigma, rate)
    numerator = Math.log(spot / strike) + ((rate + 0.5 * sigma * sigma) * tau_years)
    denominator = sigma * Math.sqrt(tau_years)
    numerator / denominator
  end

  def normal_cdf(value)
    0.5 * (1.0 + Math.erf(value / Math.sqrt(2.0)))
  end

  def inverse_normal_cdf(probability)
    raise ArgumentError, "Probability must be between 0 and 1" unless probability.positive? && probability < 1.0

    low = -10.0
    high = 10.0

    100.times do
      mid = 0.5 * (low + high)
      if normal_cdf(mid) < probability
        low = mid
      else
        high = mid
      end
    end

    0.5 * (low + high)
  end
end

GenerateOptionChain.new(ARGV).run if __FILE__ == $PROGRAM_NAME
