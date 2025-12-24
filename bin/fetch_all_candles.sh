#!/bin/bash
# Script to fetch intraday candles for multiple symbols and intervals
# Usage: bash fetch_all_candles.sh

SYMBOLS=("\$SPX" "\$VIX" "\$VIX1D" "\$VIX9D" "\$SKEW")
INTERVALS=(1 5 10)

DATE="$(date +%F)"
if [ -n "$1" ]; then
  DATE="$1"
fi

for SYMBOL in "${SYMBOLS[@]}"; do
  for INTERVAL in "${INTERVALS[@]}"; do
    echo "Fetching ${INTERVAL}-min candles for ${SYMBOL} on ${DATE}..."
    bundle exec ruby "$(dirname "$0")/fetch_candles.rb" "$SYMBOL" "$INTERVAL" "$DATE"
  done
done
