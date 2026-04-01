"""Configuration for the options dashboard."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DEV", "./data"))

DEFAULT_SYMBOL = "SPXW"
DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_DAYS_OUT = 10

STRIKE_WIDTH = 50.0
MULTIPLIER = 100.0
GAMMA_SCALE = 0.01
CANDLE_INTERVAL = 5
SESSION_START = "08:00"
SESSION_END = "15:15"

# Gamma Regime Detection Thresholds
NET_GEX_STRONG_THRESHOLD = 50_000_000  # $50M - defines "strongly positive/negative"
NET_GEX_NEUTRAL_THRESHOLD = 5_000_000  # $5M - defines "near zero"
FLIP_DISTANCE_DEADBAND = 0.002  # 0.2% - deadband for flip distance neutral zone

# Volume Data Settings
ES_VOLUME_DIR = Path(os.getenv("ES_DATA_DIR", DATA_DIR))  # /ES volume data location
DOLLAR_VOLUME_LOOKBACK = 60  # minutes - rolling window for dollar volume calculation
