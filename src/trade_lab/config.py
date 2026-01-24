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
