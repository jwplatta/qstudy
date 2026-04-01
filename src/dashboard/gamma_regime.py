"""
Gamma Regime Detection Dashboard

A Streamlit dashboard for detecting and visualizing gamma regimes using
Net GEX, Flip Distance, and Gamma Influence metrics.

Usage:
    streamlit run src/dashboard/gamma_regime.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trade_lab.charts.GEXPrice import GEXPrice
from trade_lab.charts.GEXStrike import GEXStrike
from config import (
    DATA_DIR,
    DEFAULT_DAYS_OUT,
    DEFAULT_SYMBOL,
    DOLLAR_VOLUME_LOOKBACK,
    ES_VOLUME_DIR,
    FLIP_DISTANCE_DEADBAND,
    GAMMA_SCALE,
    MULTIPLIER,
    NET_GEX_NEUTRAL_THRESHOLD,
    NET_GEX_STRONG_THRESHOLD,
    STRIKE_WIDTH,
)
from trade_lab.utils.gex import (
    calculate_flip_distance,
    calculate_gamma_influence,
    classify_regime,
    row_gross_gex,
)
from trade_lab.utils.volume import calculate_dollar_volume, load_es_volume

st.set_page_config(
    page_title="Gamma Regime Detection",
    page_icon="\u03b3",
    layout="wide",
    initial_sidebar_state="expanded",
)


def find_latest_date(data_dir, symbol="SPXW"):
    """Find the most recent date with option data available."""
    csv_files = list(Path(data_dir).glob(f"{symbol}_exp*.csv"))
    if not csv_files:
        return None

    # Extract dates from filenames
    dates = []
    for f in csv_files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            try:
                # Format: SYMBOL_expYYYY-MM-DD_YYYY-MM-DD_HH-MM-SS
                fetch_date = parts[2]
                dates.append(fetch_date)
            except Exception:
                continue

    if not dates:
        return None

    return max(dates)


def load_option_data(symbol, date, data_dir):
    """Load and aggregate option chain data for given symbol and date."""
    data_path = Path(data_dir)
    pattern = f"{symbol}_exp*_{date}_*.csv"
    files = list(data_path.glob(pattern))

    if not files:
        return None, None

    # Group files by expiration
    exp_files = {}
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            exp_date = parts[1].replace("exp", "")
            if exp_date not in exp_files:
                exp_files[exp_date] = []
            exp_files[exp_date].append(f)

    # Load most recent file per expiration
    dfs = []
    for exp_date, file_list in exp_files.items():
        # Sort by timestamp (last parts of filename)
        latest_file = sorted(file_list)[-1]
        try:
            df = pd.read_csv(latest_file)
            df["expiration_date"] = exp_date
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {latest_file.name}: {e}")
            continue

    if not dfs:
        return None, None

    # Concatenate all expirations
    all_opts = pd.concat(dfs, ignore_index=True)

    # Convert to numeric
    for col in ["strike", "open_interest", "gamma", "underlying_price"]:
        if col in all_opts.columns:
            all_opts[col] = pd.to_numeric(all_opts[col], errors="coerce")

    # Get spot price
    spot = all_opts["underlying_price"].iloc[0] if "underlying_price" in all_opts.columns else None  # noqa: E501

    return all_opts, spot


def calculate_net_gex(df, spot, multiplier, gamma_scale):
    """Calculate net GEX (calls - puts)."""
    if df is None or df.empty:
        return 0.0

    # Add gross GEX column
    df = df.copy()
    df["gross_gex"] = row_gross_gex(df, spot, multiplier, gamma_scale)

    # Split by contract type
    calls = df[df["contract_type"] == "CALL"]
    puts = df[df["contract_type"] == "PUT"]

    # Net GEX = calls - puts
    call_gex = calls["gross_gex"].sum() if not calls.empty else 0
    put_gex = puts["gross_gex"].sum() if not puts.empty else 0

    return call_gex - put_gex


def calculate_gross_gex(df, spot, multiplier, gamma_scale):
    """Calculate gross (total unsigned) GEX."""
    if df is None or df.empty:
        return 0.0

    df = df.copy()
    df["gross_gex"] = row_gross_gex(df, spot, multiplier, gamma_scale)

    return df["gross_gex"].sum()


# Sidebar controls
st.sidebar.title("Controls")

# Date picker
latest_date = find_latest_date(DATA_DIR, DEFAULT_SYMBOL)
if latest_date:
    default_date = datetime.strptime(latest_date, "%Y-%m-%d")
else:
    default_date = datetime.now()

selected_date = st.sidebar.date_input(
    "Date", value=default_date, help="Select date to analyze (defaults to latest available)"
)

# Symbol selector (SPXW only for now)
symbol = st.sidebar.selectbox(
    "Symbol", options=[DEFAULT_SYMBOL], help="Trading symbol (SPXW only currently supported)"
)

# Days out slider
days_out = st.sidebar.slider(
    "Days Out",
    min_value=1,
    max_value=45,
    value=DEFAULT_DAYS_OUT,
    help="Maximum days to expiration to include in calculations",
)

# Strike window slider
strike_window = st.sidebar.slider(
    "Strike Window",
    min_value=25,
    max_value=200,
    value=int(STRIKE_WIDTH),
    step=25,
    help="Strike range (±) around spot price",
)

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.selectbox(
        "Refresh interval", options=["30s", "1min", "5min"], index=1
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Last updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Main content
st.title("\u03b3 Gamma Regime Detection Dashboard")

# Load data
date_str = selected_date.strftime("%Y-%m-%d")
all_opts, spot = load_option_data(symbol, date_str, DATA_DIR)

if all_opts is None or spot is None:
    st.error(f"No data found for {symbol} on {date_str}")
    st.info(f"Looking for files matching: {symbol}_exp*_{date_str}_*.csv in {DATA_DIR}")
    st.stop()

# Calculate regime metrics
net_gex = calculate_net_gex(all_opts, spot, MULTIPLIER, GAMMA_SCALE)
gross_gex = calculate_gross_gex(all_opts, spot, MULTIPLIER, GAMMA_SCALE)

# Calculate flip distance
flip_distance = calculate_flip_distance(
    all_opts, spot, days_out=days_out, deadband=FLIP_DISTANCE_DEADBAND
)

# Load volume data and calculate gamma influence
volume_df = load_es_volume(date_str, data_dir=str(ES_VOLUME_DIR))
if volume_df is not None and not volume_df.empty:
    dollar_volume_series = calculate_dollar_volume(
        volume_df, lookback_minutes=DOLLAR_VOLUME_LOOKBACK
    )
    # Use most recent dollar volume
    dollar_volume = dollar_volume_series.iloc[-1] if dollar_volume_series is not None else None  # noqa: E501
    gamma_influence = calculate_gamma_influence(gross_gex, dollar_volume)
else:
    dollar_volume = None
    gamma_influence = None

# Classify regime
regime = classify_regime(
    net_gex,
    flip_distance=flip_distance,
    gamma_influence=gamma_influence,
    strong_threshold=NET_GEX_STRONG_THRESHOLD,
    neutral_threshold=NET_GEX_NEUTRAL_THRESHOLD,
)

# Regime scorecard
st.markdown("## Regime Scorecard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Net GEX", f"${net_gex / 1e6:.1f}M", help="Net Gamma Exposure (calls - puts)")

with col2:
    if flip_distance is not None:
        flip_pct = flip_distance * 100
        st.metric("Flip Distance", f"{flip_pct:+.2f}%", help="Distance from zero-gamma level")
    else:
        st.metric("Flip Distance", "N/A", help="Could not calculate zero-gamma line")

with col3:
    if gamma_influence is not None:
        st.metric(
            "Gamma Influence",
            f"{gamma_influence:.3f}",
            help="Gamma hedging impact relative to volume",
        )
    else:
        st.metric("Gamma Influence", "N/A", help="Volume data not available")

with col4:
    # Color-coded regime classification
    color_map = {
        "green": "#28a745",
        "yellow": "#ffc107",
        "red": "#dc3545",
    }
    regime_color = color_map.get(regime["color"], "#6c757d")

    st.markdown(
        f"""
        <div style="background-color: {regime_color}; padding: 20px; border-radius: 5px;">
            <h3 style="color: white; margin: 0;">{regime["regime"]}</h3>
            <p style="color: white; margin: 5px 0 0 0; font-size: 14px;">
                {regime["dealer_state"]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Regime details
with st.expander("Regime Details", expanded=False):
    st.markdown(f"**Dealer Hedging Behavior:** {regime['hedging_behavior']}")
    st.markdown(f"**Expected Market Behavior:** {regime['market_behavior']}")

    # Show regime classification table
    st.markdown("### Regime Classification Reference")
    regime_table = pd.DataFrame(
        {
            "Net GEX Regime": ["Strongly Positive", "Near Zero", "Strongly Negative"],
            "Dealer Gamma State": ["Long Gamma", "Neutral", "Short Gamma"],
            "Hedging Behavior": [
                "Sell into strength, buy into weakness",
                "Minimal mechanical hedging impact",
                "Buy into strength, sell into weakness",
            ],
            "Market Behavior": [
                "Volatility suppressed, mean reversion, pinning",
                "Price responds to flows/news, mixed behavior",
                "Trend risk, acceleration, squeezes, cascades",
            ],
        }
    )
    st.dataframe(regime_table, use_container_width=True)

# Charts section
st.markdown("## Gamma Exposure Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Current Expiry GEX (by Strike)")
    try:
        gex_strike = GEXStrike(
            symbol=symbol, start_date=date_str, days_out=days_out, data_dir=str(DATA_DIR)
        )  # noqa: E501
        fig_strike, ax_strike = gex_strike.plot(figsize=(8, 6))
        st.pyplot(fig_strike)
    except Exception as e:
        st.error(f"Error generating GEX Strike chart: {e}")

with chart_col2:
    st.markdown("### Aggregated GEX (by Price)")
    try:
        gex_price = GEXPrice(
            symbol=symbol, start_date=date_str, days_out=days_out, data_dir=str(DATA_DIR)
        )  # noqa: E501
        fig_price, ax_price = gex_price.plot(figsize=(8, 6))
        st.pyplot(fig_price)
    except Exception as e:
        st.error(f"Error generating GEX Price chart: {e}")

# Footer
st.markdown("---")
st.caption(f"Data loaded from: {DATA_DIR}")
st.caption(f"Symbol: {symbol} | Date: {date_str} | Spot: ${spot:.2f}" if spot else "")
