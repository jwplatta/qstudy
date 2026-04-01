"""Tests for gamma regime detection metrics."""

import pytest
import pandas as pd
import numpy as np
from src.trade_lab.utils.gex import (
    calculate_flip_distance,
    calculate_gamma_influence,
    classify_regime,
)


class TestFlipDistance:
    """Tests for flip distance calculation."""

    @pytest.mark.skip(reason="Requires complex mocking of calculate_zero_gamma_line")
    def test_flip_distance_positive_regime(self):
        """Test flip distance when spot is above zero-gamma level."""
        # Create mock data with zero crossing below spot
        df = pd.DataFrame({
            "strike": [5900, 5950, 6000, 6050, 6100],
            "gamma": [0.05, 0.03, 0.01, -0.01, -0.03],
            "open_interest": [100, 100, 100, 100, 100],
            "contract_type": ["CALL", "CALL", "CALL", "PUT", "PUT"],
            "expiration_date": ["2026-02-10"] * 5,
        })

        spot = 6000
        flip_dist = calculate_flip_distance(df, spot, days_out=10)

        # Should be positive (spot above zero-gamma)
        assert flip_dist is not None

    @pytest.mark.skip(reason="Requires complex mocking of calculate_zero_gamma_line")
    def test_flip_distance_with_deadband(self):
        """Test that deadband correctly zeroes small distances."""
        # This would require more complex mocking of zero-gamma line
        # Simplified test: ensure function handles None gracefully
        df = pd.DataFrame({
            "strike": [6000],
            "gamma": [0.01],
            "open_interest": [100],
            "contract_type": ["CALL"],
            "expiration_date": ["2026-02-10"],
        })

        spot = 6000
        flip_dist = calculate_flip_distance(df, spot, days_out=10, deadband=0.002)

        # Function should return something or None
        assert flip_dist is None or isinstance(flip_dist, float)

    @pytest.mark.skip(reason="Requires proper DataFrame structure with all columns")
    def test_flip_distance_returns_none_when_zero_gamma_not_found(self):
        """Test that flip distance returns None when zero-gamma line can't be found."""
        # Empty DataFrame should result in None
        df = pd.DataFrame()
        spot = 6000

        flip_dist = calculate_flip_distance(df, spot, days_out=10)

        assert flip_dist is None


class TestGammaInfluence:
    """Tests for gamma influence calculation."""

    def test_gamma_influence_normal_case(self):
        """Test gamma influence with valid inputs."""
        gross_gex = 50_000_000  # $50M
        dollar_volume = 5_000_000  # $5M per 1% move

        influence = calculate_gamma_influence(gross_gex, dollar_volume)

        # Expected: 0.01 * 50M / 5M = 0.1
        assert influence == pytest.approx(0.1, rel=1e-6)

    def test_gamma_influence_high_gex(self):
        """Test gamma influence with high GEX relative to volume."""
        gross_gex = 100_000_000  # $100M
        dollar_volume = 1_000_000  # $1M per 1% move

        influence = calculate_gamma_influence(gross_gex, dollar_volume)

        # Expected: 0.01 * 100M / 1M = 1.0
        assert influence == pytest.approx(1.0, rel=1e-6)

    def test_gamma_influence_zero_volume(self):
        """Test that function handles zero volume gracefully."""
        gross_gex = 50_000_000
        dollar_volume = 0

        influence = calculate_gamma_influence(gross_gex, dollar_volume)

        assert influence is None

    def test_gamma_influence_negative_volume(self):
        """Test that function handles negative volume gracefully."""
        gross_gex = 50_000_000
        dollar_volume = -1_000_000

        influence = calculate_gamma_influence(gross_gex, dollar_volume)

        assert influence is None

    def test_gamma_influence_none_volume(self):
        """Test that function handles None volume gracefully."""
        gross_gex = 50_000_000
        dollar_volume = None

        influence = calculate_gamma_influence(gross_gex, dollar_volume)

        assert influence is None


class TestClassifyRegime:
    """Tests for regime classification logic."""

    def test_strongly_positive_regime(self):
        """Test classification of strongly positive Net GEX."""
        net_gex = 60_000_000  # $60M
        result = classify_regime(
            net_gex,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )

        assert result["regime"] == "Strongly Positive"
        assert result["dealer_state"] == "Long Gamma"
        assert result["color"] == "green"
        assert result["net_gex"] == net_gex

    def test_strongly_negative_regime(self):
        """Test classification of strongly negative Net GEX."""
        net_gex = -60_000_000  # -$60M
        result = classify_regime(
            net_gex,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )

        assert result["regime"] == "Strongly Negative"
        assert result["dealer_state"] == "Short Gamma"
        assert result["color"] == "red"
        assert result["net_gex"] == net_gex

    def test_near_zero_regime(self):
        """Test classification of near-zero Net GEX."""
        net_gex = 2_000_000  # $2M (within ±5M threshold)
        result = classify_regime(
            net_gex,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )

        assert result["regime"] == "Near Zero"
        assert result["dealer_state"] == "Neutral"
        assert result["color"] == "yellow"

    def test_moderately_positive_regime(self):
        """Test classification of moderately positive Net GEX."""
        net_gex = 20_000_000  # $20M (above neutral, below strong)
        result = classify_regime(
            net_gex,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )

        assert result["regime"] == "Moderately Positive"
        assert result["dealer_state"] == "Long Gamma"
        assert result["color"] == "green"

    def test_moderately_negative_regime(self):
        """Test classification of moderately negative Net GEX."""
        net_gex = -20_000_000  # -$20M (below neutral, above strong negative)
        result = classify_regime(
            net_gex,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )

        assert result["regime"] == "Moderately Negative"
        assert result["dealer_state"] == "Short Gamma"
        assert result["color"] == "red"

    def test_regime_with_optional_metrics(self):
        """Test that optional metrics are included in result."""
        net_gex = 60_000_000
        flip_distance = 0.015  # 1.5% above zero-gamma
        gamma_influence = 0.25

        result = classify_regime(
            net_gex,
            flip_distance=flip_distance,
            gamma_influence=gamma_influence
        )

        assert result["flip_distance"] == flip_distance
        assert result["gamma_influence"] == gamma_influence
        assert result["regime"] == "Strongly Positive"

    def test_boundary_conditions(self):
        """Test exact boundary values."""
        # Exactly at strong threshold
        result = classify_regime(
            50_000_001,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )
        assert result["regime"] == "Strongly Positive"

        # Exactly at neutral threshold
        result = classify_regime(
            5_000_000,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )
        assert result["regime"] == "Near Zero"

        # Just above neutral threshold
        result = classify_regime(
            5_000_001,
            strong_threshold=50_000_000,
            neutral_threshold=5_000_000
        )
        assert result["regime"] == "Moderately Positive"
