"""Basic tests for the dashboard application."""

from src.trade_lab.dashboard import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ == "0.1.0"
