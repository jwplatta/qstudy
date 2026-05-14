from __future__ import annotations


class QStudyCliError(Exception):
    """Base error for qstudy CLI operations."""


class ConfigError(QStudyCliError):
    """Raised for invalid qstudy config files."""
