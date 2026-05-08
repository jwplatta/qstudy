"""
Event dates for FOMC rate decisions, CPI releases, and Employment Situation reports.
These are dates we don't want positions to expire on (high volatility events).
"""

from datetime import date


def get_event_dates():
    """
    Returns a set of dates for FOMC rate decisions, CPI releases, and employment reports.
    Covers 2021-2025.
    """
    return {
        # 2021 EVENTS
        # FOMC rate decisions
        date(2021, 1, 27),
        date(2021, 3, 17),
        date(2021, 4, 28),
        date(2021, 6, 16),
        date(2021, 7, 28),
        date(2021, 9, 22),
        date(2021, 11, 3),
        date(2021, 12, 15),
        # CPI releases
        date(2021, 1, 13),
        date(2021, 2, 10),
        date(2021, 3, 10),
        date(2021, 4, 13),
        date(2021, 5, 12),
        date(2021, 6, 10),
        date(2021, 7, 13),
        date(2021, 8, 11),
        date(2021, 9, 14),
        date(2021, 10, 13),
        date(2021, 11, 10),
        date(2021, 12, 10),
        # Employment Situation releases
        date(2021, 1, 8),
        date(2021, 2, 5),
        date(2021, 3, 5),
        date(2021, 4, 2),
        date(2021, 5, 7),
        date(2021, 6, 4),
        date(2021, 7, 2),
        date(2021, 8, 6),
        date(2021, 9, 3),
        date(2021, 10, 8),
        date(2021, 11, 5),
        date(2021, 12, 3),
        # 2022 EVENTS
        # FOMC rate decisions
        date(2022, 1, 26),
        date(2022, 3, 16),
        date(2022, 5, 4),
        date(2022, 6, 15),
        date(2022, 7, 27),
        date(2022, 9, 21),
        date(2022, 11, 2),
        date(2022, 12, 14),
        # CPI releases
        date(2022, 1, 12),
        date(2022, 2, 10),
        date(2022, 3, 10),
        date(2022, 4, 12),
        date(2022, 5, 11),
        date(2022, 6, 10),
        date(2022, 7, 13),
        date(2022, 8, 10),
        date(2022, 9, 13),
        date(2022, 10, 13),
        date(2022, 11, 10),
        date(2022, 12, 13),
        # Employment Situation releases
        date(2022, 1, 7),
        date(2022, 2, 4),
        date(2022, 3, 4),
        date(2022, 4, 1),
        date(2022, 5, 6),
        date(2022, 6, 3),
        date(2022, 7, 8),
        date(2022, 8, 5),
        date(2022, 9, 2),
        date(2022, 10, 7),
        date(2022, 11, 4),
        date(2022, 12, 2),
        # 2023 EVENTS
        # FOMC rate decisions
        date(2023, 2, 1),
        date(2023, 3, 22),
        date(2023, 5, 3),
        date(2023, 6, 14),
        date(2023, 7, 26),
        date(2023, 9, 20),
        date(2023, 11, 1),
        date(2023, 12, 13),
        # CPI releases
        date(2023, 2, 14),
        date(2023, 3, 14),
        date(2023, 4, 12),
        date(2023, 5, 10),
        date(2023, 6, 13),
        date(2023, 7, 12),
        date(2023, 8, 10),
        date(2023, 9, 13),
        date(2023, 10, 12),
        date(2023, 11, 14),
        date(2023, 12, 12),
        # Employment Situation releases
        date(2023, 2, 3),
        date(2023, 3, 10),
        date(2023, 4, 7),
        date(2023, 5, 5),
        date(2023, 6, 2),
        date(2023, 7, 7),
        date(2023, 8, 4),
        date(2023, 9, 1),
        date(2023, 10, 6),
        date(2023, 11, 3),
        date(2023, 12, 8),
        # 2024 EVENTS
        # FOMC rate decisions
        date(2024, 1, 31),
        date(2024, 3, 20),
        date(2024, 5, 1),
        date(2024, 6, 12),
        date(2024, 7, 31),
        date(2024, 9, 18),
        date(2024, 11, 7),
        date(2024, 12, 18),
        # CPI releases
        date(2024, 6, 12),
        date(2024, 7, 11),
        date(2024, 8, 14),
        date(2024, 9, 11),
        date(2024, 10, 10),
        date(2024, 11, 13),
        date(2024, 12, 11),
        # Employment Situation releases
        date(2024, 3, 8),
        date(2024, 5, 3),
        date(2024, 6, 7),
        date(2024, 7, 5),
        date(2024, 9, 6),
        date(2024, 10, 4),
        date(2024, 12, 6),
        # 2025 EVENTS
        # FOMC rate decisions
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
        # CPI releases
        date(2025, 5, 13),
        date(2025, 6, 11),
        date(2025, 7, 15),
        date(2025, 8, 12),
        date(2025, 12, 18),
        # Employment Situation releases
        date(2025, 5, 2),
        date(2025, 6, 6),
        date(2025, 7, 3),
        date(2025, 8, 1),
        date(2025, 12, 16),
    }
