from __future__ import annotations

from common import (
    build_gap_fade_study,
    build_intraday_exhaustion_study,
    build_volume_shock_continuation_study,
    emit_metrics,
    save_study,
)

STUDIES = [
    ("sp500_event_driven_v1_gap_fade", build_gap_fade_study),
    ("sp500_event_driven_v2_volume_shock_continuation", build_volume_shock_continuation_study),
    ("sp500_event_driven_v3_intraday_exhaustion_reversal", build_intraday_exhaustion_study),
]


if __name__ == "__main__":
    for name, builder in STUDIES:
        study = builder(name)
        save_study(study, name)
        emit_metrics(study)
