from common import build_volume_shock_continuation_study, emit_metrics, save_study

STUDY_NAME = "sp500_event_driven_v2_volume_shock_continuation"
study = build_volume_shock_continuation_study(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
