from common import build_gap_fade_study, emit_metrics, save_study

STUDY_NAME = "sp500_event_driven_v1_gap_fade"
study = build_gap_fade_study(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
