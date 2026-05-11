from common import build_intraday_exhaustion_study, emit_metrics, save_study

STUDY_NAME = "sp500_event_driven_v3_intraday_exhaustion_reversal"
study = build_intraday_exhaustion_study(STUDY_NAME)

if __name__ == "__main__":
    save_study(study, STUDY_NAME)
    emit_metrics(study)
