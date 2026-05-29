from database.db import SessionLocal
from database.models import Prediction


def update_outcome(prediction_id: int, real_outcome: float) -> None:
    session = SessionLocal()

    try:
        prediction = session.query(Prediction).filter(
            Prediction.id == prediction_id
        ).first()

        if prediction:
            prediction.real_outcome = real_outcome
            prediction.error = abs(prediction.fie_probability - real_outcome)
            session.commit()
        else:
            print(f"[outcome_tracker] Prediction {prediction_id} not found.")
    except Exception as e:
        session.rollback()
        print(f"[outcome_tracker] Failed to update: {e}")
    finally:
        session.close()
