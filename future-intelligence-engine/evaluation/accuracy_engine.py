import numpy as np

from database.db import SessionLocal
from database.models import Prediction


def calculate_accuracy() -> dict | None:
    session = SessionLocal()

    try:
        predictions = session.query(Prediction).filter(
            Prediction.real_outcome.isnot(None)
        ).all()
    except Exception as e:
        print(f"[accuracy_engine] DB error: {e}")
        return None
    finally:
        session.close()

    errors = [p.error for p in predictions if p.error is not None]

    if not errors:
        return None

    return {
        "mean_error": round(float(np.mean(errors)), 4),
        "max_error": round(float(np.max(errors)), 4),
        "min_error": round(float(np.min(errors)), 4),
        "predictions_count": len(errors),
    }
