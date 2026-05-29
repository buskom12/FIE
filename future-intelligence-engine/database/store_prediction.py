from database.db import SessionLocal
from database.models import Prediction


def save_prediction(
    event: str,
    fie_prob: float,
    market_prob: float,
    impact: float,
) -> None:
    session = SessionLocal()

    try:
        prediction = Prediction(
            event=event,
            fie_probability=fie_prob,
            market_probability=market_prob,
            impact_score=impact,
        )
        session.add(prediction)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"[store_prediction] Failed to save: {e}")
    finally:
        session.close()
