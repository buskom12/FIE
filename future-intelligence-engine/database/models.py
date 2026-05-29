import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Prediction(Base):

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    event = Column(String, nullable=False)
    fie_probability = Column(Float, nullable=False)
    market_probability = Column(Float, nullable=False)
    impact_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    real_outcome = Column(Float, nullable=True)
    error = Column(Float, nullable=True)
