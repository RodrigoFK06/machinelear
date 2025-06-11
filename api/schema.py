from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    sequence: List[List[float]]
    nickname: str | None = None
    expected_label: str | None = None


class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: List[float]
