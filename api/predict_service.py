import csv
from datetime import datetime
from .schema import PredictRequest, PredictResponse
from app.predict import predict
from app.config import INFERENCE_LOG_PATH


def run_prediction(req: PredictRequest) -> PredictResponse:
    label, conf, probs = predict(req.sequence)
    with open(INFERENCE_LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            req.nickname or '',
            label,
            conf,
            req.expected_label or '',
        ])
    return PredictResponse(predicted_label=label, confidence=conf, probabilities=probs)
