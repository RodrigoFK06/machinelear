from .schema import PredictRequest, PredictResponse
from app.predict import predict


def run_prediction(req: PredictRequest) -> PredictResponse:
    label, conf, probs = predict(req.sequence)
    return PredictResponse(predicted_label=label, confidence=conf, probabilities=probs)
