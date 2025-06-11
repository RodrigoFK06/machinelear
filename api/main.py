from fastapi import FastAPI
from .schema import PredictRequest, PredictResponse
from .predict_service import run_prediction

app = FastAPI(title="Gesture API")


@app.post('/predict', response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    res = run_prediction(req)
    return res
