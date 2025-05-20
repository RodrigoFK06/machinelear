from fastapi import APIRouter, HTTPException
from app.models.schema import PredictRequest, PredictResponse
from app.services.predictor import predict_sequence  # Ya guarda en MongoDB internamente

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        return await predict_sequence(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno en la predicción: {str(e)}")
