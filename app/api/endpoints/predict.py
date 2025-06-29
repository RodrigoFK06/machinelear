import traceback

from fastapi import APIRouter, HTTPException
from app.models.schema import PredictRequest, PredictResponse
from app.services.predictor import predict_sequence  # Ya guarda en MongoDB internamente

router = APIRouter()

@router.post("/predict",
             response_model=PredictResponse,
             summary="Predict a medical sign from a sequence of keypoints",
             description="Receives a sequence of keypoints representing a medical sign, processes it, and returns the predicted sign label, confidence, and evaluation. The prediction record is saved to the database."
             )
async def predict(request: PredictRequest):
    try:
        print("📩 Entrada recibida:")
        print(request.dict())
        return await predict_sequence(request)
    except ValueError as e:
        print("⚠️ ValueError en predict():", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("❌ Excepción en predict():", str(e))
        traceback.print_exc()  # 🔥 Esto mostrará el error real en consola
        raise HTTPException(status_code=500, detail=f"Error interno en la predicción: {str(e)}")
