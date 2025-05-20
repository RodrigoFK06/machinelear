from fastapi import APIRouter
from app.api.endpoints import predict, labels, records, progress

router = APIRouter()

# Registrar rutas
router.include_router(predict.router, tags=["Predicción"])
router.include_router(labels.router, tags=["Etiquetas"])
router.include_router(records.router, tags=["Registros"])
router.include_router(progress.router, tags=["Registros"])