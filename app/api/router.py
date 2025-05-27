from fastapi import APIRouter
from app.api.endpoints import predict, labels, records, progress, activity, statistics

router = APIRouter()

# Registrar rutas
router.include_router(predict.router, tags=["Predicción"])
router.include_router(labels.router, tags=["Etiquetas"])
router.include_router(records.router, tags=["Registros"])
router.include_router(progress.router, tags=["Registros"]) # Progress also uses "Registros" tag, consider if a more specific tag like "Progreso" is better
router.include_router(activity.router) # Will use the tag "User Activity" defined in activity.py
router.include_router(statistics.router) # Will use the tag "Statistics" defined in statistics.py