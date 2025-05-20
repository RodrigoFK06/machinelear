from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router as api_router

app = FastAPI(
    title="Medical Sign Recognition API",
    description="Backend para la plataforma de aprendizaje de señas médicas con evaluación en tiempo real.",
    version="1.0.0"
)

# CORS (puedes ajustar los orígenes según el frontend real)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar por dominios específicos en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar las rutas del API
app.include_router(api_router)
