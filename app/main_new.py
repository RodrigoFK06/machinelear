from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.router import router as api_router
import logging
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical Sign Recognition API",
    description="Backend para la plataforma de aprendizaje de señas médicas con evaluación en tiempo real.",
    version="1.0.0"
)

# Middleware para manejo de errores
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(f"Error no manejado en {request.url}: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Error interno del servidor",
                "path": str(request.url.path),
                "method": request.method
            }
        )

# ✅ CORS: define orígenes seguros para producción
origins = [
    "https://machinefrontend.vercel.app",  # frontend en Vercel (producción)
    "http://localhost:3000",            # opcional: frontend en desarrollo local
    "https://machinefrontend-git-main-rodrigofk06s-projects.vercel.app",
    "https://machinefrontend.vercel.app/",  # frontend en Vercel (producción)
    "http://localhost:3000/",            # opcional: frontend en desarrollo local
    "https://machinefrontend-git-main-rodrigofk06s-projects.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # restringe solo a dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Evento de inicio para logging
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Medical Sign Recognition API iniciada exitosamente")
    logger.info("📚 Documentación disponible en /docs")
    
# Evento de cierre para logging  
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Medical Sign Recognition API cerrándose")

# Montar las rutas del API
app.include_router(api_router)

# ✅ Ruta raíz para evitar error 503 en deployment
@app.get("/", tags=["Root"])
async def read_root():
    """
    Ruta raíz del servicio.
    Confirma que la API está activa y funcionando.
    """
    logger.info("Acceso a ruta raíz - API funcionando correctamente")
    return {
        "message": "Medical Sign Recognition API - Servicio activo",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint.
    Returns a status of "ok" if the application is running.
    """
    logger.info("Health check realizado")
    return {"status": "ok"}
