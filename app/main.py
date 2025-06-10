from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router as api_router

app = FastAPI(
    title="Medical Sign Recognition API",
    description="Backend para la plataforma de aprendizaje de señas médicas con evaluación en tiempo real.",
    version="1.0.0"
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

# Montar las rutas del API
app.include_router(api_router)

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Simple health check endpoint.
    Returns a status of "ok" if the application is running.
    """
    return {"status": "ok"}
