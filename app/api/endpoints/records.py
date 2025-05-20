from fastapi import APIRouter, HTTPException
from app.db.mongodb import collection
from datetime import datetime

router = APIRouter()

@router.get("/records", tags=["Registros"])
async def get_records():
    try:
        documentos = collection.find().sort("timestamp", -1)
        registros = []
        async for doc in documentos:
            doc["_id"] = str(doc["_id"])
            doc["timestamp"] = doc["timestamp"].isoformat() if isinstance(doc["timestamp"], datetime) else doc["timestamp"]
            registros.append(doc)
        return registros
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al consultar los registros: {str(e)}")
