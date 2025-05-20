from fastapi import APIRouter, HTTPException, Query
from app.db.mongodb import collection
from app.models.schema import ProgressItem
from typing import List, Optional

router = APIRouter()

@router.get("/progress", response_model=List[ProgressItem], tags=["Registros"])
async def get_progress(nickname: Optional[str] = Query(None, description="Filtrar por nickname del usuario")):
    try:
        # Filtro base
        match_stage = {}
        if nickname:
            match_stage["nickname"] = nickname

        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {
                "$group": {
                    "_id": "$expected_label",
                    "total_attempts": {"$sum": 1},
                    "correct_attempts": {
                        "$sum": {"$cond": [{"$eq": ["$evaluation", "CORRECTO"]}, 1, 0]}
                    },
                    "doubtful_attempts": {
                        "$sum": {"$cond": [{"$eq": ["$evaluation", "DUDOSO"]}, 1, 0]}
                    },
                    "incorrect_attempts": {
                        "$sum": {"$cond": [{"$eq": ["$evaluation", "INCORRECTO"]}, 1, 0]}
                    },
                    "average_confidence": {"$avg": "$confidence"},
                    "max_confidence": {"$max": "$confidence"},
                    "min_confidence": {"$min": "$confidence"},
                    "last_attempt": {"$max": "$timestamp"}
                }
            },
            {
                "$project": {
                    "label": "$_id",
                    "total_attempts": 1,
                    "correct_attempts": 1,
                    "doubtful_attempts": 1,
                    "incorrect_attempts": 1,
                    "success_rate": {
                        "$round": [
                            {"$multiply": [{"$divide": ["$correct_attempts", "$total_attempts"]}, 100]}, 2
                        ]
                    },
                    "doubtful_rate": {
                        "$round": [
                            {"$multiply": [{"$divide": ["$doubtful_attempts", "$total_attempts"]}, 100]}, 2
                        ]
                    },
                    "incorrect_rate": {
                        "$round": [
                            {"$multiply": [{"$divide": ["$incorrect_attempts", "$total_attempts"]}, 100]}, 2
                        ]
                    },
                    "average_confidence": {"$round": ["$average_confidence", 2]},
                    "max_confidence": {"$round": ["$max_confidence", 2]},
                    "min_confidence": {"$round": ["$min_confidence", 2]},
                    "last_attempt": 1,
                    "_id": 0
                }
            },
            {"$sort": {"label": 1}}
        ]

        cursor = collection.aggregate(pipeline)
        result = []
        async for doc in cursor:
            result.append(doc)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener el progreso: {str(e)}")
