from fastapi import APIRouter, HTTPException, Query
from app.db.mongodb import collection
from app.models.schema import ProgressItem
from typing import List, Optional
import logging

# TODO: INDEXING - Consider an index on (nickname, expected_label, timestamp) or (expected_label, nickname, timestamp) to optimize the /progress aggregation, especially when filtered by nickname. Also, (timestamp) is used for last_attempt.
# TODO: TESTS - Add unit tests for the progress aggregation pipeline, covering different data scenarios (e.g., no data, data for one label, data for multiple labels, with/without nickname filter) and division by zero handling (mocking MongoDB).
router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/progress",
            response_model=List[ProgressItem],
            tags=["Registros"],
            summary="Get user progress statistics per label",
            description="Retrieves aggregated progress statistics for each medical sign label. Statistics include attempt counts, success rates, and confidence levels. Can be filtered by nickname to get user-specific progress."
            )
# TODO: AUTHENTICATION - The 'nickname' parameter should be removed or made secondary.
# User identity for progress tracking should primarily come from an authenticated session/token.
# If admin access to other users' progress is needed, that should be handled by specific roles/permissions.
async def get_progress(nickname: Optional[str] = Query(None, description="Filtrar por nickname del usuario para obtener su progreso específico.")):
    try:
        # Filtro base
        match_stage = {}
        if nickname:
            match_stage["nickname"] = nickname

        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}}, # Match documents (filter by nickname if provided)
            {
                "$group": {  # Group by expected_label
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
                "$project": {  # Reshape the output
                    "label": "$_id",
                    "total_attempts": 1,
                    "correct_attempts": 1,
                    "doubtful_attempts": 1,
                    "incorrect_attempts": 1,
                    # Division by zero: MongoDB's $divide operator returns null if the divisor is zero,
                    # or if both dividend and divisor are zero. If the dividend is non-zero and divisor is zero,
                    # it returns an error in older versions or +/- Infinity in newer versions (5.0+).
                    # The $multiply by 100 and $round operations will propagate null, resulting in null rates,
                    # which is acceptable as it indicates no attempts or an undefined rate.
                    "success_rate": {
                        "$cond": {
                            "if": {"$eq": ["$total_attempts", 0]},
                            "then": 0.0, # Explicitly return 0.0 if total_attempts is 0
                            "else": {
                                "$round": [
                                    {"$multiply": [{"$divide": ["$correct_attempts", "$total_attempts"]}, 100]}, 2
                                ]
                            }
                        }
                    },
                    "doubtful_rate": {
                        "$cond": {
                            "if": {"$eq": ["$total_attempts", 0]},
                            "then": 0.0,
                            "else": {
                                "$round": [
                                    {"$multiply": [{"$divide": ["$doubtful_attempts", "$total_attempts"]}, 100]}, 2
                                ]
                            }
                        }
                    },
                    "incorrect_rate": {
                         "$cond": {
                            "if": {"$eq": ["$total_attempts", 0]},
                            "then": 0.0,
                            "else": {
                                "$round": [
                                    {"$multiply": [{"$divide": ["$incorrect_attempts", "$total_attempts"]}, 100]}, 2
                                ]
                            }
                        }
                    },
                    "average_confidence": {"$round": ["$average_confidence", 2]}, # $avg returns null if no documents, $round will propagate null
                    "max_confidence": {"$round": ["$max_confidence", 2]}, # $max returns null if no documents, $round will propagate null
                    "min_confidence": {"$round": ["$min_confidence", 2]}, # $min returns null if no documents, $round will propagate null
                    "last_attempt": 1,
                    "_id": 0  # Exclude the default _id field
                }
            },
            {"$sort": {"label": 1}} # Sort results by label
        ]

        cursor = collection.aggregate(pipeline)
        result = []
        async for doc in cursor:
            # Handle potential nulls from aggregation if needed before Pydantic validation,
            # though Pydantic should handle Optional fields correctly.
            # For example, if average_confidence could be null and Pydantic expects float:
            # doc['average_confidence'] = doc.get('average_confidence') if doc.get('average_confidence') is not None else 0.0
            result.append(doc)
        
        if not result and nickname:
            logger.info("No progress data found for nickname: %s", nickname)
        elif not result:
            logger.info("No progress data found.")
            
        return result

    except Exception as e:
        logger.error("Error calculating progress for nickname '%s': %s", nickname, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al obtener el progreso: {str(e)}")
