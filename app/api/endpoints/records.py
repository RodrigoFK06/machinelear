# TODO: INDEXING - Consider compound indexes on (nickname, timestamp), (evaluation, timestamp) and (timestamp) for optimal query performance in /records.
# TODO: TESTS - Add unit tests for filter construction logic and pagination (mocking MongoDB).
from fastapi import APIRouter, HTTPException, Query, Response
from app.db.mongodb import collection
from datetime import datetime, timezone
from typing import Optional, List

router = APIRouter()

@router.get("/records", tags=["Registros"])
async def get_records(
    response: Response,
    # TODO: AUTHENTICATION - The 'nickname' filter parameter may need adjustment based on auth roles.
    # Standard users should perhaps only see their own records (implicitly filtered by auth user ID).
    # Admins might be able to use this filter, or a more specific user ID filter.
    nickname: Optional[str] = Query(None, description="Filter by user's nickname"),
    date_from: Optional[datetime] = Query(None, description="Filter records from this date (ISO format). Example: 2023-01-01T00:00:00Z"),
    date_to: Optional[datetime] = Query(None, description="Filter records up to this date (ISO format). Example: 2023-01-31T23:59:59Z"),
    evaluation: Optional[str] = Query(None, description="Filter by evaluation type: CORRECTO, DUDOSO, INCORRECTO", regex="^(CORRECTO|DUDOSO|INCORRECTO)$"),
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of records to return per page")
):
    """
    Retrieves a paginated and filterable list of prediction records.

    Allows filtering by nickname, date range, and evaluation type.
    The 'X-Total-Count' header in the response indicates the total number of records
    matching the filter criteria.
    """
    mongo_filter = {}

    if nickname:
        mongo_filter["nickname"] = nickname

    if date_from and date_to:
        # Ensure dates are timezone-aware (UTC) if not already
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        if date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)
        mongo_filter["timestamp"] = {"$gte": date_from, "$lte": date_to}
    elif date_from:
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        mongo_filter["timestamp"] = {"$gte": date_from}
    elif date_to:
        if date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)
        mongo_filter["timestamp"] = {"$lte": date_to}

    if evaluation:
        mongo_filter["evaluation"] = evaluation

    try:
        total_count = await collection.count_documents(mongo_filter)
        
        documentos = collection.find(mongo_filter).sort("timestamp", -1).skip(skip).limit(limit)
        
        registros = []
        async for doc in documentos:
            doc["_id"] = str(doc["_id"])
            # Ensure timestamp is converted to ISO format string
            if isinstance(doc.get("timestamp"), datetime):
                doc["timestamp"] = doc["timestamp"].isoformat()
            registros.append(doc)
        
        response.headers["X-Total-Count"] = str(total_count)
        return registros
    except Exception as e:
        # Log the exception details for debugging
        # import logging
        # logger = logging.getLogger(__name__)
        # logger.error(f"Error querying records: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al consultar los registros: {str(e)}")
