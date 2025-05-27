from fastapi import APIRouter, HTTPException, Path
from app.db.mongodb import collection
from app.models.schema import DailyActivityRecord, DailyActivitySummary, DailyActivityResponse
from typing import List
from datetime import datetime, date, time, timezone
import logging

# TODO: INDEXING - Consider an index on (nickname, timestamp) for optimal query performance in /activity/daily.
# TODO: TESTS - Add unit tests for date parsing, date range generation (including timezone handling), and aggregation of daily activity stats (mocking MongoDB).
logger = logging.getLogger(__name__)
router = APIRouter(tags=["User Activity"])

@router.get("/activity/daily/{nickname}/{date_str}",
            response_model=DailyActivityResponse,
            summary="Get daily activity for a user",
            description="Retrieves a summary and detailed records of a user's practice activity for a specific day."
            )
# TODO: AUTHENTICATION - The 'nickname' path parameter should be re-evaluated.
# Typically, a user would fetch their own activity via an authenticated route (e.g., /users/me/activity/{date_str}).
# Accessing other users' activity would require admin privileges and a different path structure.
async def get_daily_activity(
    nickname: str = Path(..., description="User's nickname", example="usuario123"),
    date_str: str = Path(..., description="Date in YYYY-MM-DD format", example="2023-10-28", regex="^\d{4}-\d{2}-\d{2}$")
):
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.warning("Invalid date format received: %s", date_str)
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    # Define date range for the query (from start of day to end of day in UTC)
    # MongoDB stores timestamps as UTC (e.g., from datetime.utcnow())
    start_datetime = datetime.combine(parsed_date, time.min).replace(tzinfo=timezone.utc)
    end_datetime = datetime.combine(parsed_date, time.max).replace(tzinfo=timezone.utc)

    mongo_filter = {
        "nickname": nickname,
        "timestamp": {
            "$gte": start_datetime,
            "$lte": end_datetime  # Use $lte to include records at the very end of the day
        }
    }

    activity_records: List[DailyActivityRecord] = []
    total_practices = 0
    correct_practices = 0
    doubtful_practices = 0
    incorrect_practices = 0

    try:
        cursor = collection.find(mongo_filter).sort("timestamp", 1) # Sort by time ascending

        async for doc in cursor:
            total_practices += 1
            evaluation = doc.get("evaluation")
            if evaluation == "CORRECTO":
                correct_practices += 1
            elif evaluation == "DUDOSO":
                doubtful_practices += 1
            elif evaluation == "INCORRECTO":
                incorrect_practices += 1
            
            # Ensure timestamp is a datetime object. MongoDB stores BSON dates, which the driver converts to datetime.
            record_timestamp = doc.get("timestamp")
            if not isinstance(record_timestamp, datetime):
                # This case should ideally not happen if data is inserted correctly
                logger.warning("Record with _id %s has invalid timestamp format.", str(doc.get("_id")))
                # Default to a placeholder or skip, here we'll use epoch for now if missing or wrong type
                record_timestamp = datetime.fromtimestamp(0, tz=timezone.utc)


            activity_records.append(
                DailyActivityRecord(
                    _id=str(doc.get("_id")),
                    timestamp=record_timestamp,
                    predicted_label=doc.get("predicted_label", "N/A"),
                    expected_label=doc.get("expected_label", "N/A"),
                    confidence=doc.get("confidence", 0.0),
                    evaluation=evaluation if evaluation else "N/A"
                )
            )

        if total_practices == 0:
            logger.info("No activity found for user '%s' on date '%s'", nickname, date_str)
            # Return 200 with empty records and zeroed summary as per preference

        summary = DailyActivitySummary(
            total_practices=total_practices,
            correct_practices=correct_practices,
            doubtful_practices=doubtful_practices,
            incorrect_practices=incorrect_practices
        )

        return DailyActivityResponse(
            nickname=nickname,
            date=date_str,
            summary=summary,
            records=activity_records
        )

    except Exception as e:
        logger.error("Error fetching daily activity for user '%s' on date '%s': %s", nickname, date_str, e, exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while fetching daily activity.")
