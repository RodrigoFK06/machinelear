# Medical Sign Recognition API

This project provides a FastAPI backend that predicts medical sign language from hand keypoint sequences using a TensorFlow model. Predictions and statistics are stored in MongoDB.

## Setup

1. Create a `.env` file based on `.env.example` with your MongoDB connection string.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## Tests

Run the unit tests with:
```bash
pytest -q
```

## API Usage

This section shows how to interact with the main endpoints using `curl`. No authentication is required in the current version. Replace `localhost:8000` with your host if running on a different machine.

### POST /predict
Predict a sign sequence.

**URL**: `/predict`

**Method**: `POST`

**Headers**:
- `Content-Type: application/json`

**Body example**:
```json
{
  "sequence": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2], ... 35 frames],
  "expected_label": "dolor_de_cabeza",
  "nickname": "usuario123"
}
```

**Success response**:
```json
{
  "predicted_label": "dolor_de_cabeza",
  "confidence": 89.3,
  "evaluation": "CORRECTO",
  "observation": null,
  "success_rate": 74.2,
  "average_confidence": 82.9
}
```

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/predict_example.json
```

Possible errors: `400` invalid input, `500` server error.

### GET /labels
Retrieve all trained labels.

**URL**: `/labels`

**Method**: `GET`

Example request:
```bash
curl http://localhost:8000/labels
```

Possible errors: `404` if the dataset is missing, `500` on read error.

### GET /records
List prediction records with optional filters.

**URL**: `/records`

**Method**: `GET`

**Query parameters**:
- `nickname` – filter by user nickname
- `date_from` – ISO start date (e.g., `2025-01-01T00:00:00Z`)
- `date_to` – ISO end date
- `evaluation` – `CORRECTO`, `DUDOSO` or `INCORRECTO`
- `skip` – records to skip (pagination)
- `limit` – max records to return (1‑100)

**Sample response**:
```json
[
  {
    "_id": "60d5ec49f0b2f3a1c4d4a9c1",
    "nickname": "usuario123",
    "sequence_shape": [35, 42],
    "predicted_label": "dolor_de_cabeza",
    "expected_label": "dolor_de_cabeza",
    "confidence": 89.3,
    "evaluation": "CORRECTO",
    "observation": null,
    "timestamp": "2025-05-20T22:32:10.123Z"
  }
]
```

Example request:
```bash
curl "http://localhost:8000/records?nickname=usuario123&limit=5"
```

Possible errors: `500` database query errors.

### GET /progress
Aggregated progress per label.

**URL**: `/progress`

**Method**: `GET`

**Query parameters**:
- `nickname` – optional user nickname

**Sample response**:
```json
[
  {
    "label": "dolor_de_cabeza",
    "total_attempts": 5,
    "correct_attempts": 4,
    "doubtful_attempts": 0,
    "incorrect_attempts": 1,
    "success_rate": 80.0,
    "doubtful_rate": 0.0,
    "incorrect_rate": 20.0,
    "average_confidence": 85.4,
    "max_confidence": 90.0,
    "min_confidence": 70.0,
    "last_attempt": "2025-05-20T22:32:10.123Z"
  }
]
```

Example request:
```bash
curl "http://localhost:8000/progress?nickname=usuario123"
```

Possible errors: `500` server error.

### GET /activity/daily/{nickname}/{date}
Daily activity for a user. `date` must use `YYYY-MM-DD` format.

**URL**: `/activity/daily/{nickname}/{date}`

**Method**: `GET`

**Sample response**:
```json
{
  "nickname": "usuario123",
  "date": "2025-05-20",
  "summary": {
    "total_practices": 3,
    "correct_practices": 2,
    "doubtful_practices": 1,
    "incorrect_practices": 0
  },
  "records": [
    {
      "id": "60d5ec49f0b2f3a1c4d4a9c1",
      "timestamp": "2025-05-20T10:00:00Z",
      "predicted_label": "dolor_de_cabeza",
      "expected_label": "dolor_de_cabeza",
      "confidence": 90.5,
      "evaluation": "CORRECTO"
    }
  ]
}
```

Example request:
```bash
curl http://localhost:8000/activity/daily/usuario123/2025-05-20
```

Possible errors: `400` for invalid date format, `500` on database error.

### GET /stats/global_distribution
Get overall result distribution.

**URL**: `/stats/global_distribution`

**Method**: `GET`

**Sample response**:
```json
{
  "total_evaluations": 2000,
  "distribution": [
    {"evaluation_type": "CORRECTO", "count": 1500, "percentage": 75.0},
    {"evaluation_type": "DUDOSO", "count": 300, "percentage": 15.0},
    {"evaluation_type": "INCORRECTO", "count": 200, "percentage": 10.0}
  ]
}
```

Example request:
```bash
curl http://localhost:8000/stats/global_distribution
```

Possible errors: `500` server error.

For quick testing you can import the endpoints into tools like Postman or Insomnia by creating requests with the examples above.
