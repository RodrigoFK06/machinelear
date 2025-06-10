import numpy as np
from datetime import datetime
from app.services.model_loader import model, encoder
from app.services.evaluator import evaluate_prediction
from app.services.feedback_analyzer import analizar_error
from app.models.schema import PredictRequest, PredictResponse
from app.db.mongodb import collection, stats_collection
from typing import Optional

# Umbral mínimo para considerar una predicción como confiable
UMBRAL_CONFIANZA = 75.0

# TODO: TESTS - Add unit tests for predict_sequence, mocking the ML model and database interactions, to verify evaluation logic and statistics calculation.
async def predict_sequence(data: PredictRequest) -> PredictResponse:
    sequence = np.array(data.sequence)

    if sequence.shape != (35, 42):
        raise ValueError("La secuencia debe tener forma (35, 42)")

    prediction = model.predict(np.array([sequence]), verbose=0)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index]) * 100
    predicted_label = encoder.inverse_transform([class_index])[0]

    # Evaluación según confianza y etiqueta esperada
    evaluation, correct = evaluate_prediction(
        predicted_label, data.expected_label, confidence, UMBRAL_CONFIANZA
    )

    # Feedback adicional si es incorrecto
    observation = None
    if evaluation == "INCORRECTO":
        dataset_path = "app/legacy/dataset_medico.csv"
        observation = analizar_error(sequence, dataset_path, data.expected_label)

    # Guardar predicción en MongoDB (incluye nickname si lo hay)
    # TODO: AUTHENTICATION - Replace data.nickname with user ID from a proper authentication system.
    # The user's identity should be determined from an auth token rather than a nickname in the request body.
    # TODO: CLARIFY - Confirm if full sequence data (35x42 floats) needs to be stored in MongoDB, or if 'sequence_shape' is sufficient.
    registro = {
        "nickname": data.nickname,  # 👈 Se agrega aquí
        "sequence_shape": sequence.shape,
        "predicted_label": predicted_label,
        "expected_label": data.expected_label,
        "confidence": confidence,
        "evaluation": evaluation,
        "observation": observation,
        "timestamp": datetime.utcnow()
    }
    await collection.insert_one(registro)

    # Mantener estadísticas acumuladas en una colección separada para evitar
    # recorrer todos los documentos en cada predicción.
    stats_filter = {"expected_label": data.expected_label, "nickname": data.nickname}
    update = {
        "$inc": {
            "total": 1,
            "correct": 1 if correct else 0,
            "confidence_sum": confidence,
        }
    }
    await stats_collection.update_one(stats_filter, update, upsert=True)
    stats_doc = await stats_collection.find_one(stats_filter)
    total = stats_doc.get("total", 0)
    correct_count = stats_doc.get("correct", 0)
    confidence_sum = stats_doc.get("confidence_sum", 0.0)

    success_rate = (correct_count / total) * 100 if total else None
    average_confidence = (confidence_sum / total) if total else None

    return PredictResponse(
        predicted_label=predicted_label,
        confidence=confidence,
        evaluation=evaluation,
        observation=observation,
        success_rate=success_rate,
        average_confidence=average_confidence
    )
