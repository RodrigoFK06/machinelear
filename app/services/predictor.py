import numpy as np
from datetime import datetime
from app.services.model_loader import model, encoder
from app.services.feedback_analyzer import analizar_error
from app.models.schema import PredictRequest, PredictResponse
from app.db.mongodb import collection  # ✅ MongoDB async
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
    if predicted_label.lower() == data.expected_label.lower():
        evaluation = "CORRECTO" if confidence >= UMBRAL_CONFIANZA else "DUDOSO"
    else:
        evaluation = "INCORRECTO"

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

    # TODO: Consider optimizing statistics calculation. Currently, it queries all matching records on every prediction,
    # which could become a performance bottleneck. Strategies could include using aggregation pipelines for direct
    # calculation in the DB, or maintaining/updating statistics in a separate summary collection.
    # Cálculo de estadísticas globales reales desde MongoDB
    filtro = {"expected_label": data.expected_label}
    # TODO: AUTHENTICATION - Replace data.nickname with user ID from a proper authentication system.
    # The user's identity should be determined from an auth token rather than a nickname in the request body.
    if data.nickname:
        filtro["nickname"] = data.nickname  # 👈 Se filtra por usuario si lo proporciona

    cursor = collection.find(filtro)
    intentos, aciertos, suma_confianza = 0, 0, 0.0
    async for doc in cursor:
        intentos += 1
        suma_confianza += doc.get("confidence", 0.0)
        if doc.get("evaluation") == "CORRECTO":
            aciertos += 1

    success_rate = (aciertos / intentos) * 100 if intentos else None
    average_confidence = (suma_confianza / intentos) if intentos else None

    return PredictResponse(
        predicted_label=predicted_label,
        confidence=confidence,
        evaluation=evaluation,
        observation=observation,
        success_rate=success_rate,
        average_confidence=average_confidence
    )
