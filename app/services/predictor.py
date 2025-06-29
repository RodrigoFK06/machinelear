import os
import numpy as np
from datetime import datetime
from app.services.model_loader import model, encoder
from app.services.evaluator import evaluate_prediction
from app.services.feedback_analyzer import analizar_error
from app.models.schema import PredictRequest, PredictResponse
from app.db.mongodb import collection, stats_collection
from app.config import DATASET_PATH, MODELS_DIR

UMBRAL_CONFIANZA = 75.0
UMBRAL_RECHAZO = 20.0

# Umbrales específicos por clase
UMBRAL_POR_CLASE = {
    "dolor": 95.0,  # más exigente
    "yo": 70.0,
    "a_mi_me_duele_la_cabeza": 70.0,
    "tengo_fiebre_y_tos": 70.0
}

# Carga normalización
MEAN = np.load(str(MODELS_DIR / "mean.npy"))
STD = np.load(str(MODELS_DIR / "std.npy"))

def es_secuencia_invalida(seq: np.ndarray) -> bool:
    if np.count_nonzero(seq) < 0.5 * seq.size:
        return True
    if np.var(seq) < 1e-3:
        return True
    return False

async def predict_sequence(data: PredictRequest) -> PredictResponse:
    sequence = np.array(data.sequence, dtype=np.float32)

    if sequence.shape != (35, 42):
        raise ValueError("La secuencia debe tener forma (35, 42)")

    print("📦 Secuencia recibida (forma):", sequence.shape)
    print("📦 Primeros 3 valores del primer frame:", sequence[0][:3])
    print("📦 Varianza total:", np.var(sequence))

    if es_secuencia_invalida(sequence):
        return PredictResponse(
            predicted_label="ninguna",
            confidence=0.0,
            evaluation="NO_RECONOCIDA",
            observation="La secuencia enviada está vacía, tiene muchos ceros o es demasiado uniforme.",
            success_rate=None,
            average_confidence=None
        )

    prediction = model.predict(np.array([sequence]), verbose=0)
    print("📊 Vector de predicción completo:", prediction[0])
    for i, val in enumerate(prediction[0]):
        print(f"Clase {i} → {val}")

    class_index = int(np.argmax(prediction))
    raw_conf = float(prediction[0][class_index])
    if raw_conf > 1.0:
        raw_conf /= 100.0
    confidence = round(raw_conf * 100, 2)

    predicted_label = encoder.inverse_transform([class_index])[0]
    umbral_especifico = UMBRAL_POR_CLASE.get(predicted_label, UMBRAL_CONFIANZA)

    if confidence < max(UMBRAL_RECHAZO, umbral_especifico):
        return PredictResponse(
            predicted_label="ninguna",
            confidence=confidence,
            evaluation="NO_RECONOCIDA",
            observation=f"La predicción de '{predicted_label}' tiene confianza {confidence}%, por debajo del umbral ({umbral_especifico}%).",
            success_rate=None,
            average_confidence=None
        )

    print("🔍 Predicción cruda:", prediction[0])
    print("🔍 Confianza (raw):", raw_conf)
    print("🔍 Etiqueta predicha:", predicted_label)
    print("🔎 Primer frame recibido desde frontend:", sequence[0])

    evaluation, correct = evaluate_prediction(
        predicted_label, data.expected_label, confidence, UMBRAL_CONFIANZA
    )
    print("🏷️  Etiqueta predicha:", predicted_label)
    print("📈  Confianza:", confidence)
    print("🏷️  Etiqueta esperada:", data.expected_label)
    print("🎯  Evaluación:", evaluation)

    observation = None
    if evaluation == "INCORRECTO":
        dataset_path = str(DATASET_PATH)
        observation = analizar_error(sequence, dataset_path, data.expected_label)

    registro = {
        "nickname": data.nickname,
        "sequence_shape": sequence.shape,
        "predicted_label": predicted_label,
        "expected_label": data.expected_label,
        "confidence": confidence,
        "evaluation": evaluation,
        "observation": observation,
        "timestamp": datetime.utcnow()
    }
    await collection.insert_one(registro)

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

    success_rate = (correct_count / total) if total else None
    average_confidence = (confidence_sum / total) if total else None

    return PredictResponse(
        predicted_label=predicted_label,
        confidence=confidence,
        evaluation=evaluation,
        observation=observation,
        success_rate=round(success_rate * 100, 2) if success_rate else None,
        average_confidence=round(average_confidence, 2) if average_confidence else None
    )
