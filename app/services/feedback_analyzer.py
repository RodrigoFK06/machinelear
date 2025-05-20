import pandas as pd
import numpy as np

FRAMES = 35
FEATURES = 42

def cargar_referencias(path_csv: str, nombre_sena: str) -> np.ndarray | None:
    df = pd.read_csv(path_csv, header=None)
    etiqueta_col = df.columns[-1]
    df = df[df[etiqueta_col] == nombre_sena]

    if df.empty:
        return None

    secuencias = df.iloc[:, :-1].values
    return secuencias.reshape((-1, FRAMES, FEATURES))

def dividir_en_tres(secuencia: np.ndarray) -> list:
    return [
        secuencia[0:12],
        secuencia[12:24],
        secuencia[24:35]
    ]

def promedio_por_segmento(secuencias: np.ndarray) -> list:
    partes = [[], [], []]  # inicio, mitad, final
    for seq in secuencias:
        inicio, mitad, final = dividir_en_tres(seq)
        partes[0].append(inicio)
        partes[1].append(mitad)
        partes[2].append(final)
    return [np.mean(p, axis=0) for p in partes]

def distancia_segmento(secuencia_usuario: np.ndarray, referencias: list) -> list:
    resultados = []
    partes_usuario = dividir_en_tres(np.array(secuencia_usuario))
    for i, ref in enumerate(referencias):
        dist = np.linalg.norm(partes_usuario[i] - ref)
        resultados.append(dist)
    return resultados

def analizar_error(secuencia_usuario: np.ndarray, dataset_csv: str, nombre_sena: str) -> str:
    ref_data = cargar_referencias(dataset_csv, nombre_sena)
    if ref_data is None:
        return "No hay datos de referencia para esa seña."

    refs = promedio_por_segmento(ref_data)
    dists = distancia_segmento(secuencia_usuario, refs)
    secciones = ["inicial", "media", "final"]
    indice = np.argmax(dists)
    return f"Observación: El error ocurrió principalmente en la parte {secciones[indice]}."


# app/services/predictor.py
import numpy as np
from app.services.model_loader import model, encoder
from app.services.feedback_analyzer import analizar_error
from app.models.schema import PredictRequest, PredictResponse

UMBRAL_CONFIANZA = 75.0
estadisticas_globales = {}

def predict_sequence(data: PredictRequest) -> PredictResponse:
    sequence = np.array(data.sequence)

    if sequence.shape != (35, 42):
        raise ValueError("La secuencia debe tener forma (35, 42)")

    prediction = model.predict(np.array([sequence]), verbose=0)
    class_index = np.argmax(prediction)
    confidence = float(prediction[0][class_index]) * 100
    predicted_label = encoder.inverse_transform([class_index])[0]

    if predicted_label.lower() == data.expected_label.lower():
        evaluation = "CORRECTO" if confidence >= UMBRAL_CONFIANZA else "DUDOSO"
    else:
        evaluation = "INCORRECTO"

    est = estadisticas_globales.setdefault(data.expected_label, {
        "intentos": 0, "aciertos": 0, "confianza_total": 0
    })
    est["intentos"] += 1
    if evaluation == "CORRECTO":
        est["aciertos"] += 1
    est["confianza_total"] += confidence

    success_rate = (est["aciertos"] / est["intentos"] * 100) if est["intentos"] else None
    average_confidence = (est["confianza_total"] / est["intentos"]) if est["intentos"] else None

    observation = None
    if evaluation == "INCORRECTO":
        dataset_path = "app/legacy/dataset_medico.csv"
        observation = analizar_error(sequence, dataset_path, data.expected_label)

    return PredictResponse(
        predicted_label=predicted_label,
        confidence=confidence,
        evaluation=evaluation,
        observation=observation,
        success_rate=success_rate,
        average_confidence=average_confidence
    )