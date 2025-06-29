# TODO: TESTS - Add unit tests for Pydantic model validators, especially for PredictRequest sequence and label validation.
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import os
import pandas as pd

# Cargar etiquetas válidas desde el CSV
def load_labels():
    dataset_path = "D:/machinelear/data/dataset_medico.csv"
    if not os.path.exists(dataset_path):
        return []
    df = pd.read_csv(dataset_path, header=None)
    label_col = df.columns[-2]
    return df[label_col].dropna().unique().tolist()

VALID_LABELS = load_labels()

class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(..., example=[[0.1] * 42] * 35, description="Sequence of keypoints, typically 35 frames with 42 keypoints each.")
    expected_label: str = Field(..., example="tengo_fiebre_y_tos", description="The expected medical sign label for this sequence.")
    nickname: Optional[str] = Field(None, example="usuario123", description="Optional user's nickname for tracking purposes.")

    @validator("sequence")
    def validate_sequence(cls, value):
        if len(value) < 30:
            raise ValueError(f"La secuencia debe tener al menos 30 frames, pero se recibieron {len(value)}.")
        for frame in value:
            if len(frame) != 42:
                raise ValueError(f"Cada frame en la secuencia debe tener exactamente 42 valores (keypoints), pero se encontró un frame con {len(frame)} valores.")
            for i, val in enumerate(frame):
                try:
                    frame[i] = float(val)
                except Exception:
                    raise ValueError(f"Valor no convertible a float: {val}")

        return value

    @validator("expected_label")
    def validate_label(cls, value):
        print("🔄 Validando etiqueta:", value)
        value = value.strip().lower()
        valid_labels = [label.strip().lower() for label in load_labels()]
        print("📋 Etiquetas válidas cargadas:", valid_labels)
        if value not in valid_labels:
            raise ValueError(f"La etiqueta '{value}' no es válida. Por favor, use una etiqueta conocida.")
        return value


class PredictResponse(BaseModel):
    predicted_label: str = Field(..., description="La etiqueta predicha por el modelo.", example="dolor_de_cabeza")
    confidence: float = Field(..., description="La confianza de la predicción, en porcentaje (0-100).", example=95.5)
    evaluation: str = Field(..., description="Evaluación de la predicción (CORRECTO, DUDOSO, INCORRECTO).", example="CORRECTO") # "CORRECTO", "DUDOSO", "INCORRECTO"
    observation: Optional[str] = Field(None, description="Observación adicional, especialmente si la evaluación es INCORRECTO (puede incluir sugerencias).", example="Intenta separar más los movimientos.")
    success_rate: Optional[float] = Field(None, description="Tasa de éxito histórica para la etiqueta esperada (y usuario, si se proporcionó), en porcentaje.", example=75.0)
    average_confidence: Optional[float] = Field(None, description="Confianza promedio histórica para la etiqueta esperada (y usuario, si se proporcionó), en porcentaje.", example=82.3)


class ProgressItem(BaseModel):
    label: str = Field(..., example="tengo_fiebre_y_tos", description="Etiqueta de la seña evaluada")
    total_attempts: int = Field(..., example=10, description="Número total de intentos")
    correct_attempts: int = Field(..., example=7, description="Número de aciertos (evaluación == 'CORRECTO')")
    doubtful_attempts: int = Field(..., example=2, description="Número de intentos evaluados como 'DUDOSO'")
    incorrect_attempts: int = Field(..., example=1, description="Número de errores (evaluación == 'INCORRECTO')")

    success_rate: float = Field(..., example=70.0, description="Porcentaje de aciertos")
    doubtful_rate: float = Field(..., example=20.0, description="Porcentaje de evaluaciones dudosas")
    incorrect_rate: float = Field(..., example=10.0, description="Porcentaje de errores")

    average_confidence: float = Field(..., example=83.25, description="Confianza promedio")
    max_confidence: float = Field(..., example=92.5, description="Confianza máxima")
    min_confidence: float = Field(..., example=60.0, description="Confianza mínima")

    last_attempt: Optional[datetime] = Field(None, example="2025-05-20T22:32:10.123Z",
                                             description="Fecha del último intento")


class DailyActivityRecord(BaseModel):
    id: str = Field(..., alias="_id", description="El ID del registro de MongoDB", example="60d5ec49f0b2f3a1c4d4a9c1")
    timestamp: datetime = Field(..., description="Fecha y hora completa del registro de la práctica", example="2023-10-26T10:30:00.123Z")
    predicted_label: str = Field(..., description="Etiqueta predicha por el modelo", example="dolor_de_cabeza")
    expected_label: str = Field(..., description="Etiqueta esperada por el usuario", example="dolor_de_cabeza")
    confidence: float = Field(..., description="Confianza de la predicción (0-100)", example=92.75)
    evaluation: str = Field(..., description="Evaluación de la práctica (CORRECTO, DUDOSO, INCORRECTO)", example="CORRECTO")


class DailyActivitySummary(BaseModel):
    total_practices: int = Field(..., description="Número total de prácticas realizadas en el día", example=25)
    correct_practices: int = Field(..., description="Número de prácticas evaluadas como 'CORRECTO'", example=18)
    doubtful_practices: int = Field(..., description="Número de prácticas evaluadas como 'DUDOSO'", example=5)
    incorrect_practices: int = Field(..., description="Número de prácticas evaluadas como 'INCORRECTO'", example=2)


class DailyActivityResponse(BaseModel):
    nickname: str = Field(..., description="Nickname del usuario", example="usuario_activo_123")
    date: str = Field(..., description="Fecha de la actividad solicitada, en formato YYYY-MM-DD", example="2023-10-26")
    summary: DailyActivitySummary = Field(..., description="Resumen de la actividad del día")
    records: List[DailyActivityRecord] = Field(..., description="Lista de registros de actividad para el día")


class GlobalResultDistributionItem(BaseModel):
    evaluation_type: str = Field(..., description="Type of evaluation (e.g., CORRECTO, DUDOSO, INCORRECTO)", example="CORRECTO")
    count: int = Field(..., description="Total count for this evaluation type", example=1500)
    percentage: float = Field(..., description="Percentage of this evaluation type out of the total evaluations", example=75.0)


class GlobalResultsDistributionResponse(BaseModel):
    total_evaluations: int = Field(..., description="Total number of evaluations processed in the system", example=2000)
    distribution: List[GlobalResultDistributionItem] = Field(..., description="List of counts and percentages per evaluation type")
