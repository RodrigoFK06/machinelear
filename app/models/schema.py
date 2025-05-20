from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import os
import pandas as pd

# Cargar etiquetas válidas desde el CSV
def load_labels():
    dataset_path = os.path.join("app", "legacy", "dataset_medico.csv")
    if not os.path.exists(dataset_path):
        return []
    df = pd.read_csv(dataset_path, header=None)
    label_col = df.columns[-1]
    return df[label_col].dropna().unique().tolist()

VALID_LABELS = load_labels()

class PredictRequest(BaseModel):
    sequence: List[List[float]] = Field(..., example=[[0.1] * 42] * 35)
    expected_label: str = Field(..., example="tengo_fiebre_y_tos")
    nickname: Optional[str] = Field(None, example="usuario123")

    @validator("sequence")
    def validate_sequence(cls, value):
        if len(value) < 30:
            raise ValueError("La secuencia debe tener al menos 30 frames.")
        for frame in value:
            if len(frame) != 42:
                raise ValueError("Cada frame debe tener exactamente 42 valores (keypoints).")
        return value

    @validator("expected_label")
    def validate_label(cls, value):
        if value not in VALID_LABELS:
            raise ValueError(f"La etiqueta '{value}' no es válida. Usa una de: {VALID_LABELS}")
        return value

class PredictResponse(BaseModel):
    predicted_label: str
    confidence: float
    evaluation: str  # "CORRECTO", "DUDOSO", "INCORRECTO"
    observation: Optional[str] = None
    success_rate: Optional[float] = None
    average_confidence: Optional[float] = None


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
