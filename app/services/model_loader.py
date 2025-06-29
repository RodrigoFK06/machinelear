import os
import tensorflow as tf
import joblib  # Usamos joblib para evitar errores de pickle
from app.config import CNN_LSTM_MODEL_PATH, ENCODER_PATH

# Rutas desde configuración centralizada
MODEL_PATH = os.getenv("MODEL_PATH", str(CNN_LSTM_MODEL_PATH))
ENCODER_PATH_STR = os.getenv("ENCODER_PATH", str(ENCODER_PATH))

# Verificación de existencia
if not os.path.exists(MODEL_PATH):
    raise OSError(
        f"❌ Model file not found at {MODEL_PATH}. "
        "Check MODEL_PATH or place cnn_lstm_model.h5 in the models/ folder."
    )

if not os.path.exists(ENCODER_PATH_STR):
    raise OSError(
        f"❌ Encoder file not found at {ENCODER_PATH_STR}. "
        "Check ENCODER_PATH or place label_encoder.pkl in the models/ folder."
    )

# Cargar modelo y codificador
model = tf.keras.models.load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH_STR)

__all__ = ["model", "encoder"]
