import os
import tensorflow as tf
import joblib  # Usamos joblib para evitar errores de pickle

# Usamos la ruta absoluta al directorio actual (services/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas al root del proyecto (../models)
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(BASE_DIR, "..", "..", "models", "cnn_lstm_model.h5")
)
ENCODER_PATH = os.getenv(
    "ENCODER_PATH", os.path.join(BASE_DIR, "..", "..", "models", "label_encoder.pkl")
)

# Verificación de existencia
if not os.path.exists(MODEL_PATH):
    raise OSError(
        f"❌ Model file not found at {MODEL_PATH}. "
        "Check MODEL_PATH or place cnn_lstm_model.h5 in the models/ folder."
    )

if not os.path.exists(ENCODER_PATH):
    raise OSError(
        f"❌ Encoder file not found at {ENCODER_PATH}. "
        "Check ENCODER_PATH or place label_encoder.pkl in the models/ folder."
    )

# Cargar modelo y codificador
model = tf.keras.models.load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

__all__ = ["model", "encoder"]
