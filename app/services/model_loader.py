import os
import pickle
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Leer rutas desde variables de entorno, con valores por defecto para .h5 y .pkl
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(BASE_DIR, "models", "lstm_gestos_model.h5")
)
ENCODER_PATH = os.getenv(
    "ENCODER_PATH", os.path.join(BASE_DIR, "models", "label_encoder_lstm.pkl")
)

# Verificaciones de existencia
if not os.path.exists(MODEL_PATH):
    raise OSError(
        f"Model file not found at {MODEL_PATH}. Configure MODEL_PATH or place "
        "lstm_gestos_model.h5 in the models/ folder."
    )

if not os.path.exists(ENCODER_PATH):
    raise OSError(
        f"Encoder file not found at {ENCODER_PATH}. Configure ENCODER_PATH or "
        "place label_encoder_lstm.pkl in the models/ folder."
    )

# Cargar modelo y codificador
model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

__all__ = ["model", "encoder"]
