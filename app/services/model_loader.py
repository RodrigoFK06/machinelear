import os
import pickle
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Permitir configurar rutas a través de variables de entorno para facilitar
# despliegues donde los modelos se almacenan en ubicaciones personalizadas.
MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(BASE_DIR, "..", "lstm_gestos_model.h5")
)
ENCODER_PATH = os.getenv(
    "ENCODER_PATH", os.path.join(BASE_DIR, "..", "label_encoder_lstm.pkl")
)

if not os.path.exists(MODEL_PATH):
    raise OSError(
        f"Model file not found at {MODEL_PATH}. Configure MODEL_PATH or place "
        "lstm_gestos_model.h5 in the project root."
    )

if not os.path.exists(ENCODER_PATH):
    raise OSError(
        f"Encoder file not found at {ENCODER_PATH}. Configure ENCODER_PATH or "
        "place label_encoder_lstm.pkl in the project root."
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

__all__ = ["model", "encoder"]
