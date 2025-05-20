import os
import pickle
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "lstm_gestos_model.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "label_encoder_lstm.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

__all__ = ["model", "encoder"]
