import numpy as np
from .model_utils import load_keras_model, load_encoder
from .config import CNN_LSTM_MODEL_PATH, ENCODER_PATH

model = load_keras_model(CNN_LSTM_MODEL_PATH)
encoder = load_encoder(ENCODER_PATH)


def predict(sequence):
    arr = np.array(sequence).reshape(1, 35, 42)
    probs = model.predict(arr)[0]
    idx = np.argmax(probs)
    label = encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx]) * 100
    return label, confidence, probs.tolist()
