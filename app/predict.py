import numpy as np
from .model_utils import load_keras_model, load_encoder
from .config import CNN_LSTM_MODEL_PATH, ENCODER_PATH

# Variables globales para lazy loading
_model = None
_encoder = None

def _get_model():
    """Carga el modelo de forma diferida."""
    global _model
    if _model is None:
        _model = load_keras_model(CNN_LSTM_MODEL_PATH)
    return _model

def _get_encoder():
    """Carga el encoder de forma diferida."""
    global _encoder
    if _encoder is None:
        _encoder = load_encoder(ENCODER_PATH)
    return _encoder

def predict(sequence):
    # Cargar modelo y encoder de forma lazy
    model = _get_model()
    encoder = _get_encoder()
    
    arr = np.array(sequence).reshape(1, 35, 42)
    probs = model.predict(arr)[0]
    idx = np.argmax(probs)
    label = encoder.inverse_transform([idx])[0]
    confidence = float(probs[idx]) * 100
    return label, confidence, probs.tolist()
