import os
import joblib  # Usamos joblib para evitar errores de pickle
from app.config import CNN_LSTM_MODEL_PATH, ENCODER_PATH

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # Mock TensorFlow for development/testing when TensorFlow is not available
    print("⚠️ TensorFlow no disponible. Modo de desarrollo activo.")
    TENSORFLOW_AVAILABLE = False
    
    class MockModel:
        def predict(self, data):
            # Return mock prediction for development
            import numpy as np
            return np.array([[0.8, 0.1, 0.1]])  # Mock confidence scores
    
    class MockEncoder:
        def __init__(self):
            self.classes_ = ['dolor_de_cabeza', 'mareo', 'fatiga']  # Mock classes
        
        def inverse_transform(self, encoded_labels):
            # Mock label decoding
            if hasattr(encoded_labels, '__iter__'):
                return [self.classes_[int(label) % len(self.classes_)] for label in encoded_labels]
            else:
                return self.classes_[int(encoded_labels) % len(self.classes_)]
    
    class MockTensorFlow:
        class keras:
            class models:
                @staticmethod
                def load_model(path):
                    print(f"🔄 Mock: Cargando modelo desde {path}")
                    return MockModel()
    
    tf = MockTensorFlow()

# Rutas desde configuración centralizada
MODEL_PATH = os.getenv("MODEL_PATH", str(CNN_LSTM_MODEL_PATH))
ENCODER_PATH_STR = os.getenv("ENCODER_PATH", str(ENCODER_PATH))

# Variables globales para lazy loading
_model = None
_encoder = None

def _validate_paths():
    """Valida que los archivos de modelo y encoder existan."""
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

def get_model():
    """Carga el modelo de forma diferida (lazy loading)."""
    global _model
    if _model is None:
        if not TENSORFLOW_AVAILABLE:
            print("🔄 Mock: Creando modelo simulado para desarrollo")
            _model = MockModel()
            return _model
            
        _validate_paths()
        print("🔄 Cargando modelo CNN-LSTM...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo cargado exitosamente")
    return _model

def get_encoder():
    """Carga el encoder de forma diferida (lazy loading)."""
    global _encoder
    if _encoder is None:
        if not TENSORFLOW_AVAILABLE:
            print("🔄 Mock: Creando encoder simulado para desarrollo")
            _encoder = MockEncoder()
            return _encoder
            
        _validate_paths()
        print("🔄 Cargando encoder...")
        _encoder = joblib.load(ENCODER_PATH_STR)
        print("✅ Encoder cargado exitosamente")
    return _encoder

# Solo validar rutas al importar, NO cargar los modelos (solo si TensorFlow está disponible)
if TENSORFLOW_AVAILABLE:
    try:
        _validate_paths()
        print("✅ Rutas de modelo y encoder validadas")
    except OSError as e:
        print(f"⚠️ Advertencia: {e}")
        print("Los modelos se intentarán cargar cuando sean necesarios")
else:
    print("ℹ️ Modo desarrollo: Validación de rutas omitida (TensorFlow no disponible)")

# Para mantener compatibilidad con código existente que usa `from model_loader import model, encoder`
# Definimos funciones que actúan como las variables
def model():
    return get_model()

def encoder():
    return get_encoder()

__all__ = ["get_model", "get_encoder", "model", "encoder"]
