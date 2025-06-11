import joblib
import os

# Ajusta la ruta si cambió tu estructura de carpetas
encoder_path = os.path.join("models", "label_encoder.pkl")

try:
    encoder = joblib.load(encoder_path)
    print("✅ Encoder cargado exitosamente.")
    print(f"Clases: {encoder.classes_}")
except Exception as e:
    print("❌ Error al cargar el encoder:")
    print(e)
