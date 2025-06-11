import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "lstm_gestos_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

sample_data = np.random.rand(1, 35, 42)
prediction = model.predict(sample_data)

print("✅ Modelo cargado y predicción generada exitosamente.")
print("🔮 Output:", prediction)
