import os
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from app.config import DATASET_PATH, MODELS_DIR

# Configuración
CSV_PATH = str(DATASET_PATH)
MODELS_DIR_STR = str(MODELS_DIR)
os.makedirs(MODELS_DIR_STR, exist_ok=True)

# Cargar el dataset
df = pd.read_csv(CSV_PATH, header=None)

# Asumimos que la última columna es la etiqueta
X = df.iloc[:, :-1].values  # Todos los datos numéricos
y = df.iloc[:, -1].values   # Etiquetas

# Verificar que se pueden redimensionar
SEQUENCE_LENGTH = 35
FEATURES = 42
if X.shape[1] != SEQUENCE_LENGTH * FEATURES:
    print(f"❌ El dataset tiene {X.shape[1]} columnas, pero se esperaban {SEQUENCE_LENGTH * FEATURES}.")
    exit()

# Redimensionar a [n_samples, 35, 42]
X = X.reshape((-1, SEQUENCE_LENGTH, FEATURES))

# Codificar etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Guardar el encoder
encoder_path = os.path.join(MODELS_DIR_STR, "label_encoder_lstm.pkl")
with open(encoder_path, "wb") as f:
    pickle.dump(encoder, f)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Definición del modelo sin InputLayer explícito
model = models.Sequential([
    # La primera capa LSTM incluye input_shape directamente
    layers.LSTM(64, return_sequences=False, input_shape=(SEQUENCE_LENGTH, FEATURES)),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),  # Opcional: para regularización
    layers.Dense(len(encoder.classes_), activation='softmax')
])

# Compilación
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Guardado compatible
model_path = os.path.join(MODELS_DIR_STR, "lstm_gestos_model.h5")
model.save(model_path, save_format="h5")

# Alternativa: SavedModel format (más robusto)
# model_dir = os.path.join(MODELS_DIR, "lstm_gestos_model")
# model.save(model_dir, save_format="tf")

print(f"Modelo guardado en: {model_path}")

# Verificación de carga (para testing)
try:
    loaded_model = tf.keras.models.load_model(model_path)
    print("✅ Modelo cargado exitosamente")
    print(f"Input shape: {loaded_model.input_shape}")
except Exception as e:
    print(f"❌ Error al cargar: {e}")