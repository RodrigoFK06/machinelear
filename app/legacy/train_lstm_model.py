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

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset_medico.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

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
encoder_path = os.path.join(MODELS_DIR, "label_encoder_lstm.pkl")
with open(encoder_path, "wb") as f:
    pickle.dump(encoder, f)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Crear modelo LSTM
model = models.Sequential([
    layers.Input(shape=(SEQUENCE_LENGTH, FEATURES)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Entrenar
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=1)

# Evaluar
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Precisión: {acc * 100:.2f}%")

# Guardar modelo
model_path = os.path.join(MODELS_DIR, "lstm_gestos_model.h5")
model.save(model_path)
print(f"✅ Modelo LSTM guardado en: {model_path}")
print(f"✅ Codificador guardado en: {encoder_path}")
print("🎯 Etiquetas:", list(encoder.classes_))
