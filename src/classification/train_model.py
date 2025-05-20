import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

# Obtener ruta absoluta al archivo dataset_medico.csv
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(BASE_DIR, "dataset_medico.csv")  # ¡Nuevo nombre de archivo!

# Cargar el dataset
data = pd.read_csv(CSV_PATH)

# Separar características (X) y etiquetas (y)
X = data.drop("label", axis=1).values
y = data["label"].values

# Codificar etiquetas (ej: fiebre, dolor_pecho, etc.)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Guardar el encoder
with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(encoder, f)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Crear modelo secuencial
model = models.Sequential([
    layers.Input(shape=(42,)),  # 21 puntos x 2 (x, y)
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=30, validation_split=0.2)

# Evaluar
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Precisión en datos de prueba: {accuracy * 100:.2f}%")

# Guardar el modelo
model_path = os.path.join(BASE_DIR, "sign_language_model.h5")
model.save(model_path)
print(f"✅ Modelo guardado en: {model_path}")

# Mostrar etiquetas reconocidas
print("Etiquetas reconocidas por el modelo:", list(encoder.classes_))
