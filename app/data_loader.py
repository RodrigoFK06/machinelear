import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from .config import DATA_PATH, TEST_SIZE, RANDOM_STATE

FRAMES = 35
FEATURES = 42

def load_dataset():
    # 1. Cargar y barajar el dataset
    try:
        df = pd.read_csv(DATA_PATH, header=None, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️ UTF-8 falló, intentando con ISO-8859-1...")
        df = pd.read_csv(DATA_PATH, header=None, encoding="ISO-8859-1")

    df = shuffle(df, random_state=RANDOM_STATE)

    # ✅ 2. Separar datos (X), etiqueta (y), ignorar nivel
    X = df.iloc[:, :-2].values.astype(np.float32)   # todos los landmarks (35x42 = 1470)
    y = df.iloc[:, -2].values                       # penúltima columna = etiqueta
    # nivel = df.iloc[:, -1].values                # (opcional) último campo, lo ignoramos

    # 3. Aplicar Z-score: (x - media) / desviación
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6  # evitar división por cero
    X = (X - mean) / std

    # 4. Redimensionar a [num_samples, 35, 42]
    X = X.reshape((-1, FRAMES, FEATURES))

    # 5. Codificar etiquetas
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 6. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    # 7. Diagnóstico
    print("📊 X media:", np.mean(X), "std:", np.std(X))
    print("🟢 y clases:", encoder.classes_)

    return X_train, X_test, y_train, y_test, encoder

