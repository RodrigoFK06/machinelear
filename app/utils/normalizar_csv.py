import pandas as pd
import os

# Rutas de entrada y salida
INPUT_CSV = "D:/machinelear/data/dataset_medico.csv"
OUTPUT_CSV = os.path.join("data", "dataset_medico_normalizado.csv")

# Resolución estándar usada por MediaPipe o cámara
WIDTH = 640
HEIGHT = 480

# Cargar el dataset original
df = pd.read_csv(INPUT_CSV, header=None)

# Separar las columnas de coordenadas y etiqueta final
coords = df.iloc[:, :-1]  # Todas menos la última
labels = df.iloc[:, -1]   # Última columna

# Verificar que el número de columnas sea múltiplo de 2
assert coords.shape[1] % 2 == 0, "Las columnas de coordenadas deben estar en pares x/y"

# Normalizar columnas alternadas
normalized_coords = coords.copy()

for i in range(coords.shape[1]):
    if i % 2 == 0:  # columna x
        normalized_coords[i] = coords[i] / WIDTH
    else:  # columna y
        normalized_coords[i] = coords[i] / HEIGHT

# Combinar coordenadas normalizadas con etiquetas
normalized_df = pd.concat([normalized_coords, labels], axis=1)

# Guardar a nuevo CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
normalized_df.to_csv(OUTPUT_CSV, index=False, header=False)

print(f"✅ CSV normalizado guardado en: {OUTPUT_CSV}")
