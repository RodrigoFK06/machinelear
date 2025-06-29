import pandas as pd
import numpy as np

csv_path = "D:/machinelear/data/dataset_medico.csv"
etiqueta_objetivo = "me_duele_la_cabeza"

# Cargar sin encabezado
df = pd.read_csv(csv_path, header=None)

# Verifica que cada fila tenga 1471 columnas (1470 datos + 1 etiqueta)
if df.shape[1] != 1471:
    raise ValueError(f"Se esperaban 1471 columnas por fila, pero se encontraron {df.shape[1]}")

# Asigna nombre a columnas (0–1469 son los valores, 1470 es la etiqueta)
df.columns = list(range(1470)) + ["label"]

# Filtra por etiqueta
df_clase = df[df["label"] == etiqueta_objetivo]

# Asegura que hay datos
if df_clase.empty:
    raise ValueError(f"No hay datos con la etiqueta: {etiqueta_objetivo}")

# Toma la primera fila y extrae la secuencia (35x42)
fila = df_clase.iloc[0, :1470].values.astype(np.float32)
secuencia = fila.reshape((35, 42))

# Muestra para comparar
print("📦 Secuencia del CSV:")
print("Forma:", secuencia.shape)
print("Varianza:", np.var(secuencia))
print("Primer frame:", secuencia[0])
