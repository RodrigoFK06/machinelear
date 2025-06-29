import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FRAME_PATH = "frame_actual.txt"
CSV_PATH = "D:/machinelear/data/dataset_medico.csv"  # o la ruta donde esté realmente
LABEL = "hola_buenos_dias"

# Cargar frame real
frame_actual = np.loadtxt(FRAME_PATH, delimiter=",")
if frame_actual.shape[0] != 42:
    raise ValueError("Frame actual no tiene 42 elementos")

# Cargar CSV
df = pd.read_csv(CSV_PATH, header=None)
df.columns = [f"f{i}" for i in range(1470)] + ["label"]

# Buscar un ejemplo del mismo label
ejemplo = df[df["label"] == LABEL].iloc[0, :42].values.astype(np.float32)

# Graficar comparación
plt.figure(figsize=(12, 4))
plt.plot(frame_actual, label="Frame Actual (grabado)", marker="o")
plt.plot(ejemplo, label=f"Ejemplo CSV: {LABEL}", marker="x")
plt.title("Comparación: Frame actual vs CSV")
plt.xlabel("Feature index (x/y alternado)")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
