import os
from app.config import DATASET_PATH

path = str(DATASET_PATH)


with open(path, "r") as f:
    lines = f.readlines()

# Contar columnas esperadas
EXPECTED_COLS = 1471

# Filtrar solo líneas con 1471 columnas
filtered_lines = [line for line in lines if len(line.strip().split(",")) == EXPECTED_COLS]

# Sobrescribir archivo limpio
with open(path, "w") as f:
    f.writelines(filtered_lines)

print("✅ Dataset médico limpiado. Solo quedan secuencias válidas.")
