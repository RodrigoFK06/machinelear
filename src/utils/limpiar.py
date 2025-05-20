import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE_DIR, "dataset_medico.csv")


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
