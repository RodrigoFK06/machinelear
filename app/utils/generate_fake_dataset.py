# app/utils/generate_fake_dataset.py
import os
import numpy as np
import pandas as pd

# Asegurarse de que exista la carpeta 'data' en la raíz del proyecto
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
os.makedirs(data_dir, exist_ok=True)

N = 100
frames = 35
features = 42
labels = ['dolor_de_cabeza', 'mareo', 'fatiga']

rows = []
for _ in range(N):
    data = np.random.rand(frames * features).tolist()
    label = np.random.choice(labels)
    rows.append(data + [label])

df = pd.DataFrame(rows)

output_path = os.path.join(data_dir, 'dataset_medico.csv')
df.to_csv(output_path, index=False, header=False)

print(f"✅ Dataset falso guardado en: {output_path}")
