# app/utils/generate_fake_dataset.py
import os
import numpy as np
import pandas as pd
from app.config import DATASET_PATH

# Ensure the data directory exists
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

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

output_path = str(DATASET_PATH)
df.to_csv(output_path, index=False, header=False)

print(f"✅ Dataset falso guardado en: {output_path}")
