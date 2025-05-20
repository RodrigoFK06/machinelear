import csv
import os

class DataCollector:
    def __init__(self, output_path='dataset.csv'):
        self.output_path = output_path
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.output_path):
            with open(self.output_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                # 42 columnas de puntos + 1 para la etiqueta
                header = [f'X{i}' for i in range(42)] + ['label']
                writer.writerow(header)

    def save_sample(self, vector, label):
        if vector is None:
            return
        with open(self.output_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = list(vector) + [label]
            writer.writerow(row)

    def save_sequence(self, sequence, label):
        row = []
        for frame in sequence:
            row.extend(frame)  # Aplana todos los vectores de cada frame
        row.append(label)
        with open(self.output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
