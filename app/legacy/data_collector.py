import csv
import os

class DataCollector:
    def __init__(self, output_path='dataset.csv', num_landmarks=21, sequence_len=35):
        self.output_path = output_path
        self.num_landmarks = num_landmarks
        self.sequence_len = sequence_len
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.output_path):
            with open(self.output_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = []
                for frame_idx in range(self.sequence_len):
                    for i in range(self.num_landmarks):
                        header += [f'X{frame_idx}_{i}', f'Y{frame_idx}_{i}']
                header.append('label')
                writer.writerow(header)

    def save_sample(self, vector, label):
        if vector is None:
            return
        with open(self.output_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = list(vector) + [label]
            writer.writerow(row)

    def save_sequence(self, sequence, label):
        if len(sequence) != self.sequence_len:
            print(f"⚠️ Secuencia incompleta ({len(sequence)} frames). Se esperaban {self.sequence_len}. No se guarda.")
            return
        flat_sequence = []
        for frame in sequence:
            flat_sequence.extend(frame)
        flat_sequence.append(label)
        with open(self.output_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flat_sequence)
