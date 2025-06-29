import cv2
import time
import os
import numpy as np
from app.utils.hand_tracking import HandTracker
from app.utils.data_processing import DataProcessor
from app.config import DATASET_PATH

SEQUENCE_DURATION = 4.0
FPS = 10
FRAMES_TOTAL = int(SEQUENCE_DURATION * FPS)
FRAMES_TO_SKIP = 5
FRAMES_TO_SAVE = FRAMES_TOTAL - FRAMES_TO_SKIP
LANDMARKS_PER_FRAME = 42
BOX_WIDTH = 560
BOX_HEIGHT = 480

tracker = HandTracker()
processor = DataProcessor()

CSV_PATH = str(DATASET_PATH)

def save_sequence(sequence, label, level, output_path=CSV_PATH):
    if len(sequence) != FRAMES_TO_SAVE:
        raise ValueError(f"La secuencia debe tener exactamente {FRAMES_TO_SAVE} frames.")
    for i, frame in enumerate(sequence):
        if len(frame) != LANDMARKS_PER_FRAME:
            raise ValueError(f"El frame {i} no tiene {LANDMARKS_PER_FRAME} valores.")
    row = []
    for frame in sequence:
        row.extend(frame)
    row.append(label)
    row.append(level)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='a', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(row)

# Selección de manos
while True:
    print("🙋 ¿Cuántas manos se usarán para esta seña?")
    print("  1 - Una sola mano")
    print("  2 - Dos manos")
    manos_str = input("Selecciona (1 o 2): ").strip()
    if manos_str in ["1", "2"]:
        manos_requeridas = int(manos_str)
        break
    else:
        print("⚠️ Selección no válida. Ingresa 1 o 2.")

# Etiqueta de la seña
print("🔠 Ingresa el nombre de la frase o secuencia (ej: tengo_fiebre_y_tos):")
etiqueta = input("Etiqueta: ").strip().lower().replace(" ", "_")

# Nivel de dificultad
niveles_validos = {"1": "principiante", "2": "intermedio", "3": "avanzado"}
nivel = ""
while nivel not in niveles_validos:
    print("\n📈 Selecciona el nivel de dificultad:")
    print("  1 - Principiante")
    print("  2 - Intermedio")
    print("  3 - Avanzado")
    nivel = input("Nivel (1, 2 o 3): ").strip()

nivel_nombre = niveles_validos[nivel]

print(f"\n🎥 Presiona 'R' para grabar la secuencia '{etiqueta}' con nivel '{nivel_nombre}'.")
print("Presiona 'Q' para salir.")

cap = cv2.VideoCapture(0)
recording = False
buffer_sequence = []
frames_descartados = 0
countdown = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    box_left = int(w / 2 - BOX_WIDTH / 2)
    box_top = int(h / 2 - BOX_HEIGHT / 2)
    box_right = int(w / 2 + BOX_WIDTH / 2)
    box_bottom = int(h / 2 + BOX_HEIGHT / 2)
    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 0, 255), 2)

    results = tracker.detect_hands(frame)
    tracker.draw_landmarks(frame, results)

    if countdown > 0:
        cv2.putText(frame, f"{countdown}", (w//2 - 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4)
        time.sleep(1)
        countdown -= 1
        if countdown == 0:
            print("🎬 ¡Grabando!")
            recording = True
            buffer_sequence = []
            frames_descartados = 0

    elif recording:
        if len(buffer_sequence) < FRAMES_TOTAL:
            merged = []
            hands = results.multi_hand_landmarks
            if hands and len(hands) >= manos_requeridas:
                for hand_landmarks in hands[:2]:
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                    normalized = processor.normalize_landmarks(landmarks)
                    merged.extend(normalized[:21])  # 21 puntos por mano

                if len(merged) < LANDMARKS_PER_FRAME:
                    merged += [0.0] * (LANDMARKS_PER_FRAME - len(merged))
                final_frame = merged[:LANDMARKS_PER_FRAME]

                final_frame = np.array(final_frame, dtype=np.float32)
                var = np.var(final_frame)

                if len(final_frame) == LANDMARKS_PER_FRAME and var > 1e-4:
                    buffer_sequence.append(final_frame.tolist())
                    print(f"✅ Frame {len(buffer_sequence)}/{FRAMES_TOTAL} (varianza: {var:.5f})")
                else:
                    print(f"❌ Frame descartado: varianza demasiado baja ({var:.5f})")
                    frames_descartados += 1
            else:
                print(f"❌ No se detectaron las {manos_requeridas} mano(s) requeridas.")
                frames_descartados += 1

            cv2.putText(frame, f"Grabando {len(buffer_sequence)}/{FRAMES_TOTAL}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            time.sleep(1.0 / FPS)
        else:
            try:
                trimmed_sequence = buffer_sequence[FRAMES_TO_SKIP:]
                if len(trimmed_sequence) < FRAMES_TO_SAVE:
                    raise ValueError(f"Solo {len(trimmed_sequence)} frames válidos. Se necesitan {FRAMES_TO_SAVE}.")
                final_sequence = trimmed_sequence[:FRAMES_TO_SAVE]
                save_sequence(final_sequence, etiqueta, nivel_nombre)
                print(f"✅ Secuencia guardada con {len(final_sequence)} frames.")
                print(f"📉 Frames descartados: {frames_descartados}")
            except Exception as e:
                print(f"⚠️ Error al guardar: {e}")
            buffer_sequence = []
            frames_descartados = 0
            recording = False

    cv2.imshow("Grabador de Frases", frame)

    key = cv2.waitKeyEx(1)
    if key == ord('q') or key == ord('Q'):
        print("⛔ Cerrando...")
        break

    # Si no está grabando ni en cuenta regresiva, iniciar automáticamente
    if not recording and countdown == 0:
        print("⏳ Grabación iniciará en:")
        countdown = 3

cap.release()
cv2.destroyAllWindows()
