import cv2
import time
import os
from app.utils.hand_tracking import HandTracker
from app.utils.data_processing import DataProcessor
from app.legacy.data_collector import DataCollector

SEQUENCE_DURATION = 3.5
FPS = 10
FRAMES_NEEDED = int(SEQUENCE_DURATION * FPS)
BOX_WIDTH = 560
BOX_HEIGHT = 480

tracker = HandTracker()
processor = DataProcessor()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "dataset_medico.csv")
collector = DataCollector(output_path=CSV_PATH)

cap = cv2.VideoCapture(0)

print("🔠 Ingresa el nombre de la frase o secuencia (ej: me_duele_la_cabeza):")
etiqueta = input("Etiqueta: ").strip().lower().replace(" ", "_")

print("\n🎥 Presiona 'R' para grabar la secuencia de la frase.")
print("Presiona 'Q' para salir.")

recording = False
sequence = []
start_time = None

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
    cv2.putText(frame, "Presiona 'R' para grabar, 'Q' para salir", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    results = tracker.detect_hands(frame)
    tracker.draw_landmarks(frame, results)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

        if all(box_left < x < box_right and box_top < y < box_bottom for (x, y) in landmarks):
            normalized = processor.normalize_landmarks(landmarks)

            if recording:
                if len(sequence) < FRAMES_NEEDED:
                    sequence.append(normalized)
                    cv2.putText(frame, "Grabando...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    time.sleep(1.0 / FPS)
                else:
                    if len(sequence) == FRAMES_NEEDED:
                        collector.save_sequence(sequence, etiqueta)
                        print(f"✅ Secuencia de {len(sequence)} muestras guardada en dataset_medico.csv")
                    else:
                        print(f"⚠️ Secuencia incompleta ({len(sequence)} frames). No se guardó.")
                    sequence = []
                    recording = False

    cv2.imshow("Grabador de Frases", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = True
        sequence = []
        start_time = time.time()
        print("🎬 Grabando...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
