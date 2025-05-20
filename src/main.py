import cv2
import time
from detection.hand_tracking import HandTracker
from preprocessing.data_processing import DataProcessor
from utils.data_collector import DataCollector
import os
import numpy as np

# Configuración
NOMBRE_ARCHIVO = "dataset_medico.csv"
RUTA_ARCHIVO = NOMBRE_ARCHIVO

# Zona de detección
BOX_WIDTH = 300
BOX_HEIGHT = 380

# Configuración de grabación
CAPTURE_KEY = ord('y')  # Tecla para iniciar grabación
STOP_KEY = ord('t')     # Tecla para terminar grabación
BRIGHTNESS_THRESHOLD = 60  # Umbral para detectar poca luz (0-255)

def calcular_brillo_promedio(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara seleccionada.")
        exit()
    else:
        print("📸 Cámara integrada activada correctamente.")

    tracker = HandTracker()
    processor = DataProcessor()
    collector = DataCollector(output_path=RUTA_ARCHIVO)

    print("🔠 Ingresa la palabra que deseas capturar (ej: FIEBRE, DOLOR_PECHO):")
    current_label = input("Etiqueta: ").strip().upper()

    print("\n🎥 Cámara activa. Presiona [Y] para grabar. [T] para detener. [N] para nueva etiqueta. [ESC] para salir.\n")

    recording = False
    countdown = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Coordenadas del cuadro de detección
        box_left = int(w / 2 - BOX_WIDTH / 2)
        box_top = int(h / 2 - BOX_HEIGHT / 2)
        box_right = int(w / 2 + BOX_WIDTH / 2)
        box_bottom = int(h / 2 + BOX_HEIGHT / 2)

        # Dibujar el cuadro rojo
        cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), (0, 0, 255), 2)

        # Verificar iluminación
        brillo = calcular_brillo_promedio(frame)
        if brillo < BRIGHTNESS_THRESHOLD:
            cv2.putText(frame, "⚠️ Iluminacion insuficiente", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        results = tracker.detect_hands(frame)
        tracker.draw_landmarks(frame, results)

        if countdown > 0:
            cv2.putText(frame, f"Comenzando en {countdown}", (w//2 - 80, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            time.sleep(1)
            countdown -= 1
            if countdown == 0:
                recording = True
            cv2.imshow("Recolector de Gestos Médicos", frame)
            key = cv2.waitKey(1) & 0xFF
            continue

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    landmarks.append((x, y))

                # Verificar que la mano esté dentro del área
                if all(box_left < x < box_right and box_top < y < box_bottom for (x, y) in landmarks):
                    if recording:
                        vector = processor.normalize_landmarks(landmarks)
                        if vector is not None and current_label:
                            collector.save_sample(vector, current_label)
                            print(f"✅ Muestra guardada con etiqueta: {current_label}")
                else:
                    cv2.putText(frame, "✋ Fuera del area de deteccion", (10, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Mostrar instrucciones
        estado = "Grabando..." if recording else "En espera"
        cv2.putText(frame, f"Estado: {estado}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Etiqueta: {current_label}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Recolector de Gestos Médicos", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('n') or key == ord('N'):
            print("\n🔠 Ingresa nueva etiqueta:")
            current_label = input("Etiqueta: ").strip().upper()
            recording = False
        elif key == CAPTURE_KEY:  # Iniciar grabación
            print("⏳ Iniciando grabacion en 3 segundos...")
            countdown = 3
        elif key == STOP_KEY:  # Detener grabación
            recording = False
            print("🛑 Grabacion detenida.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
