import cv2
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
import os
import threading
import time
import csv
from datetime import datetime
from collections import defaultdict

from app.utils.hand_tracking import HandTracker
from app.utils.data_processing import DataProcessor
from app.legacy.sequence_recorder import SequenceRecorder
from app.legacy.segment_analyzer import analizar_error

PREDICTION_COOLDOWN = 2
GRABACION_DURACION = 3

def reproducir_audio(texto):
    engine = pyttsx3.init()
    engine.say(texto)
    engine.runAndWait()

def main():
    print("🧠 ARCHIVO CORRECTO: realtime_lstm_predictor.py")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "src", "lstm_gestos_model.h5")
    ENCODER_PATH = os.path.join(BASE_DIR, "src", "label_encoder_lstm.pkl")
    REGISTROS_DIR = os.path.join(BASE_DIR, "registros")
    os.makedirs(REGISTROS_DIR, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    tracker = HandTracker()
    processor = DataProcessor()
    recorder = SequenceRecorder(max_length=35)

    nombre_usuario = input("👤 Ingrese su nombre: ").strip()
    nombre_sena = input("✋ Ingrese el nombre de la seña que va a realizar: ").strip().lower()
    if nombre_sena not in encoder.classes_:
        print(f"⚠️ La seña '{nombre_sena}' no está registrada en el modelo.")
        return

    estadisticas = defaultdict(lambda: {"intentos": 0, "aciertos": 0, "confianza_total": 0})

    cap = cv2.VideoCapture(0)
    print("📹 Presiona R para iniciar. Q para salir.")

    modo_espera = True
    grabando = False
    tiempo_inicio = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('r') and modo_espera:
            modo_espera = False
            grabando = True
            tiempo_inicio = time.time()
            recorder.reset()
            print("🕒 Preparándote para grabar...")

        if grabando:
            tiempo_transcurrido = time.time() - tiempo_inicio
            if tiempo_transcurrido <= 3:
                contador = int(3 - tiempo_transcurrido) + 1
                cv2.putText(frame, f"Prepárate: {contador}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            elif tiempo_transcurrido <= 3 + GRABACION_DURACION:
                results = tracker.detect_hands(frame)
                tracker.draw_landmarks(frame, results)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    h, w, _ = frame.shape
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
                    vector = processor.normalize_landmarks(landmarks)
                    recorder.add_frame(vector)

                cv2.putText(frame, "🎥 Grabando seña...", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            else:
                sequence = recorder.get_sequence()
                if sequence is not None:
                    if len(sequence) != 35:
                        print(f"⚠️ Se capturaron solo {len(sequence)} frames. Se requieren exactamente 35.")
                        cv2.putText(frame, "❗ Secuencia incompleta. Intenta otra vez.", (10, frame.shape[0] - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        grabando = False
                        modo_espera = True
                        continue

                    prediction = model.predict(np.array([sequence]), verbose=0)
                    class_index = np.argmax(prediction)
                    confidence = float(prediction[0][class_index]) * 100
                    predicted_phrase = encoder.inverse_transform([class_index])[0]

                    # UMBRAL ESTRICTO
                    UMBRAL_CONFIANZA = 75  # Puedes subirlo a 80 si aún te parece muy permisivo

                    if predicted_phrase.lower() == nombre_sena and confidence >= UMBRAL_CONFIANZA:
                        evaluacion = "CORRECTO"
                    elif predicted_phrase.lower() == nombre_sena and confidence < UMBRAL_CONFIANZA:
                        evaluacion = "DUDOSO"
                    else:
                        evaluacion = "INCORRECTO"

                    color = (0, 255, 0) if evaluacion == "CORRECTO" else (0, 255, 255) if evaluacion == "DUDOSO" else (0, 0, 255)
                    output_file = f"dataset_{evaluacion.lower()}.csv"
                    csv_path = os.path.join(REGISTROS_DIR, output_file)

                    if evaluacion == "INCORRECTO":
                        dataset_path = os.path.join(BASE_DIR, "src", "dataset_medico.csv")
                        observacion = analizar_error(sequence, dataset_path, nombre_sena)
                        print(observacion)

                        cv2.putText(frame, observacion, (10, frame.shape[0] - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

                        threading.Thread(target=reproducir_audio, args=(observacion,), daemon=True).start()

                        with open(csv_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                datetime.now().isoformat(),
                                nombre_usuario,
                                nombre_sena,
                                predicted_phrase,
                                evaluacion,
                                round(confidence, 2),
                                recorder.get_sequence().tolist(),
                                observacion
                            ])
                    else:
                        threading.Thread(target=reproducir_audio, args=("Bien hecho" if evaluacion == "CORRECTO" else "Intenta nuevamente",), daemon=True).start()
                        with open(csv_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                datetime.now().isoformat(),
                                nombre_usuario,
                                nombre_sena,
                                predicted_phrase,
                                evaluacion,
                                round(confidence, 2),
                                recorder.get_sequence().tolist()
                            ])

                    est = estadisticas[nombre_sena]
                    est["intentos"] += 1
                    if evaluacion == "CORRECTO":
                        est["aciertos"] += 1
                    est["confianza_total"] += confidence

                grabando = False
                modo_espera = True
                print("✅ Fin de ciclo. Presiona R para iniciar.")

        est = estadisticas[nombre_sena]
        porcentaje_exito = (est["aciertos"] / est["intentos"]) * 100 if est["intentos"] > 0 else 0
        promedio_confianza = est["confianza_total"] / est["intentos"] if est["intentos"] > 0 else 0

        cv2.putText(frame, nombre_usuario, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 150, 0), 2)

        if modo_espera:
            cv2.putText(frame, "Presiona R para iniciar", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        texto_stats = f"N:{est['intentos']}  NA:{est['aciertos']}  P:{porcentaje_exito:.1f}%  PC:{promedio_confianza:.1f}%"
        text_size = cv2.getTextSize(texto_stats, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        x_offset = frame.shape[1] - text_size[0] - 20
        cv2.putText(frame, texto_stats, (x_offset, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

        cv2.imshow("Detector de Secuencias", frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
