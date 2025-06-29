import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import pyttsx3
from app.utils.hand_tracking import HandTracker
from app.utils.data_processing import DataProcessor
from app.config import BASE_DIR

# Inicializar voz
engine = pyttsx3.init()

# Configuración de rutas
from app.config import MODELS_DIR
MODEL_PATH = str(MODELS_DIR / "sign_language_model.h5")
ENCODER_PATH = str(MODELS_DIR / "label_encoder.pkl")

# Cargar modelo y encoder
model = tf.keras.models.load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Inicializar componentes
tracker = HandTracker()
processor = DataProcessor()
cap = cv2.VideoCapture(0)
last_prediction = None

print("🖐️ Muestra un gesto. Presiona Q para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = tracker.detect_hands(frame)
    tracker.draw_landmarks(frame, results)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            landmarks = []
            h, w, _ = frame.shape
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))

            vector = processor.normalize_landmarks(landmarks)

            if vector is not None:
                prediction = model.predict(np.array([vector]), verbose=0)
                class_index = np.argmax(prediction)
                predicted_word = label_encoder.inverse_transform([class_index])[0]

                # Solo si cambió la predicción
                if predicted_word != last_prediction:
                    print(f"🗣️ {predicted_word}")
                    engine.say(predicted_word)
                    engine.runAndWait()
                    last_prediction = predicted_word

                # Mostrar en pantalla
                cv2.putText(frame, f"{predicted_word}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Reconocimiento Estático", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 