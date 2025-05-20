import numpy as np

class DataProcessor:
    def normalize_landmarks(self, landmarks):
        """
        Normaliza los puntos clave de la mano para que sean independientes
        del tamaño y la posición en la imagen.
        """
        if not landmarks:
            return None

        # Convertimos a array de numpy por facilidad
        landmarks_array = np.array(landmarks)

        # Usamos el primer punto (muñeca) como origen
        base_x, base_y = landmarks_array[0]
        normalized = [(x - base_x, y - base_y) for x, y in landmarks_array]

        # Aplanamos el array (de 21 puntos a un vector de 42 valores)
        return np.array(normalized).flatten()
