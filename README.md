# Medical Sign Recognition API

Este backend en **FastAPI** provee servicios de reconocimiento de señas médicas mediante un modelo LSTM entrenado con secuencias de puntos clave de las manos. Está pensado para integrarse con aplicaciones web o móviles que capturen gestos en tiempo real y necesiten obtener una predicción inmediata junto con métricas de rendimiento.

## Tabla de contenidos
- [Instalación](#instalación)
- [Entrenamiento del modelo](#entrenamiento-del-modelo)
- [Estructura de las secuencias](#estructura-de-las-secuencias)
- [Uso de la API](#uso-de-la-api)
  - [/predict](#post-predict)
  - [/labels](#get-labels)
  - [/records](#get-records)
  - [/progress](#get-progress)
  - [/activity/daily](#get-activitydailynicknamedate)
  - [/stats/global_distribution](#get-statsglobal_distribution)
- [Integración desde frontend](#integración-desde-frontend)
- [Pruebas automatizadas](#pruebas-automatizadas)
- [Colección Postman](#colección-postman)

## Instalación
1. Clona el repositorio y crea un entorno virtual de Python 3.11 o superior.
2. Copia `.env.example` a `.env` y ajusta las variables:
   - `MONGO_URI` con tu instancia de MongoDB
   - `MODEL_PATH` ruta al archivo `lstm_gestos_model.h5`
   - `ENCODER_PATH` ruta al archivo `label_encoder_lstm.pkl`
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta el servidor en modo desarrollo:
   ```bash
   uvicorn app.main:app --reload
   ```

## Entrenamiento del modelo
El script `app/legacy/train_lstm_model.py` permite entrenar el modelo LSTM a partir del archivo `dataset_medico.csv` (35×42 características por frame). Al finalizar se guarda `lstm_gestos_model.h5` y el codificador de etiquetas `label_encoder_lstm.pkl` que son cargados automáticamente por la API.

Para lanzar el entrenamiento:
```bash
python app/legacy/train_lstm_model.py
```
Asegúrate de que `dataset_medico.csv` esté limpio y estructurado correctamente (la última columna debe contener la etiqueta de la seña).

## Estructura de las secuencias
Las predicciones se realizan con secuencias de **35 frames**, cada frame compuesto por **42 valores flotantes** que representan los puntos clave normalizados de ambas manos. El frontend puede obtener estos puntos con librerías como [MediaPipe](https://developers.google.com/mediapipe) o TensorFlow.js.

Ejemplo de cuerpo JSON:
```json
{
  "sequence": [[0.1, 0.2, ... 0.42], ... 35 frames],
  "expected_label": "dolor_de_cabeza",
  "nickname": "usuario123"
}
```
`expected_label` debe pertenecer a la lista devuelta por `/labels`. `nickname` es opcional y se utiliza únicamente para generar estadísticas por usuario (en el futuro será reemplazado por autenticación JWT).

## Uso de la API
Todas las rutas están prefijadas en la raíz. No se requiere token de autorización en esta versión.

### POST `/predict`
- **Headers**: `Content-Type: application/json`
- **Body**: estructura descrita en [Estructura de las secuencias](#estructura-de-las-secuencias)
- **Respuesta exitosa**:
  ```json
  {
    "predicted_label": "dolor_de_cabeza",
    "confidence": 89.3,
    "evaluation": "CORRECTO",
    "observation": null,
    "success_rate": 74.2,
    "average_confidence": 82.9
  }
  ```
- **Curl**:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d @data/predict_example.json
  ```
- **Códigos de error**: `400` entrada inválida, `500` error interno.

### GET `/labels`
- Devuelve la lista de etiquetas disponibles en el dataset.
- **Curl**:
  ```bash
  curl http://localhost:8000/labels
  ```
- **Códigos de error**: `404` dataset no encontrado, `500` error de lectura.

### GET `/records`
- Permite consultar registros de predicción filtrando por `nickname`, rango de fechas (`date_from`, `date_to`) y `evaluation`.
- **Curl**:
  ```bash
  curl "http://localhost:8000/records?nickname=usuario123&limit=5"
  ```
- La cabecera `X-Total-Count` indica el número total de coincidencias.

### GET `/progress`
- Devuelve estadísticas agregadas por etiqueta y, opcionalmente, por usuario si se especifica `nickname`.
- **Curl**:
  ```bash
  curl "http://localhost:8000/progress?nickname=usuario123"
  ```

### GET `/activity/daily/{nickname}/{date}`
- Entrega el detalle de prácticas de un usuario en un día específico (formato de fecha `YYYY-MM-DD`).
- **Curl**:
  ```bash
  curl http://localhost:8000/activity/daily/usuario123/2025-05-20
  ```

### GET `/stats/global_distribution`
- Muestra la distribución global de evaluaciones (CORRECTO, DUDOSO, INCORRECTO).
- **Curl**:
  ```bash
  curl http://localhost:8000/stats/global_distribution
  ```

## Integración desde frontend
1. Captura los puntos clave de ambas manos con MediaPipe o TensorFlow.js.
2. Normaliza los valores de cada frame entre 0 y 1 y construye un arreglo de 35×42.
3. Envía la secuencia al endpoint `/predict` en formato JSON como se mostró anteriormente.
4. Utiliza la respuesta para mostrar la etiqueta predicha y la confianza al usuario. Guarda también la evaluación para generar métricas locales si lo deseas.

Aunque actualmente se usa el campo `nickname`, se recomienda planificar un flujo de autenticación con JWT para identificar al usuario y registrar su progreso de manera segura.

### Ejemplo en React (fetch)
```javascript
const body = { sequence, expected_label: "dolor_de_cabeza", nickname: "demo" };
fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body)
}).then(res => res.json()).then(console.log);
```

### Ejemplo en Flutter (http)
```dart
final response = await http.post(
  Uri.parse('http://localhost:8000/predict'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'sequence': sequence,
    'expected_label': 'dolor_de_cabeza',
    'nickname': 'demo'
  }),
);
```

## Pruebas automatizadas
El repositorio incluye pruebas unitarias con **pytest**. Para ejecutarlas:
```bash
pytest -q
```
Los tests utilizan stubs para evitar dependencias reales de TensorFlow y MongoDB, por lo que pueden correr sin configurar servicios externos.

## Colección Postman
Puedes importar los endpoints en Postman utilizando la siguiente colección de ejemplo: [`postman_collection.json`](postman_collection.json). Si prefieres otras herramientas como Insomnia o Hoppscotch, basta con replicar las peticiones `curl` mostradas.


