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

Se proveen dos scripts actualizados para entrenar modelos desde cero utilizando `data/dataset_medico.csv`.
- `app/train_lstm_model.py` entrena un modelo LSTM.
- `app/train_cnn_lstm_model.py` entrena una arquitectura CNN+LSTM.

Configura hiperparámetros en `app/config.py` y ejecuta:
```bash
python app/train_lstm_model.py
# o
python app/train_cnn_lstm_model.py
```

## Estructura de las secuencias

Las predicciones se realizan con secuencias de **35 frames**, cada frame compuesto por **42 valores flotantes** que representan los puntos clave normalizados de ambas manos. El frontend puede obtener estos puntos con librerías como [MediaPipe](https://developers.google.com/mediapipe) o TensorFlow.js.

Ejemplo de cuerpo JSON:
```json
{
  "sequence": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2], ... 35 frames],
  "expected_label": "dolor_de_cabeza",
  "nickname": "usuario123"
}
```

`expected_label` debe pertenecer a la lista devuelta por `/labels`. `nickname` es opcional y se utiliza únicamente para generar estadísticas por usuario (en el futuro será reemplazado por autenticación JWT).

## Uso de la API

Todas las rutas están prefijadas en la raíz. No se requiere token de autorización en esta versión. Reemplaza `localhost:8000` con tu host si ejecutas en una máquina diferente.

### POST `/predict`

Predice una secuencia de señas médicas.

**URL**: `/predict`

**Método**: `POST`

**Headers**: 
- `Content-Type: application/json`

**Body**: estructura descrita en [Estructura de las secuencias](#estructura-de-las-secuencias)

**Respuesta exitosa**:
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

**Ejemplo con cURL**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/predict_example.json
```

**Códigos de error**: `400` entrada inválida, `500` error interno.

### GET `/labels`

Devuelve la lista de etiquetas disponibles en el dataset.

**URL**: `/labels`

**Método**: `GET`

**Ejemplo con cURL**:
```bash
curl http://localhost:8000/labels
```

**Códigos de error**: `404` dataset no encontrado, `500` error de lectura.

### GET `/records`

Permite consultar registros de predicción con filtros opcionales.

**URL**: `/records`

**Método**: `GET`

**Parámetros de consulta**:
- `nickname` – filtrar por nickname de usuario
- `date_from` – fecha de inicio ISO (ej., `2025-01-01T00:00:00Z`)
- `date_to` – fecha de fin ISO
- `evaluation` – `CORRECTO`, `DUDOSO` o `INCORRECTO`
- `skip` – registros a omitir (paginación)
- `limit` – máximo de registros a devolver (1‑100)

**Respuesta de ejemplo**:
```json
[
  {
    "_id": "60d5ec49f0b2f3a1c4d4a9c1",
    "nickname": "usuario123",
    "sequence_shape": [35, 42],
    "predicted_label": "dolor_de_cabeza",
    "expected_label": "dolor_de_cabeza",
    "confidence": 89.3,
    "evaluation": "CORRECTO",
    "observation": null,
    "timestamp": "2025-05-20T22:32:10.123Z"
  }
]
```

**Ejemplo con cURL**:
```bash
curl "http://localhost:8000/records?nickname=usuario123&limit=5"
```

La cabecera `X-Total-Count` indica el número total de coincidencias.

**Códigos de error**: `500` errores de consulta en base de datos.

### GET `/progress`

Devuelve estadísticas agregadas por etiqueta y, opcionalmente, por usuario si se especifica `nickname`.

**URL**: `/progress`

**Método**: `GET`

**Parámetros de consulta**:
- `nickname` – nickname de usuario opcional

**Respuesta de ejemplo**:
```json
[
  {
    "label": "dolor_de_cabeza",
    "total_attempts": 5,
    "correct_attempts": 4,
    "doubtful_attempts": 0,
    "incorrect_attempts": 1,
    "success_rate": 80.0,
    "doubtful_rate": 0.0,
    "incorrect_rate": 20.0,
    "average_confidence": 85.4,
    "max_confidence": 90.0,
    "min_confidence": 70.0,
    "last_attempt": "2025-05-20T22:32:10.123Z"
  }
]
```

**Ejemplo con cURL**:
```bash
curl "http://localhost:8000/progress?nickname=usuario123"
```

**Códigos de error**: `500` error del servidor.

### GET `/activity/daily/{nickname}/{date}`

Entrega el detalle de prácticas de un usuario en un día específico. El formato de fecha debe ser `YYYY-MM-DD`.

**URL**: `/activity/daily/{nickname}/{date}`

**Método**: `GET`

**Respuesta de ejemplo**:
```json
{
  "nickname": "usuario123",
  "date": "2025-05-20",
  "summary": {
    "total_practices": 3,
    "correct_practices": 2,
    "doubtful_practices": 1,
    "incorrect_practices": 0
  },
  "records": [
    {
      "id": "60d5ec49f0b2f3a1c4d4a9c1",
      "timestamp": "2025-05-20T10:00:00Z",
      "predicted_label": "dolor_de_cabeza",
      "expected_label": "dolor_de_cabeza",
      "confidence": 90.5,
      "evaluation": "CORRECTO"
    }
  ]
}
```

**Ejemplo con cURL**:
```bash
curl http://localhost:8000/activity/daily/usuario123/2025-05-20
```

**Códigos de error**: `400` formato de fecha inválido, `500` error de base de datos.

### GET `/stats/global_distribution`

Muestra la distribución global de evaluaciones (CORRECTO, DUDOSO, INCORRECTO).

**URL**: `/stats/global_distribution`

**Método**: `GET`

**Respuesta de ejemplo**:
```json
{
  "total_evaluations": 2000,
  "distribution": [
    {"evaluation_type": "CORRECTO", "count": 1500, "percentage": 75.0},
    {"evaluation_type": "DUDOSO", "count": 300, "percentage": 15.0},
    {"evaluation_type": "INCORRECTO", "count": 200, "percentage": 10.0}
  ]
}
```

**Ejemplo con cURL**:
```bash
curl http://localhost:8000/stats/global_distribution
```

**Códigos de error**: `500` error del servidor.

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

Puedes importar los endpoints en Postman utilizando la siguiente colección de ejemplo: [`postman_collection.json`](postman_collection.json). Si prefieres otras herramientas como Insomnia o Hoppscotch, basta con replicar las peticiones `curl` mostradas en cada endpoint.

Para pruebas rápidas también puedes crear las solicitudes manualmente utilizando los ejemplos de cURL proporcionados en cada sección.
### Nueva API ligera
Para realizar inferencias con el modelo CNN+LSTM sin depender de la estructura anterior se incluye `api/main.py`. Se inicia con:
```bash
uvicorn api.main:app --reload
```

