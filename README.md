# Medical Sign Recognition System

Este proyecto implementa un sistema de reconocimiento de señas médicas basado en una arquitectura **CNN+LSTM**. Incluye los scripts de entrenamiento y evaluación, una API ligera con FastAPI y utilidades para generar datasets de prueba.

## Tabla de contenidos
- [Requisitos](#requisitos)
- [Instalación](#instalacion)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Preparación del dataset](#preparacion-del-dataset)
- [Entrenamiento](#entrenamiento)
- [Evaluación del modelo](#evaluacion-del-modelo)
- [Uso de la API](#uso-de-la-api)
- [Pruebas](#pruebas)
- [Legacy](#legacy)

## Requisitos
- Python 3.11 o superior
- `virtualenv` o similar para aislar dependencias

## Instalacion
1. Clona este repositorio.
2. Crea y activa un entorno virtual:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Estructura del proyecto
- **app/**: utilidades de entrenamiento y evaluacion.
- **api/**: API FastAPI para realizar inferencias con el modelo CNN+LSTM.
- **models/**: modelos entrenados y artefactos generados.
- **data/**: dataset en CSV (real o sintético).
- **tests/**: pruebas unitarias con `pytest`.
- **config.py**: define rutas y parámetros globales usados por los scripts y la API.

## Preparacion del dataset
El modelo se entrena con `data/dataset_medico.csv`, donde cada fila contiene 35×42 valores (puntos clave de ambas manos) y la etiqueta correspondiente.

Si no cuentas con datos reales puedes generar un dataset ficticio ejecutando:
```bash
python app/utils/generate_fake_dataset.py
```
El archivo se guardará en `data/dataset_medico.csv`.

## Entrenamiento
El script principal para entrenar el modelo actual es:
```bash
python app/train_cnn_lstm_model.py
```
Generará `models/cnn_lstm_model.h5` y `models/label_encoder.pkl`. Además se guardará la curva de pérdida en `models/loss_plot_cnn_lstm.png`.

La arquitectura CNN+LSTM añade capas convolucionales previas al LSTM, lo que mejora la extracción de características frente al modelo LSTM puro (`train_lstm_model.py`, mantenido solo con fines históricos).

## Evaluacion del modelo
Para evaluar el rendimiento del modelo entrenado:
```bash
python app/evaluate_model.py
```
Se obtendrán las métricas en `models/metrics.json`, la matriz de confusión en `models/confusion_matrix.png` y el reporte de clasificación en `models/classification_report.txt`.

## Uso de la API
Inicia la API ligera con:
```bash
uvicorn api.main:app --reload
```
Cada petición a `/predict` se registrará en `models/inference_log.csv` con la fecha UTC, el *nickname* (si se envía) y la confianza obtenida.

### Ejemplo de solicitud
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": [[0.0, ..., 0.1]], "nickname": "demo"}'
```
La secuencia debe contener 35 arreglos de 42 flotantes. La respuesta incluye la etiqueta predicha, la confianza y el vector de probabilidades.

## Pruebas
Las pruebas unitarias se ejecutan con:
```bash
pytest -q
```

## Legacy
El repositorio conserva scripts y archivos del modelo LSTM original (`lstm_gestos_model.h5` y `label_encoder_lstm.pkl`). También existe una configuración `.env` orientada a una API con MongoDB. Estos elementos se mantienen por compatibilidad, pero el flujo recomendado utiliza el modelo CNN+LSTM y no requiere variables de entorno.
