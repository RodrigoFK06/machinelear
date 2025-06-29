from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Dataset and model paths
DATASET_PATH = DATA_DIR / "dataset_medico.csv"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.h5"
CNN_LSTM_MODEL_PATH = MODELS_DIR / "cnn_lstm_model.h5"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# Training parameters
EPOCHS = 25
BATCH_SIZE = 8
DATA_PATH = str(DATASET_PATH)  # For backward compatibility
TEST_SIZE = 0.2
RANDOM_STATE = 42
LSTM_PLOT_PATH = 'models/loss_plot_lstm.png'
CNN_LSTM_PLOT_PATH = 'models/loss_plot_cnn_lstm.png'
METRICS_JSON_PATH = 'models/metrics.json'
CONFUSION_MATRIX_PATH = 'models/confusion_matrix.png'
REPORT_PATH = 'models/classification_report.txt'
INFERENCE_LOG_PATH = 'models/inference_log.csv'
