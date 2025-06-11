import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from .data_loader import load_dataset
from .model_utils import save_model, save_encoder, plot_metrics
from .config import (
    EPOCHS,
    BATCH_SIZE,
    LSTM_MODEL_PATH,
    ENCODER_PATH,
    LSTM_PLOT_PATH,
)


def build_model(num_classes: int) -> Sequential:
    model = Sequential([
        LSTM(64, input_shape=(35, 42)),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    X_train, X_test, y_train, y_test, encoder = load_dataset()
    model = build_model(len(encoder.classes_))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    save_model(model, LSTM_MODEL_PATH)
    save_encoder(encoder, ENCODER_PATH)
    plot_metrics(history, LSTM_PLOT_PATH)
    return history


if __name__ == '__main__':
    main()
