import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model


def save_model(model, path):
    model.save(path)


def load_keras_model(path):
    return load_model(path)


def save_encoder(encoder, path):
    joblib.dump(encoder, path)


def load_encoder(path):
    return joblib.load(path)


def plot_metrics(history, out_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.legend()
    if out_path:
        plt.savefig(out_path)
    plt.close()
