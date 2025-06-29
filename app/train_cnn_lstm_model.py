import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dropout,
    BatchNormalization, LSTM, Dense, Input
)
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os

from .data_loader import load_dataset
from .model_utils import save_model, save_encoder, plot_metrics
from .config import (
    EPOCHS,
    BATCH_SIZE,
    CNN_LSTM_MODEL_PATH,
    ENCODER_PATH,
    CNN_LSTM_PLOT_PATH,
)


def build_model(num_classes: int) -> Sequential:
    model = Sequential([
        Input(shape=(35, 42)),
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(),
        Dropout(0.4),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(),
        Dropout(0.4),

        LSTM(128, return_sequences=True),
        LSTM(64),

        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_confusion(y_true, y_pred, encoder, save_path=None):
    if save_path is None:
        from app.config import CONFUSION_MATRIX_PATH
        save_path = str(CONFUSION_MATRIX_PATH)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = encoder.classes_
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.xticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)) + 0.5, labels=labels, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Matriz de confusión guardada en: {save_path}")


def main():
    X_train, X_test, y_train, y_test, encoder = load_dataset()
    # ✅ Calcular y guardar mean y std de todo X_train para normalización posterior
    mean = np.mean(X_train, axis=(0, 1))  # Promedio por cada feature
    std = np.std(X_train, axis=(0, 1))  # Desviación estándar por cada feature

    # 💾 Guardar en archivos binarios .npy
    os.makedirs("app/models", exist_ok=True)
    np.save("app/models/mean.npy", mean)
    np.save("app/models/std.npy", std)

    print("\n📁 mean.npy y std.npy guardados correctamente en app/models/")
    print("🧠 Ejemplo de mean[:5]:", mean[:5])
    print("🧠 Ejemplo de std[:5]:", std[:5])

    # 🔍 Verificación rápida del dataset cargado
    print("\n🔍 Ejemplo de valores normalizados:")
    print("Primer frame del primer ejemplo (X_train[0][0]):")
    print(X_train[0][0])  # Muestra 42 valores del primer frame

    print("\n📏 Máximos por frame:", np.max(X_train[0][0]))
    print("📏 Mínimos por frame:", np.min(X_train[0][0]))

    print("\n🔎 Validación del dataset cargado:")
    print(f"➡️ Clases detectadas por el encoder: {encoder.classes_}")
    print(f"➡️ Total de clases: {len(encoder.classes_)}")

    print(f"🟢 y_train contiene clases: {np.unique(y_train)} (total: {len(y_train)})")
    print(f"🔵 y_test contiene clases: {np.unique(y_test)} (total: {len(y_test)})")

    # Distribución por clase
    import collections
    train_dist = collections.Counter(y_train)
    test_dist = collections.Counter(y_test)

    print("\n📊 Distribución en y_train:")
    for label, count in train_dist.items():
        class_name = encoder.inverse_transform([label])[0]
        print(f"  - {class_name}: {count} muestras")

    print("\n📊 Distribución en y_test:")
    for label, count in test_dist.items():
        class_name = encoder.inverse_transform([label])[0]
        print(f"  - {class_name}: {count} muestras")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    model = build_model(len(encoder.classes_))

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        shuffle=True
    )

    save_model(model, CNN_LSTM_MODEL_PATH)
    save_encoder(encoder, ENCODER_PATH)
    plot_metrics(history, CNN_LSTM_PLOT_PATH)

    print("\n📊 Evaluación en test set completo:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Accuracy en test: {test_acc:.4f} - Loss: {test_loss:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Clasificación textual
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    print(report)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Matriz de confusión
    plot_confusion(y_test, y_pred, encoder)

    # Muestra algunas predicciones
    for i in range(min(5, len(X_test))):
        real = encoder.inverse_transform([y_test[i]])[0]
        pred = encoder.inverse_transform([np.argmax(y_pred_probs[i])])[0]
        print(f"▶️ Real: {real}")
        print(f"🤖 Pred: {pred}")
        print(f"📊 Probabilidades: {np.round(y_pred_probs[i], 3)}")
        print("-" * 30)


if __name__ == '__main__':
    main()
