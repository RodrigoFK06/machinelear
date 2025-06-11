import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from .data_loader import load_dataset
from .model_utils import load_keras_model
from .config import (
    CNN_LSTM_MODEL_PATH,
    METRICS_JSON_PATH,
    CONFUSION_MATRIX_PATH,
    REPORT_PATH,
)


def main():
    X_train, X_test, y_train, y_test, encoder = load_dataset()
    model = load_keras_model(CNN_LSTM_MODEL_PATH)
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weight = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('Accuracy:', acc)
    print('F1 macro:', f1_macro)
    print('F1 weighted:', f1_weight)
    print('Confusion Matrix:\n', cm)
    print('Report:\n', report)

    # Save metrics
    with open(METRICS_JSON_PATH, 'w') as f:
        json.dump({'accuracy': acc, 'f1_macro': f1_macro, 'f1_weighted': f1_weight}, f)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


if __name__ == '__main__':
    main()
