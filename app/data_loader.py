import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .config import DATA_PATH, TEST_SIZE, RANDOM_STATE

FRAMES = 35
FEATURES = 42


def load_dataset():
    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X = X.reshape((-1, FRAMES, FEATURES))
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    return (X_train, X_test, y_train, y_test, encoder)
