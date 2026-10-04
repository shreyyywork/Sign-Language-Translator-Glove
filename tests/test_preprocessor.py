"""
Unit tests for Sensor Preprocessor.
"""

import numpy as np
import pandas as pd
from ml.preprocessor import SensorPreprocessor


def test_preprocessor_clean_and_fit():
    data = {
        "timestamp": [1.0, 2.0, 3.0, 4.0],
        "user_id": ["U01", "U01", "U02", "U02"],
        "gesture": ["HELLO", "HELLO", "YES", "YES"],
        "thumb": [0.1, 0.12, 0.9, 0.95],
        "index": [0.05, 0.08, 0.95, 0.98],
        "middle": [0.05, 0.06, 0.95, 0.95],
        "ring": [0.05, 0.05, 0.95, 0.95],
        "little": [0.05, 0.05, 0.95, 0.95],
        "acc_x": [0.0, 0.01, 0.1, 0.12],
        "acc_y": [0.98, 0.97, 0.7, 0.72],
        "acc_z": [0.15, 0.14, 0.65, 0.64],
        "gyro_x": [0.0, 0.0, 0.0, 0.0],
        "gyro_y": [0.0, 0.0, 0.0, 0.0],
        "gyro_z": [0.0, 0.0, 0.0, 0.0]
    }
    df = pd.DataFrame(data)

    prep = SensorPreprocessor()
    X, y = prep.extract_features_and_labels(df)
    assert X.shape == (4, 11)
    assert len(y) == 4

    X_scaled, y_enc = prep.fit_transform(X, y)
    assert X_scaled.shape == (4, 11)
    assert len(y_enc) == 4
    assert prep.is_fitted is True


def test_single_sample_transform():
    prep = SensorPreprocessor()
    # Dummy fit
    dummy_X = np.random.randn(10, 11)
    prep.fit(dummy_X)

    sample = {
        "thumb": 0.5, "index": 0.5, "middle": 0.5, "ring": 0.5, "little": 0.5,
        "acc_x": 0.0, "acc_y": 1.0, "acc_z": 0.0,
        "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0
    }
    scaled = prep.transform_single_sample(sample)
    assert scaled.shape == (1, 11)
