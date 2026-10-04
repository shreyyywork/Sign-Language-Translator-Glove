"""
Data Preprocessing Module for ISL Wearable Sensor Glove.
Handles:
  - Feature validation and cleaning
  - Calibration normalization (mapping raw ADC to 0.0-1.0 and IMU bias offset)
  - Feature scaling with Scikit-Learn StandardScaler / MinMaxScaler
  - Label encoding and inverse decoding
  - Sliding window extraction for dynamic time-series gesture recognition
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SensorPreprocessor:
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        scaler_path: Optional[str] = None,
        encoder_path: Optional[str] = None
    ):
        self.feature_names = feature_names or [
            "thumb", "index", "middle", "ring", "little",
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]
        self.scaler: StandardScaler = StandardScaler()
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.is_fitted: bool = False

        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.is_fitted = True
        if encoder_path and os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe: drop NaNs, strip strings, filter required columns."""
        df_clean = df.copy()
        
        # Ensure required feature columns exist
        for col in self.feature_names:
            if col not in df_clean.columns:
                raise ValueError(f"Missing required feature column: {col}")
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        df_clean = df_clean.dropna(subset=self.feature_names)

        # Clip sensor values to physical limits to remove noise spikes
        for f in ["thumb", "index", "middle", "ring", "little"]:
            # If normalized range [0, 1]
            if df_clean[f].max() <= 1.5:
                df_clean[f] = df_clean[f].clip(0.0, 1.0)
            else:
                # Raw ADC range [0, 4095]
                df_clean[f] = df_clean[f].clip(0.0, 4095.0)

        # Accel clip ±4g, Gyro clip ±20 rad/s
        for a in ["acc_x", "acc_y", "acc_z"]:
            df_clean[a] = df_clean[a].clip(-4.0, 4.0)
        for g in ["gyro_x", "gyro_y", "gyro_z"]:
            df_clean[g] = df_clean[g].clip(-20.0, 20.0)

        return df_clean

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SensorPreprocessor":
        """Fit scaler and optional label encoder."""
        self.scaler.fit(X)
        if y is not None:
            self.label_encoder.fit(y)
        self.is_fitted = True
        return self

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Scale features using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor scaler is not fitted yet.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit scaler and transform features."""
        self.fit(X, y)
        X_scaled = self.transform_features(X)
        y_encoded = self.label_encoder.transform(y) if y is not None else None
        return X_scaled, y_encoded

    def extract_features_and_labels(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract numpy feature matrix X and label vector y from cleaned dataframe."""
        df_clean = self.clean_dataframe(df)
        X = df_clean[self.feature_names].values.astype(np.float32)
        y = df_clean["gesture"].values.astype(str)
        return X, y

    def transform_single_sample(self, sample_dict: Dict[str, float]) -> np.ndarray:
        """Extract and scale a single 11-feature dictionary for real-time inference."""
        raw_vals = [float(sample_dict.get(k, 0.0)) for k in self.feature_names]
        feat_array = np.array([raw_vals], dtype=np.float32)
        if self.is_fitted:
            return self.scaler.transform(feat_array)
        return feat_array

    def create_sliding_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        step_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D sequence tensor (samples, sequence_length, features) for dynamic LSTM model.
        Groups by user_id and gesture to prevent cross-gesture sequence contamination.
        """
        df_clean = self.clean_dataframe(df)
        X_seqs = []
        y_seqs = []

        # Group by user and continuous gesture runs
        group_cols = ["user_id", "gesture"] if "user_id" in df_clean.columns else ["gesture"]
        for _, group in df_clean.groupby(group_cols):
            feat_vals = group[self.feature_names].values.astype(np.float32)
            if self.is_fitted:
                feat_vals = self.scaler.transform(feat_vals)
            gesture_label = group["gesture"].iloc[0]

            n_samples = len(feat_vals)
            if n_samples >= sequence_length:
                for start_idx in range(0, n_samples - sequence_length + 1, step_size):
                    window = feat_vals[start_idx : start_idx + sequence_length]
                    X_seqs.append(window)
                    y_seqs.append(gesture_label)

        if not X_seqs:
            raise ValueError(f"No sequences of length {sequence_length} could be created. Dataset too small.")

        X_array = np.array(X_seqs, dtype=np.float32)
        y_array = np.array(y_seqs)

        if self.is_fitted and hasattr(self.label_encoder, "classes_"):
            y_encoded = self.label_encoder.transform(y_array)
        else:
            y_encoded = self.label_encoder.fit_transform(y_array)

        return X_array, y_encoded

    def save(self, scaler_path: str, encoder_path: str):
        """Save fitted scaler and encoder to disk."""
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Saved preprocessor artifacts to {scaler_path} and {encoder_path}")
