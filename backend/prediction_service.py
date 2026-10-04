"""
Real-Time Prediction Engine for ISL Wearable Sensor Glove.
Processes 11 numerical features:
  - Flex: thumb, index, middle, ring, little
  - IMU: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
Performs normalization, scaling, inference, confidence probability estimation,
and temporal debouncing/smoothing.
"""

import os
import time
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from ml.model_selector import ModelSelector
from backend.calibration_service import CalibrationService


class RealTimePredictionService:
    def __init__(
        self,
        models_dir: str = "models/saved",
        eval_dir: str = "models/evaluation",
        smoothing_window_size: int = 5,
        confidence_threshold: float = 0.60
    ):
        self.models_dir = models_dir
        self.eval_dir = eval_dir
        self.smoothing_window_size = smoothing_window_size
        self.confidence_threshold = confidence_threshold

        self.model_selector = ModelSelector(models_dir=models_dir, eval_dir=eval_dir)
        self.calibration_service = CalibrationService()
        self.prediction_history: deque = deque(maxlen=smoothing_window_size)
        self.feature_names = [
            "thumb", "index", "middle", "ring", "little",
            "acc_x", "acc_y", "acc_z",
            "gyro_x", "gyro_y", "gyro_z"
        ]

    def reload_model(self, preferred_model: Optional[str] = None) -> bool:
        """Reload the active machine learning model from disk."""
        return self.model_selector.load_best_or_default(preferred_model)

    def predict(
        self,
        raw_sensor_dict: Dict[str, float],
        user_id: Optional[str] = None,
        apply_calibration: bool = True
    ) -> Dict[str, Any]:
        """
        Accepts 11 sensor features and performs real-time gesture inference.
        Returns:
          predicted gesture, probability confidence, timestamp, top_k alternatives, stability status.
        """
        if not self.model_selector.active_model:
            # Try loading
            if not self.reload_model():
                return {
                    "gesture": "MODEL_NOT_TRAINED",
                    "confidence": 0.0,
                    "timestamp": round(time.time(), 3),
                    "error": "No trained model found. Please train a model first.",
                    "is_stable": False
                }

        # Step 1: Calibration Normalization
        if apply_calibration:
            norm_features = self.calibration_service.normalize(raw_sensor_dict, user_id=user_id)
        else:
            norm_features = raw_sensor_dict

        # Step 2: Extract ordered feature vector
        feat_vals = [float(norm_features.get(f, 0.0)) for f in self.feature_names]
        feat_array = np.array([feat_vals], dtype=np.float32)

        # Step 3: Standardize features using preprocessor scaler
        if self.model_selector.scaler is not None:
            X_scaled = self.model_selector.scaler.transform(feat_array)
        else:
            X_scaled = feat_array

        # Step 4: ML Model Inference
        clf = self.model_selector.active_model
        top_k = []
        confidence = 0.0
        predicted_gesture = "UNKNOWN"

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_scaled)[0]
            classes = self.model_selector.label_encoder.classes_

            # Sort probabilities descending
            top_indices = np.argsort(probs)[::-1]
            top_idx = int(top_indices[0])
            confidence = float(probs[top_idx])
            predicted_gesture = str(classes[top_idx])

            for idx in top_indices[:3]:
                top_k.append({
                    "gesture": str(classes[idx]),
                    "probability": round(float(probs[idx]), 4)
                })
        else:
            pred_idx = clf.predict(X_scaled)[0]
            if self.model_selector.label_encoder:
                predicted_gesture = str(self.model_selector.label_encoder.inverse_transform([pred_idx])[0])
            else:
                predicted_gesture = str(pred_idx)
            confidence = 0.85  # Default confidence for models without predict_proba

        # Step 5: Temporal smoothing / debouncing
        self.prediction_history.append(predicted_gesture)
        
        # Check stability across recent window
        recent_preds = list(self.prediction_history)
        most_common_pred = max(set(recent_preds), key=recent_preds.count)
        stability_count = recent_preds.count(most_common_pred)
        is_stable = (stability_count >= (self.smoothing_window_size // 2 + 1))

        # Check confidence threshold
        effective_gesture = predicted_gesture if confidence >= self.confidence_threshold else "UNCERTAIN"

        return {
            "gesture": effective_gesture,
            "raw_gesture": predicted_gesture,
            "confidence": round(confidence, 4),
            "timestamp": round(time.time(), 3),
            "model_name": self.model_selector.active_model_name,
            "top_k": top_k,
            "is_stable": is_stable,
            "calibrated_features": norm_features,
            "raw_features": raw_sensor_dict
        }
