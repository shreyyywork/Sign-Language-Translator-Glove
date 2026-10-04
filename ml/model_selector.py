"""
Model Selector for ISL Recognition.
Enables dynamic selection of active model based on empirical validation metrics
or explicit user override. Does not assume Random Forest is best by default.
"""

import os
import json
import joblib
from typing import Dict, Any, Optional, Tuple


class ModelSelector:
    def __init__(
        self,
        models_dir: str = "models/saved",
        eval_dir: str = "models/evaluation"
    ):
        self.models_dir = models_dir
        self.eval_dir = eval_dir
        self.active_model_name: Optional[str] = None
        self.active_model: Any = None
        self.scaler: Any = None
        self.label_encoder: Any = None
        self.load_best_or_default()

    def load_best_or_default(self, preferred_model: Optional[str] = None) -> bool:
        """
        Load the best validated model according to model_comparison.json,
        or load preferred_model if specified.
        """
        eval_path = os.path.join(self.eval_dir, "model_comparison.json")
        model_to_load = preferred_model

        if not model_to_load and os.path.exists(eval_path):
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                    model_to_load = eval_data.get("best_model", "random_forest")
            except Exception as e:
                print(f"Error reading model comparison: {e}")
                model_to_load = "random_forest"

        if not model_to_load:
            model_to_load = "random_forest"

        model_file = os.path.join(self.models_dir, f"{model_to_load}.joblib")
        if not os.path.exists(model_file):
            # Fallback to best_model.joblib
            model_file = os.path.join(self.models_dir, "best_model.joblib")

        scaler_file = os.path.join(self.models_dir, "scaler.joblib")
        encoder_file = os.path.join(self.models_dir, "label_encoder.joblib")

        if os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(encoder_file):
            try:
                self.active_model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                self.label_encoder = joblib.load(encoder_file)
                self.active_model_name = model_to_load
                print(f"Active model successfully loaded: {model_to_load} from {model_file}")
                return True
            except Exception as e:
                print(f"Error loading model artifacts: {e}")
                return False
        return False

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of active model and available trained models."""
        available = []
        if os.path.exists(self.models_dir):
            for f in os.listdir(self.models_dir):
                if f.endswith(".joblib") and not f.startswith("scaler") and not f.startswith("label_encoder"):
                    name = f.replace(".joblib", "")
                    available.append(name)

        return {
            "active_model": self.active_model_name,
            "is_loaded": self.active_model is not None,
            "available_models": available,
            "classes": list(self.label_encoder.classes_) if self.label_encoder else []
        }
