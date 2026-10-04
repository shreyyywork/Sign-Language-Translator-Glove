"""
Baseline Model Training Pipeline for ISL Sensor Glove.
Trains and compares:
  1. Decision Tree
  2. Random Forest
  3. Support Vector Machine (SVM)
Calculates Accuracy, Precision, Recall, F1 score, and Confusion Matrix.
Saves models to disk and dynamically selects the best validated model.
"""

import os
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from ml.preprocessor import SensorPreprocessor
from ml.dataset_manager import DatasetManager


class BaselineModelTrainer:
    def __init__(
        self,
        models_dir: str = "models/saved",
        eval_dir: str = "models/evaluation"
    ):
        self.models_dir = models_dir
        self.eval_dir = eval_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def train_and_compare(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train Decision Tree, Random Forest, and SVM on train_df and evaluate on test_df.
        Calculates all required metrics and selects the best model based on validation F1 score.
        """
        preprocessor = SensorPreprocessor(feature_names=feature_names)

        # Fit preprocessor on training data
        X_train_raw, y_train_raw = preprocessor.extract_features_and_labels(train_df)
        X_test_raw, y_test_raw = preprocessor.extract_features_and_labels(test_df)

        X_train, y_train = preprocessor.fit_transform(X_train_raw, y_train_raw)
        X_test = preprocessor.transform_features(X_test_raw)
        y_test = preprocessor.label_encoder.transform(y_test_raw)

        # Save preprocessor scaler & encoder
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        encoder_path = os.path.join(self.models_dir, "label_encoder.joblib")
        preprocessor.save(scaler_path, encoder_path)

        classes = list(preprocessor.label_encoder.classes_)

        # Define the candidate baseline models
        candidate_models = {
            "decision_tree": DecisionTreeClassifier(
                criterion="gini",
                max_depth=15,
                min_samples_split=4,
                random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_split=3,
                random_state=42,
                n_jobs=1
            ),
            "svm": SVC(
                C=10.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=42
            )
        }

        results = {}
        trained_models = {}

        for name, clf in candidate_models.items():
            start_t = time.time()
            clf.fit(X_train, y_train)
            train_time = round(time.time() - start_t, 4)

            # Predict on test set
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

            # Calculate metrics
            acc = float(accuracy_score(y_test, y_pred))
            prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred).tolist()
            cm_normalized = confusion_matrix(y_test, y_pred, normalize="true").tolist()

            # Per-class report
            report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True, zero_division=0)

            # Save individual model
            model_path = os.path.join(self.models_dir, f"{name}.joblib")
            joblib.dump(clf, model_path)
            trained_models[name] = clf

            results[name] = {
                "model_name": name,
                "model_path": model_path,
                "training_time_sec": train_time,
                "test_samples": len(y_test),
                "train_samples": len(y_train),
                "accuracy": round(acc, 4),
                "precision_weighted": round(prec_weighted, 4),
                "recall_weighted": round(rec_weighted, 4),
                "f1_weighted": round(f1_weighted, 4),
                "precision_macro": round(prec_macro, 4),
                "recall_macro": round(rec_macro, 4),
                "f1_macro": round(f1_macro, 4),
                "confusion_matrix": cm,
                "confusion_matrix_normalized": [[round(v, 4) for v in row] for row in cm_normalized],
                "classes": classes,
                "per_class_report": report_dict
            }

        # Select best model based on highest validation F1 macro score
        best_model_name = max(results.keys(), key=lambda k: results[k]["f1_weighted"])
        best_model_path = os.path.join(self.models_dir, "best_model.joblib")
        joblib.dump(trained_models[best_model_name], best_model_path)

        summary = {
            "timestamp": time.time(),
            "best_model": best_model_name,
            "best_model_path": best_model_path,
            "scaler_path": scaler_path,
            "encoder_path": encoder_path,
            "models": results
        }

        # Save evaluation summary to JSON
        eval_path = os.path.join(self.eval_dir, "model_comparison.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"Model comparison completed. Best model: {best_model_name} (F1: {results[best_model_name]['f1_weighted']})")
        return summary
