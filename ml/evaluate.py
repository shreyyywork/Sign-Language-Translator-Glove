"""
Model Evaluation and User-Independent Validation for ISL Sensor Glove.
Provides:
  - User-independent cross-validation (Leave-One-User-Out or custom Train/Test users)
  - Detailed performance metrics comparison
  - Confusion matrix calculations and metrics export
"""

import os
import json
import time
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ml.dataset_manager import DatasetManager
from ml.preprocessor import SensorPreprocessor
from ml.train_baseline import BaselineModelTrainer


class ModelEvaluator:
    def __init__(
        self,
        models_dir: str = "models/saved",
        eval_dir: str = "models/evaluation"
    ):
        self.models_dir = models_dir
        self.eval_dir = eval_dir
        self.trainer = BaselineModelTrainer(models_dir=models_dir, eval_dir=eval_dir)
        self.dataset_manager = DatasetManager()

    def run_user_independent_evaluation(
        self,
        df: pd.DataFrame,
        train_users: List[str],
        test_users: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate models when trained exclusively on train_users (e.g. U01, U02, U03, U04)
        and tested exclusively on unseen test_users (e.g. U05).
        """
        train_df, test_df = self.dataset_manager.user_independent_split(
            df,
            train_users=train_users,
            test_users=test_users
        )

        print(f"Running user-independent evaluation: Train on {train_users} ({len(train_df)} samples), Test on {test_users} ({len(test_df)} samples)")
        summary = self.trainer.train_and_compare(train_df, test_df)

        summary["evaluation_mode"] = "user_independent"
        summary["train_users"] = train_users
        summary["test_users"] = test_users

        eval_path = os.path.join(self.eval_dir, "user_independent_eval.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def run_stratified_evaluation(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Evaluate models using stratified train/test split.
        """
        train_df, test_df = self.dataset_manager.stratified_split(df, test_size=test_size)
        print(f"Running stratified evaluation: Train={len(train_df)} samples, Test={len(test_df)} samples")
        summary = self.trainer.train_and_compare(train_df, test_df)

        summary["evaluation_mode"] = "stratified"
        summary["test_size"] = test_size

        eval_path = os.path.join(self.eval_dir, "stratified_eval.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def get_latest_evaluation_report(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent evaluation comparison report."""
        eval_path = os.path.join(self.eval_dir, "model_comparison.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
