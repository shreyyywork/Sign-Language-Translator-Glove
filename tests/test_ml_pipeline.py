"""
Unit tests for Machine Learning Baseline Training and Evaluation Pipeline.
"""

import os
import pandas as pd
from backend.simulator import GloveSensorSimulator
from ml.dataset_manager import DatasetManager
from ml.train_baseline import BaselineModelTrainer
from ml.evaluate import ModelEvaluator
from ml.model_selector import ModelSelector


def test_baseline_training_and_comparison(tmp_path):
    models_dir = os.path.join(tmp_path, "models")
    eval_dir = os.path.join(tmp_path, "eval")

    sim = GloveSensorSimulator("config/gestures.json")
    train_csv = os.path.join(tmp_path, "train.csv")
    sim.generate_full_dataset(
        users=["U01", "U02", "U03", "U04"],
        samples_per_gesture=15,
        output_csv_path=train_csv
    )

    test_csv = os.path.join(tmp_path, "test.csv")
    sim.generate_full_dataset(
        users=["U05"],
        samples_per_gesture=15,
        output_csv_path=test_csv
    )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    trainer = BaselineModelTrainer(models_dir=models_dir, eval_dir=eval_dir)
    summary = trainer.train_and_compare(train_df, test_df)

    assert "best_model" in summary
    assert summary["best_model"] in ["decision_tree", "random_forest", "svm"]
    assert "decision_tree" in summary["models"]
    assert "random_forest" in summary["models"]
    assert "svm" in summary["models"]

    # Verify metric keys
    rf_metrics = summary["models"]["random_forest"]
    assert "accuracy" in rf_metrics
    assert "f1_weighted" in rf_metrics
    assert "precision_weighted" in rf_metrics
    assert "recall_weighted" in rf_metrics
    assert "confusion_matrix" in rf_metrics
    assert len(rf_metrics["classes"]) == 20

    # Test Model Selector
    selector = ModelSelector(models_dir=models_dir, eval_dir=eval_dir)
    assert selector.active_model is not None
    assert selector.active_model_name == summary["best_model"]
