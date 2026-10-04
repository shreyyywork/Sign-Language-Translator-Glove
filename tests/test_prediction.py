"""
Unit tests for Real-Time Prediction Engine.
"""

import os
import pandas as pd
from backend.simulator import GloveSensorSimulator
from ml.train_baseline import BaselineModelTrainer
from backend.prediction_service import RealTimePredictionService


def test_real_time_prediction(tmp_path):
    models_dir = os.path.join(tmp_path, "models")
    eval_dir = os.path.join(tmp_path, "eval")

    sim = GloveSensorSimulator("config/gestures.json")
    dataset_csv = os.path.join(tmp_path, "sim_dataset.csv")
    sim.generate_full_dataset(
        users=["U01", "U02"],
        samples_per_gesture=20,
        output_csv_path=dataset_csv
    )
    df = pd.read_csv(dataset_csv)

    trainer = BaselineModelTrainer(models_dir=models_dir, eval_dir=eval_dir)
    trainer.train_and_compare(df, df)

    pred_service = RealTimePredictionService(models_dir=models_dir, eval_dir=eval_dir)
    pred_service.reload_model()

    # Generate a HELLO sample
    hello_sample = sim.generate_sample("HELLO", user_id="U01")
    result = pred_service.predict(hello_sample, user_id="U01")

    assert "gesture" in result
    assert "confidence" in result
    assert "timestamp" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert "top_k" in result
    assert len(result["top_k"]) > 0
