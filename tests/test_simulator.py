"""
Unit tests for Sensor Data Simulator.
"""

import pytest
import os
from backend.simulator import GloveSensorSimulator


def test_simulator_initialization():
    sim = GloveSensorSimulator("config/gestures.json")
    assert len(sim.gesture_list) == 20
    assert "HELLO" in sim.gestures
    assert "THANK_YOU" in sim.gestures


def test_single_sample_generation():
    sim = GloveSensorSimulator("config/gestures.json")
    sample = sim.generate_sample("HELLO", user_id="U01", step_index=0)

    # Verify 11 numerical features + metadata
    required_features = [
        "thumb", "index", "middle", "ring", "little",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    ]
    for feat in required_features:
        assert feat in sample
        assert isinstance(sample[feat], (int, float))

    assert sample["gesture"] == "HELLO"
    assert sample["user_id"] == "U01"
    assert sample["is_simulated"] is True


def test_finger_bending_bounds():
    sim = GloveSensorSimulator("config/gestures.json")
    # Test flat open hand vs tight fist
    hello_sample = sim.generate_sample("HELLO", user_id="U01", step_index=0)
    yes_sample = sim.generate_sample("YES", user_id="U01", step_index=0)

    # In YES (fist), fingers should be significantly more bent than in HELLO
    assert yes_sample["index"] > hello_sample["index"]
    assert yes_sample["middle"] > hello_sample["middle"]


def test_dataset_generation(tmp_path):
    sim = GloveSensorSimulator("config/gestures.json")
    test_csv = os.path.join(tmp_path, "test_dataset.csv")
    csv_path = sim.generate_full_dataset(
        users=["U01", "U02"],
        samples_per_gesture=10,
        output_csv_path=test_csv
    )
    assert os.path.exists(csv_path)
    
    import pandas as pd
    df = pd.read_csv(csv_path)
    assert len(df) == 2 * 20 * 10  # 400 rows
    assert df["gesture"].nunique() == 20
