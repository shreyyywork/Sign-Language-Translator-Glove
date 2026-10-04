"""
Unit tests for Calibration Service.
"""

from backend.calibration_service import CalibrationService


def test_calibration_service_default(tmp_path):
    calib = CalibrationService(calibration_dir=str(tmp_path))
    profile = calib.get_profile("DEFAULT")
    assert "flex" in profile
    assert "thumb" in profile["flex"]
    assert profile["flex"]["thumb"]["straight"] == 1200.0
    assert profile["flex"]["thumb"]["bent"] == 3100.0


def test_compute_calibration_from_samples(tmp_path):
    calib = CalibrationService(calibration_dir=str(tmp_path))

    relaxed = [{"thumb": 1800, "index": 1750, "middle": 1700, "ring": 1750, "little": 1850, "acc_x": 0.0, "acc_y": 0.98, "acc_z": 0.0, "gyro_x": 0.02, "gyro_y": 0.01, "gyro_z": -0.01}]
    straight = [{"thumb": 1100, "index": 1050, "middle": 1000, "ring": 1050, "little": 1150}]
    bent = [{"thumb": 3300, "index": 3350, "middle": 3400, "ring": 3350, "little": 3250}]

    profile = calib.compute_calibration_from_samples("U01", relaxed, straight, bent)
    assert profile["user_id"] == "U01"
    assert profile["flex"]["thumb"]["straight"] == 1100.0
    assert profile["flex"]["thumb"]["bent"] == 3300.0

    # Test normalization
    raw_sample = {"thumb": 1100, "index": 3350, "middle": 2200, "ring": 1050, "little": 3250, "acc_x": 0, "acc_y": 1, "acc_z": 0, "gyro_x": 0.02, "gyro_y": 0.01, "gyro_z": -0.01}
    norm = calib.normalize(raw_sample, "U01")

    assert norm["thumb"] == 0.0    # Straight
    assert norm["index"] == 1.0    # Bent
    assert 0.4 <= norm["middle"] <= 0.6  # Mid bend
    assert norm["gyro_x"] == 0.0   # Bias removed
