"""
Calibration Service for ISL Wearable Sensor Glove.
Handles 3-step user calibration:
  1. Relaxed hand
  2. Fingers straight
  3. Fingers bent
Stores per-user calibration parameters and provides sensor normalization functions.
"""

import os
import json
from typing import Dict, Any, Optional, List
import numpy as np


class CalibrationService:
    def __init__(self, calibration_dir: str = "data/calibration"):
        self.calibration_dir = calibration_dir
        os.makedirs(self.calibration_dir, exist_ok=True)
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self._load_all_profiles()
        self._ensure_default_profile()

    def _ensure_default_profile(self):
        """Ensure a default calibration baseline exists."""
        default_path = os.path.join(self.calibration_dir, "default_calibration.json")
        if not os.path.exists(default_path):
            default_data = {
                "user_id": "DEFAULT",
                "timestamp": "initial",
                "flex": {
                    "thumb": {"straight": 1200.0, "relaxed": 1800.0, "bent": 3100.0},
                    "index": {"straight": 1150.0, "relaxed": 1750.0, "bent": 3200.0},
                    "middle": {"straight": 1100.0, "relaxed": 1700.0, "bent": 3250.0},
                    "ring": {"straight": 1150.0, "relaxed": 1750.0, "bent": 3150.0},
                    "little": {"straight": 1250.0, "relaxed": 1850.0, "bent": 3050.0}
                },
                "imu": {
                    "acc_offset": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "gyro_offset": {"x": 0.0, "y": 0.0, "z": 0.0}
                }
            }
            self.save_profile("DEFAULT", default_data)

    def _load_all_profiles(self):
        """Load all saved calibration profiles from disk."""
        if not os.path.exists(self.calibration_dir):
            return
        for fname in os.listdir(self.calibration_dir):
            if fname.endswith("_calibration.json") or fname == "default_calibration.json":
                fpath = os.path.join(self.calibration_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        uid = data.get("user_id", fname.replace("_calibration.json", ""))
                        self.user_profiles[uid] = data
                except Exception as e:
                    print(f"Error loading calibration profile {fpath}: {e}")

    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> str:
        """Save calibration data for a specific user."""
        profile_data["user_id"] = user_id
        self.user_profiles[user_id] = profile_data
        filename = f"{user_id}_calibration.json" if user_id != "DEFAULT" else "default_calibration.json"
        fpath = os.path.join(self.calibration_dir, filename)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)
        return fpath

    def get_profile(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get calibration profile for a user, falling back to DEFAULT."""
        if user_id and user_id in self.user_profiles:
            return self.user_profiles[user_id]
        if "DEFAULT" in self.user_profiles:
            return self.user_profiles["DEFAULT"]
        return {}

    def compute_calibration_from_samples(
        self,
        user_id: str,
        relaxed_samples: List[Dict[str, float]],
        straight_samples: List[Dict[str, float]],
        bent_samples: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compute mean calibration limits from 3-step recorded sample batches:
          1. Relaxed hand
          2. Straight fingers
          3. Bent fingers
        """
        fingers = ["thumb", "index", "middle", "ring", "little"]
        flex_calib = {}

        for finger in fingers:
            straight_vals = [s[finger] for s in straight_samples if finger in s]
            relaxed_vals = [s[finger] for s in relaxed_samples if finger in s]
            bent_vals = [s[finger] for s in bent_samples if finger in s]

            s_mean = float(np.mean(straight_vals)) if straight_vals else 1200.0
            r_mean = float(np.mean(relaxed_vals)) if relaxed_vals else 1800.0
            b_mean = float(np.mean(bent_vals)) if bent_vals else 3200.0

            # Safeguard: ensure bent > straight
            if b_mean <= s_mean:
                b_mean = s_mean + 1500.0

            flex_calib[finger] = {
                "straight": round(s_mean, 2),
                "relaxed": round(r_mean, 2),
                "bent": round(b_mean, 2)
            }

        # IMU neutral offset calculation from relaxed state
        acc_x = [s.get("acc_x", 0.0) for s in relaxed_samples]
        acc_y = [s.get("acc_y", 0.0) for s in relaxed_samples]
        acc_z = [s.get("acc_z", 0.0) for s in relaxed_samples]
        gyro_x = [s.get("gyro_x", 0.0) for s in relaxed_samples]
        gyro_y = [s.get("gyro_y", 0.0) for s in relaxed_samples]
        gyro_z = [s.get("gyro_z", 0.0) for s in relaxed_samples]

        profile = {
            "user_id": user_id,
            "flex": flex_calib,
            "imu": {
                "acc_offset": {
                    "x": round(float(np.mean(acc_x)) if acc_x else 0.0, 4),
                    "y": round(float(np.mean(acc_y)) if acc_y else 0.0, 4),
                    "z": round(float(np.mean(acc_z)) if acc_z else 0.0, 4)
                },
                "gyro_offset": {
                    "x": round(float(np.mean(gyro_x)) if gyro_x else 0.0, 4),
                    "y": round(float(np.mean(gyro_y)) if gyro_y else 0.0, 4),
                    "z": round(float(np.mean(gyro_z)) if gyro_z else 0.0, 4)
                }
            }
        }
        self.save_profile(user_id, profile)
        return profile

    def normalize(self, raw_sample: Dict[str, float], user_id: Optional[str] = None) -> Dict[str, float]:
        """
        Normalize 11 raw sensor values into standard ranges:
          - flex fingers: [0.0 (straight), 1.0 (fully bent)]
          - acc_x, acc_y, acc_z: g-force with neutral bias subtraction
          - gyro_x, gyro_y, gyro_z: deg/s or rad/s with zero-rate bias subtraction
        """
        profile = self.get_profile(user_id)
        flex_calib = profile.get("flex", {})
        gyro_offset = profile.get("imu", {}).get("gyro_offset", {})

        normalized = {}
        fingers = ["thumb", "index", "middle", "ring", "little"]

        for finger in fingers:
            raw_val = raw_sample.get(finger, 0.0)
            if finger in flex_calib:
                s = flex_calib[finger].get("straight", 1200.0)
                b = flex_calib[finger].get("bent", 3200.0)
                span = b - s if (b - s) != 0 else 1.0
                norm_val = (raw_val - s) / span
                normalized[finger] = round(float(np.clip(norm_val, 0.0, 1.0)), 4)
            else:
                # If already in [0, 1] range (e.g. simulator)
                if 0.0 <= raw_val <= 1.0:
                    normalized[finger] = round(raw_val, 4)
                else:
                    normalized[finger] = round(float(np.clip(raw_val / 4095.0, 0.0, 1.0)), 4)

        # Accelerometer features
        normalized["acc_x"] = round(float(raw_sample.get("acc_x", 0.0)), 4)
        normalized["acc_y"] = round(float(raw_sample.get("acc_y", 0.0)), 4)
        normalized["acc_z"] = round(float(raw_sample.get("acc_z", 0.0)), 4)

        # Gyroscope features (subtract bias)
        gx_bias = gyro_offset.get("x", 0.0)
        gy_bias = gyro_offset.get("y", 0.0)
        gz_bias = gyro_offset.get("z", 0.0)

        normalized["gyro_x"] = round(float(raw_sample.get("gyro_x", 0.0) - gx_bias), 4)
        normalized["gyro_y"] = round(float(raw_sample.get("gyro_y", 0.0) - gy_bias), 4)
        normalized["gyro_z"] = round(float(raw_sample.get("gyro_z", 0.0) - gz_bias), 4)

        return normalized
