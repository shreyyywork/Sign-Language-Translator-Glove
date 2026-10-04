"""
ISL Glove Sensor Simulator.
Provides realistic biomechanical simulation for Indian Sign Language gestures:
  - 5 Flex sensors (thumb, index, middle, ring, little) with anatomical noise & natural tremor
  - MPU6050 6-DOF IMU (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) with gravity vectors & rotational velocities
  - Dynamic trajectory modeling for moving gestures (nodding, waving, shaking)
  - User anatomical variability (simulating different users U01, U02, etc.)
  - Explicit 'is_simulated' metadata flagging.
"""

import os
import json
import time
import math
import random
from typing import Dict, Any, List, Optional, Generator
import numpy as np


class GloveSensorSimulator:
    def __init__(self, gestures_config_path: str = "config/gestures.json"):
        self.gestures_config_path = gestures_config_path
        self.gestures: Dict[str, Dict[str, Any]] = {}
        self.gesture_list: List[Dict[str, Any]] = []
        self._load_gestures()
        self._user_biometrics: Dict[str, Dict[str, float]] = {}
        self._time_offset = 0.0

    def _load_gestures(self):
        """Load the gesture vocabulary definition."""
        if os.path.exists(self.gestures_config_path):
            with open(self.gestures_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.gesture_list = data.get("gestures", [])
                self.gestures = {g["id"]: g for g in self.gesture_list}
        else:
            raise FileNotFoundError(f"Gestures config not found at {self.gestures_config_path}")

    def _get_user_bias(self, user_id: str) -> Dict[str, float]:
        """Generate consistent anatomical bias per user (e.g. finger flexibility, hand size)."""
        if user_id not in self._user_biometrics:
            # Deterministic seed based on user_id string
            seed = sum(ord(c) for c in user_id)
            rng = np.random.RandomState(seed)
            self._user_biometrics[user_id] = {
                "thumb_bias": float(rng.normal(0.0, 0.03)),
                "index_bias": float(rng.normal(0.0, 0.03)),
                "middle_bias": float(rng.normal(0.0, 0.03)),
                "ring_bias": float(rng.normal(0.0, 0.03)),
                "little_bias": float(rng.normal(0.0, 0.03)),
                "acc_bias_x": float(rng.normal(0.0, 0.04)),
                "acc_bias_y": float(rng.normal(0.0, 0.04)),
                "acc_bias_z": float(rng.normal(0.0, 0.04)),
                "tremor_scale": float(rng.uniform(0.8, 1.3))
            }
        return self._user_biometrics[user_id]

    def generate_sample(
        self,
        gesture_id: str,
        user_id: str = "U01",
        step_index: int = 0,
        frequency_hz: float = 50.0,
        raw_adc: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a single 11-feature sensor sample for the specified gesture.
        Features returned:
          - thumb, index, middle, ring, little (0.0 to 1.0 or raw ADC 0-4095)
          - acc_x, acc_y, acc_z (in g, e.g. -2.0 to 2.0)
          - gyro_x, gyro_y, gyro_z (in rad/s or deg/s scaled)
          - timestamp, user_id, gesture, is_simulated
        """
        if gesture_id not in self.gestures:
            raise ValueError(f"Unknown gesture ID: {gesture_id}. Available: {list(self.gestures.keys())}")

        gesture_info = self.gestures[gesture_id]
        finger_base = gesture_info.get("finger_profile", {})
        imu_base = gesture_info.get("imu_profile", {})
        is_dynamic = gesture_info.get("is_dynamic", False)

        user_bias = self._get_user_bias(user_id)
        t = step_index / frequency_hz + self._time_offset
        tremor_factor = user_bias["tremor_scale"]

        # Finger calculations with tremor & slight Gaussian noise
        fingers = {}
        for f_name in ["thumb", "index", "middle", "ring", "little"]:
            base_val = finger_base.get(f_name, 0.0)
            bias = user_bias.get(f"{f_name}_bias", 0.0)
            tremor = np.sin(2 * np.pi * 8.0 * t + random.random()) * 0.008 * tremor_factor
            noise = np.random.normal(0.0, 0.015)
            val = np.clip(base_val + bias + tremor + noise, 0.0, 1.0)
            
            if raw_adc:
                # Map 0.0-1.0 to realistic raw ESP32 12-bit ADC value (e.g. 1100 straight, 3200 bent)
                adc_val = 1200.0 + val * 2000.0 + np.random.normal(0.0, 10.0)
                fingers[f_name] = round(float(adc_val), 1)
            else:
                fingers[f_name] = round(float(val), 4)

        # IMU calculations (accel + gyro) with dynamic oscillation if gesture is dynamic
        dyn_acc_x = 0.0
        dyn_acc_y = 0.0
        dyn_acc_z = 0.0
        dyn_gyro_x = 0.0
        dyn_gyro_y = 0.0
        dyn_gyro_z = 0.0

        if is_dynamic:
            # Frequency of physical movement ~ 1.5 - 2.5 Hz
            freq = 2.0
            phase = 2 * np.pi * freq * t

            if gesture_id == "HELLO":
                # Waving hand side-to-side (roll oscillation)
                dyn_acc_x = 0.3 * np.sin(phase)
                dyn_gyro_y = 0.8 * np.cos(phase)
            elif gesture_id == "YES":
                # Pitch nodding oscillation
                dyn_acc_z = 0.35 * np.sin(phase)
                dyn_gyro_x = 1.0 * np.cos(phase)
            elif gesture_id == "NO":
                # Horizontal shaking
                dyn_acc_x = 0.4 * np.sin(phase)
                dyn_gyro_z = 0.9 * np.cos(phase)
            elif gesture_id == "EMERGENCY":
                # Fast violent shaking
                dyn_acc_x = 0.7 * np.sin(3.5 * phase)
                dyn_gyro_x = 1.8 * np.sin(3.5 * phase)
                dyn_gyro_y = 1.5 * np.cos(3.5 * phase)
            elif gesture_id in ["EAT", "DRINK"]:
                # Movement towards mouth and back
                dyn_acc_y = 0.25 * np.sin(1.2 * phase)
                dyn_acc_z = 0.3 * np.cos(1.2 * phase)
                dyn_gyro_x = 0.5 * np.sin(1.2 * phase)
            elif gesture_id == "BATHROOM":
                # Quick twisting wrist
                dyn_gyro_y = 1.1 * np.sin(3.0 * phase)
            elif gesture_id == "THANK_YOU":
                # Forward stroke
                dyn_acc_z = 0.4 * np.sin(1.5 * phase)
                dyn_gyro_x = 0.6 * np.cos(1.5 * phase)

        # Base IMU + dynamic component + user bias + noise
        acc_x = imu_base.get("acc_x", 0.0) + user_bias["acc_bias_x"] + dyn_acc_x + np.random.normal(0.0, 0.02)
        acc_y = imu_base.get("acc_y", 0.98) + user_bias["acc_bias_y"] + dyn_acc_y + np.random.normal(0.0, 0.02)
        acc_z = imu_base.get("acc_z", 0.0) + user_bias["acc_bias_z"] + dyn_acc_z + np.random.normal(0.0, 0.02)

        gyro_x = imu_base.get("gyro_x", 0.0) + dyn_gyro_x + np.random.normal(0.0, 0.03)
        gyro_y = imu_base.get("gyro_y", 0.0) + dyn_gyro_y + np.random.normal(0.0, 0.03)
        gyro_z = imu_base.get("gyro_z", 0.0) + dyn_gyro_z + np.random.normal(0.0, 0.03)

        current_timestamp = round(time.time(), 3)

        return {
            "timestamp": current_timestamp,
            "user_id": user_id,
            "gesture": gesture_id,
            "thumb": fingers["thumb"],
            "index": fingers["index"],
            "middle": fingers["middle"],
            "ring": fingers["ring"],
            "little": fingers["little"],
            "acc_x": round(float(acc_x), 4),
            "acc_y": round(float(acc_y), 4),
            "acc_z": round(float(acc_z), 4),
            "gyro_x": round(float(gyro_x), 4),
            "gyro_y": round(float(gyro_y), 4),
            "gyro_z": round(float(gyro_z), 4),
            "is_simulated": True
        }

    def generate_sequence(
        self,
        gesture_id: str,
        user_id: str = "U01",
        num_samples: int = 50,
        frequency_hz: float = 50.0,
        raw_adc: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate a sequential time-series recording of a gesture."""
        samples = []
        for i in range(num_samples):
            sample = self.generate_sample(
                gesture_id=gesture_id,
                user_id=user_id,
                step_index=i,
                frequency_hz=frequency_hz,
                raw_adc=raw_adc
            )
            samples.append(sample)
        return samples

    def generate_full_dataset(
        self,
        users: List[str] = ["U01", "U02", "U03", "U04", "U05"],
        samples_per_gesture: int = 100,
        output_csv_path: str = "data/simulated/isl_simulated_dataset.csv"
    ) -> str:
        """
        Generate a multi-user simulated dataset across all 20 gestures.
        Saves cleanly to CSV format for ML model training and validation.
        """
        import pandas as pd
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        all_records = []

        gesture_ids = list(self.gestures.keys())
        for user_id in users:
            for gesture_id in gesture_ids:
                seq = self.generate_sequence(
                    gesture_id=gesture_id,
                    user_id=user_id,
                    num_samples=samples_per_gesture,
                    frequency_hz=50.0,
                    raw_adc=False
                )
                for s in seq:
                    all_records.append({
                        "timestamp": s["timestamp"],
                        "user_id": s["user_id"],
                        "gesture": s["gesture"],
                        "thumb": s["thumb"],
                        "index": s["index"],
                        "middle": s["middle"],
                        "ring": s["ring"],
                        "little": s["little"],
                        "acc_x": s["acc_x"],
                        "acc_y": s["acc_y"],
                        "acc_z": s["acc_z"],
                        "gyro_x": s["gyro_x"],
                        "gyro_y": s["gyro_y"],
                        "gyro_z": s["gyro_z"],
                        "is_simulated": True
                    })

        df = pd.DataFrame(all_records)
        df.to_csv(output_csv_path, index=False)
        print(f"Generated simulated dataset with {len(df)} samples across {len(users)} users at {output_csv_path}")
        return output_csv_path
