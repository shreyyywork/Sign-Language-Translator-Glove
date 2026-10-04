"""
Dataset Manager for ISL Wearable Sensor Glove.
Handles:
  - CSV storage for both simulated and real sensor data
  - Dataset validation and format integrity checking
  - User-independent train/test splitting (e.g. Train U01-U04, Test U05)
  - Stratified random splitting
  - Dataset health metrics and class distribution summaries
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold


REQUIRED_COLUMNS = [
    "timestamp", "user_id", "gesture",
    "thumb", "index", "middle", "ring", "little",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z"
]


class DatasetManager:
    def __init__(
        self,
        simulated_dir: str = "data/simulated",
        real_dir: str = "data/real"
    ):
        self.simulated_dir = simulated_dir
        self.real_dir = real_dir
        os.makedirs(self.simulated_dir, exist_ok=True)
        os.makedirs(self.real_dir, exist_ok=True)

    def load_dataset(
        self,
        dataset_type: str = "all",  # "simulated", "real", or "all"
        custom_filepath: Optional[str] = None
    ) -> pd.DataFrame:
        """Load and concatenate datasets based on type or custom path."""
        if custom_filepath:
            if not os.path.exists(custom_filepath):
                raise FileNotFoundError(f"Custom dataset file not found: {custom_filepath}")
            df = pd.read_csv(custom_filepath)
            self._validate_dataframe(df)
            return df

        files_to_load = []
        if dataset_type in ["simulated", "all"]:
            files_to_load.extend(glob.glob(os.path.join(self.simulated_dir, "*.csv")))
        if dataset_type in ["real", "all"]:
            files_to_load.extend(glob.glob(os.path.join(self.real_dir, "*.csv")))

        if not files_to_load:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        dfs = []
        for f in files_to_load:
            try:
                sub_df = pd.read_csv(f)
                if all(col in sub_df.columns for col in REQUIRED_COLUMNS):
                    # Flag source if not present
                    if "is_simulated" not in sub_df.columns:
                        sub_df["is_simulated"] = ("simulated" in f.lower())
                    dfs.append(sub_df)
            except Exception as e:
                print(f"Warning: Failed to load dataset file {f}: {e}")

        if not dfs:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def _validate_dataframe(self, df: pd.DataFrame):
        """Validate presence of all 14 required columns."""
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

    def append_recording_session(
        self,
        samples: List[Dict[str, Any]],
        user_id: str,
        gesture: str,
        is_simulated: bool = False,
        custom_filename: Optional[str] = None
    ) -> str:
        """Save a new recording session of sensor samples to CSV."""
        if not samples:
            raise ValueError("No samples provided to save.")

        target_dir = self.simulated_dir if is_simulated else self.real_dir
        if custom_filename:
            target_file = os.path.join(target_dir, custom_filename)
        else:
            prefix = "simulated" if is_simulated else "real"
            target_file = os.path.join(target_dir, f"isl_{prefix}_dataset.csv")

        records = []
        for s in samples:
            rec = {
                "timestamp": s.get("timestamp", 0.0),
                "user_id": user_id,
                "gesture": gesture,
                "thumb": float(s.get("thumb", 0.0)),
                "index": float(s.get("index", 0.0)),
                "middle": float(s.get("middle", 0.0)),
                "ring": float(s.get("ring", 0.0)),
                "little": float(s.get("little", 0.0)),
                "acc_x": float(s.get("acc_x", 0.0)),
                "acc_y": float(s.get("acc_y", 0.0)),
                "acc_z": float(s.get("acc_z", 0.0)),
                "gyro_x": float(s.get("gyro_x", 0.0)),
                "gyro_y": float(s.get("gyro_y", 0.0)),
                "gyro_z": float(s.get("gyro_z", 0.0)),
                "is_simulated": is_simulated
            }
            records.append(rec)

        new_df = pd.DataFrame(records)

        if os.path.exists(target_file):
            existing_df = pd.read_csv(target_file)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined.to_csv(target_file, index=False)
        else:
            new_df.to_csv(target_file, index=False)

        return target_file

    def user_independent_split(
        self,
        df: pd.DataFrame,
        train_users: List[str],
        test_users: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by user ID to test generalizability across unseen users.
        E.g. Train: ['U01', 'U02', 'U03', 'U04'], Test: ['U05']
        """
        self._validate_dataframe(df)
        train_df = df[df["user_id"].isin(train_users)].copy()
        test_df = df[df["user_id"].isin(test_users)].copy()

        if len(train_df) == 0:
            raise ValueError(f"No training data found for users {train_users}")
        if len(test_df) == 0:
            raise ValueError(f"No testing data found for users {test_users}")

        return train_df, test_df

    def stratified_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform standard stratified train/test split on gestures."""
        self._validate_dataframe(df)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["gesture"],
            random_state=random_state
        )
        return train_df, test_df

    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for dashboard and reports."""
        if df.empty:
            return {
                "total_samples": 0,
                "gesture_counts": {},
                "user_counts": {},
                "simulated_count": 0,
                "real_count": 0,
                "users": [],
                "gestures": []
            }

        sim_count = int(df["is_simulated"].sum()) if "is_simulated" in df.columns else 0
        real_count = len(df) - sim_count

        return {
            "total_samples": int(len(df)),
            "gesture_counts": df["gesture"].value_counts().to_dict(),
            "user_counts": df["user_id"].value_counts().to_dict(),
            "simulated_count": sim_count,
            "real_count": real_count,
            "users": sorted(list(df["user_id"].unique())),
            "gestures": sorted(list(df["gesture"].unique()))
        }
