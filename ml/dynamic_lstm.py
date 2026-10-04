"""
Dynamic Gesture Recognition Pipeline using Time-Series LSTM.
Advanced feature: Captures continuous hand motions and trajectory dynamics over time windows (e.g. 30 frames = 0.6s at 50Hz).
Supports training, checkpointing, and real-time sliding window inference.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.preprocessor import SensorPreprocessor


class DynamicLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 20,
        dropout: float = 0.2
    ):
        super(DynamicLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        # Take output of last time step
        last_step_out = out[:, -1, :]
        dense = self.relu(self.fc1(last_step_out))
        dense = self.dropout(dense)
        logits = self.fc2(dense)
        return logits


class DynamicLSTMPipeline:
    def __init__(
        self,
        models_dir: str = "models/saved",
        eval_dir: str = "models/evaluation",
        sequence_length: int = 30
    ):
        self.models_dir = models_dir
        self.eval_dir = eval_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.sequence_length = sequence_length
        self.model: Optional[DynamicLSTMClassifier] = None
        self.label_encoder = None
        self.preprocessor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_lstm(
        self,
        df: pd.DataFrame,
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 0.003,
        step_size: int = 5
    ) -> Dict[str, Any]:
        """Train the LSTM model on time-series sequences extracted from df."""
        preprocessor = SensorPreprocessor()
        X_seq, y_seq = preprocessor.create_sliding_sequences(
            df,
            sequence_length=self.sequence_length,
            step_size=step_size
        )

        num_classes = len(np.unique(y_seq))
        num_features = X_seq.shape[2]

        # Convert to PyTorch Tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        y_tensor = torch.tensor(y_seq, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 80/20 train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.model = DynamicLSTMClassifier(
            input_size=num_features,
            hidden_size=64,
            num_layers=2,
            num_classes=num_classes
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        history = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_acc = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss_acc += loss.item() * batch_x.size(0)

            train_loss = train_loss_acc / train_size

            # Validation
            self.model.eval()
            val_loss_acc = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss_acc += loss.item() * batch_x.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            val_loss = val_loss_acc / val_size if val_size > 0 else 0.0
            val_acc = correct / total if total > 0 else 0.0

            history["epochs"].append(epoch)
            history["train_loss"].append(round(train_loss, 4))
            history["val_loss"].append(round(val_loss, 4))
            history["val_accuracy"].append(round(val_acc, 4))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save model checkpoint
                lstm_model_path = os.path.join(self.models_dir, "lstm_model.pth")
                torch.save(self.model.state_dict(), lstm_model_path)

        # Save metadata
        metadata = {
            "model_type": "lstm",
            "input_size": num_features,
            "sequence_length": self.sequence_length,
            "num_classes": num_classes,
            "classes": list(preprocessor.label_encoder.classes_),
            "best_val_accuracy": round(best_val_acc, 4),
            "history": history
        }

        with open(os.path.join(self.eval_dir, "lstm_training_report.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.label_encoder = preprocessor.label_encoder
        self.preprocessor = preprocessor
        return metadata

    def load_model(self) -> bool:
        """Load trained LSTM weights and configuration."""
        meta_path = os.path.join(self.eval_dir, "lstm_training_report.json")
        weights_path = os.path.join(self.models_dir, "lstm_model.pth")

        if os.path.exists(meta_path) and os.path.exists(weights_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.sequence_length = meta["sequence_length"]
            self.model = DynamicLSTMClassifier(
                input_size=meta["input_size"],
                hidden_size=64,
                num_layers=2,
                num_classes=meta["num_classes"]
            ).to(self.device)
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.model.eval()

            # Load label encoder
            enc_path = os.path.join(self.models_dir, "label_encoder.joblib")
            if os.path.exists(enc_path):
                self.label_encoder = joblib.load(enc_path)
            return True
        return False

    def predict_sequence(self, sequence_array: np.ndarray) -> Dict[str, Any]:
        """
        Predict gesture from a sequence window of shape (sequence_length, 11).
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("LSTM model not loaded or trained.")

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(sequence_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])

            if self.label_encoder:
                predicted_gesture = str(self.label_encoder.inverse_transform([top_idx])[0])
            else:
                predicted_gesture = f"CLASS_{top_idx}"

            return {
                "gesture": predicted_gesture,
                "confidence": round(confidence, 4),
                "timestamp": round(time.time(), 3),
                "model_type": "lstm_dynamic"
            }
