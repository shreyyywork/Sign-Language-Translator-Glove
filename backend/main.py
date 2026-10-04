"""
FastAPI Main Application Server for ISL Wearable Sensor Glove.
Provides:
  - REST APIs for Data Collection, Calibration, ML Training, Evaluation, and Prediction
  - WebSocket for 50Hz real-time sensor streaming, 3D hand visualizer data, and live sign prediction
  - Static file hosting for the modern web dashboard
  - Seamless switching between Hardware Glove and Biomechanical Simulator
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ml.dataset_manager import DatasetManager
from ml.train_baseline import BaselineModelTrainer
from ml.evaluate import ModelEvaluator
from ml.model_selector import ModelSelector
from ml.dynamic_lstm import DynamicLSTMPipeline
from ml.tinyml_exporter import TinyMLExporter
from backend.simulator import GloveSensorSimulator
from backend.calibration_service import CalibrationService
from backend.prediction_service import RealTimePredictionService
from backend.serial_bluetooth import SerialBluetoothManager
from backend.tts_service import TTSService


# Initialize Services
simulator = GloveSensorSimulator()
calibration_service = CalibrationService()
prediction_service = RealTimePredictionService()
dataset_manager = DatasetManager()
evaluator = ModelEvaluator()
tts_service = TTSService()
serial_manager = SerialBluetoothManager()
lstm_pipeline = DynamicLSTMPipeline()

# Global State
system_state = {
    "mode": "simulated",          # "simulated" or "hardware"
    "active_sim_gesture": "HELLO",
    "active_sim_user": "U01",
    "is_recording": False,
    "recording_user_id": "U01",
    "recording_gesture": "HELLO",
    "recording_buffer": [],
    "is_streaming": True
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try loading existing trained model on startup
    prediction_service.reload_model()
    yield
    # Cleanup on shutdown
    serial_manager.disconnect()


app = FastAPI(
    title="ISL Wearable Sensor Glove Recognition System",
    description="Machine Learning Based Indian Sign Language Recognition with Wearable Sensor Glove",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Request Models
# ==========================================
class ModeRequest(BaseModel):
    mode: str = Field(..., description="'simulated' or 'hardware'")

class SimGestureRequest(BaseModel):
    gesture: str
    user_id: Optional[str] = "U01"

class RecordSessionRequest(BaseModel):
    user_id: str
    gesture: str
    samples: List[Dict[str, Any]]
    is_simulated: bool = False

class TrainRequest(BaseModel):
    evaluation_mode: str = "user_independent"  # "user_independent" or "stratified"
    train_users: Optional[List[str]] = ["U01", "U02", "U03", "U04"]
    test_users: Optional[List[str]] = ["U05"]
    test_size: Optional[float] = 0.2
    dataset_type: Optional[str] = "all"  # "all", "simulated", "real"

class CalibrationComputeRequest(BaseModel):
    user_id: str
    relaxed_samples: List[Dict[str, float]]
    straight_samples: List[Dict[str, float]]
    bent_samples: List[Dict[str, float]]

class PredictRequest(BaseModel):
    features: Dict[str, float]
    user_id: Optional[str] = "U01"
    apply_calibration: bool = True

class ModelSelectRequest(BaseModel):
    model_name: str


# ==========================================
# REST API Endpoints
# ==========================================

@app.get("/api/health")
async def get_health():
    """System status and health metrics."""
    return {
        "status": "online",
        "timestamp": time.time(),
        "mode": system_state["mode"],
        "active_sim_gesture": system_state["active_sim_gesture"],
        "active_sim_user": system_state["active_sim_user"],
        "hardware_connected": serial_manager.is_connected,
        "hardware_status": serial_manager.get_status(),
        "model_status": prediction_service.model_selector.get_model_status()
    }

@app.get("/api/gestures")
async def get_gestures():
    """Get the list of 20 ISL vocabulary gestures."""
    return {
        "count": len(simulator.gesture_list),
        "gestures": simulator.gesture_list
    }

@app.post("/api/mode")
async def set_mode(req: ModeRequest):
    """Switch between simulated sensor mode and real hardware glove mode."""
    if req.mode not in ["simulated", "hardware"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'simulated' or 'hardware'.")
    system_state["mode"] = req.mode
    return {"message": f"Mode switched to {req.mode}", "mode": system_state["mode"]}

@app.post("/api/simulator/set_gesture")
async def set_simulator_gesture(req: SimGestureRequest):
    """Set the active gesture generated by the biomechanical simulator."""
    if req.gesture not in simulator.gestures:
        raise HTTPException(status_code=400, detail=f"Unknown gesture: {req.gesture}")
    system_state["active_sim_gesture"] = req.gesture
    if req.user_id:
        system_state["active_sim_user"] = req.user_id
    return {
        "message": f"Simulator set to gesture {req.gesture} for user {system_state['active_sim_user']}",
        "gesture": req.gesture,
        "user_id": system_state["active_sim_user"]
    }

@app.post("/api/simulator/generate_dataset")
async def generate_simulated_dataset(
    users: List[str] = Query(default=["U01", "U02", "U03", "U04", "U05"]),
    samples_per_gesture: int = Query(default=100)
):
    """Generate multi-user simulated dataset across all 20 gestures."""
    try:
        csv_path = simulator.generate_full_dataset(
            users=users,
            samples_per_gesture=samples_per_gesture,
            output_csv_path="data/simulated/isl_simulated_dataset.csv"
        )
        stats = dataset_manager.get_summary_statistics(dataset_manager.load_dataset("simulated"))
        return {
            "message": "Simulated dataset generated successfully",
            "file_path": csv_path,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Serial / Bluetooth Endpoints
# ==========================================
@app.get("/api/serial/ports")
async def list_ports():
    """List available USB serial and Bluetooth COM ports."""
    return {"ports": SerialBluetoothManager.list_available_ports()}

@app.post("/api/serial/connect")
async def connect_serial(port: str = Query(...)):
    """Connect to a specific serial/Bluetooth COM port."""
    success = serial_manager.connect(port)
    if success:
        system_state["mode"] = "hardware"
        return {"status": "connected", "port": port}
    raise HTTPException(status_code=500, detail=f"Failed to connect to {port}")

@app.post("/api/serial/disconnect")
async def disconnect_serial():
    """Disconnect from serial/Bluetooth port."""
    serial_manager.disconnect()
    system_state["mode"] = "simulated"
    return {"status": "disconnected", "mode": "simulated"}

# ==========================================
# Dataset Management Endpoints
# ==========================================
@app.get("/api/dataset/stats")
async def get_dataset_stats(dataset_type: str = "all"):
    """Get statistics of collected datasets."""
    df = dataset_manager.load_dataset(dataset_type)
    stats = dataset_manager.get_summary_statistics(df)
    return stats

@app.post("/api/dataset/record")
async def save_recorded_session(req: RecordSessionRequest):
    """Save newly recorded gesture session to CSV."""
    try:
        saved_path = dataset_manager.append_recording_session(
            samples=req.samples,
            user_id=req.user_id,
            gesture=req.gesture,
            is_simulated=req.is_simulated
        )
        df = dataset_manager.load_dataset("all")
        stats = dataset_manager.get_summary_statistics(df)
        return {
            "message": f"Saved {len(req.samples)} samples to {saved_path}",
            "file_path": saved_path,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Calibration Endpoints
# ==========================================
@app.get("/api/calibration/profiles")
async def get_calibration_profiles():
    """Get all saved calibration profiles."""
    return {"profiles": calibration_service.user_profiles}

@app.get("/api/calibration/{user_id}")
async def get_user_calibration(user_id: str):
    """Get calibration profile for a specific user."""
    profile = calibration_service.get_profile(user_id)
    return profile

@app.post("/api/calibration/compute")
async def compute_user_calibration(req: CalibrationComputeRequest):
    """Compute calibration limits from 3-step recorded samples."""
    try:
        profile = calibration_service.compute_calibration_from_samples(
            user_id=req.user_id,
            relaxed_samples=req.relaxed_samples,
            straight_samples=req.straight_samples,
            bent_samples=req.bent_samples
        )
        return {
            "message": f"Calibration computed and saved for user {req.user_id}",
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Machine Learning Endpoints
# ==========================================
@app.post("/api/ml/train")
async def train_models(req: TrainRequest):
    """Train and compare Decision Tree, Random Forest, and SVM."""
    df = dataset_manager.load_dataset(req.dataset_type)
    if df.empty or len(df) < 20:
        # If no dataset exists yet, auto-generate simulated dataset
        simulator.generate_full_dataset()
        df = dataset_manager.load_dataset("all")

    try:
        if req.evaluation_mode == "user_independent":
            # Check if train/test users exist in dataset
            existing_users = df["user_id"].unique().tolist()
            train_u = [u for u in req.train_users if u in existing_users]
            test_u = [u for u in req.test_users if u in existing_users]

            if not train_u or not test_u:
                # Fallback to splitting existing users
                if len(existing_users) > 1:
                    test_u = [existing_users[-1]]
                    train_u = existing_users[:-1]
                else:
                    return evaluator.run_stratified_evaluation(df, test_size=0.2)

            summary = evaluator.run_user_independent_evaluation(df, train_users=train_u, test_users=test_u)
        else:
            summary = evaluator.run_stratified_evaluation(df, test_size=req.test_size or 0.2)

        # Reload active model in prediction service
        prediction_service.reload_model()

        return {
            "message": "Model training and evaluation completed successfully",
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/api/ml/evaluation")
async def get_evaluation_report():
    """Retrieve the latest model comparison evaluation results."""
    report = evaluator.get_latest_evaluation_report()
    if not report:
        raise HTTPException(status_code=404, detail="No evaluation report found. Please train models first.")
    return report

@app.post("/api/ml/train_lstm")
async def train_lstm_model(epochs: int = Query(default=20)):
    """Train advanced dynamic time-series LSTM model."""
    df = dataset_manager.load_dataset("all")
    if df.empty or len(df) < 50:
        simulator.generate_full_dataset()
        df = dataset_manager.load_dataset("all")

    try:
        report = lstm_pipeline.train_lstm(df, epochs=epochs)
        return {
            "message": "LSTM Dynamic Model trained successfully",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM training failed: {str(e)}")

@app.post("/api/ml/select_model")
async def select_active_model(req: ModelSelectRequest):
    """Switch active prediction model."""
    success = prediction_service.reload_model(preferred_model=req.model_name)
    if success:
        return {
            "message": f"Active model set to {req.model_name}",
            "status": prediction_service.model_selector.get_model_status()
        }
    raise HTTPException(status_code=400, detail=f"Failed to load model {req.model_name}")

@app.post("/api/ml/export_tinyml")
async def export_tinyml_model(model_name: str = Query(default="decision_tree")):
    """Export trained model to pure C++ code for on-device ESP32 inference."""
    try:
        exporter = TinyMLExporter(models_dir="models/saved", esp32_dir="esp32")
        res = exporter.export_decision_tree(model_name=model_name)
        return {
            "message": "TinyML C++ model generated successfully for ESP32",
            "files": res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TinyML export failed: {str(e)}")

# ==========================================
# Real-Time Prediction & TTS Endpoints
# ==========================================
@app.post("/api/predict")
async def predict_sample(req: PredictRequest):
    """Predict gesture from 11-feature sensor dictionary."""
    result = prediction_service.predict(
        raw_sensor_dict=req.features,
        user_id=req.user_id,
        apply_calibration=req.apply_calibration
    )
    # Add translation
    gesture = result.get("gesture", "UNKNOWN")
    result["text_en"] = tts_service.get_speech_text(gesture, "en")
    result["text_hi"] = tts_service.get_speech_text(gesture, "hi")
    return result

@app.post("/api/tts/sentence/add")
async def add_to_sentence(gesture: str = Query(...)):
    """Add recognized gesture to fluent sentence builder."""
    buf = tts_service.add_to_sentence(gesture)
    return {
        "sentence_en": tts_service.get_full_sentence("en"),
        "buffer": buf
    }

@app.post("/api/tts/sentence/clear")
async def clear_sentence():
    """Clear sentence builder."""
    tts_service.clear_sentence()
    return {"sentence_en": "", "buffer": []}


# ==========================================
# Real-Time WebSocket Streaming
# ==========================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket streaming 50Hz sensor data, 3D hand visualizer telemetry,
    and real-time gesture prediction.
    """
    await websocket.accept()
    step = 0
    try:
        while True:
            # Check source
            if system_state["mode"] == "hardware" and serial_manager.is_connected:
                latest = serial_manager.get_latest_sample()
                if latest:
                    sample = latest
                else:
                    sample = simulator.generate_sample(
                        gesture_id=system_state["active_sim_gesture"],
                        user_id=system_state["active_sim_user"],
                        step_index=step
                    )
            else:
                sample = simulator.generate_sample(
                    gesture_id=system_state["active_sim_gesture"],
                    user_id=system_state["active_sim_user"],
                    step_index=step
                )

            step += 1

            # Run real-time prediction
            pred_result = prediction_service.predict(
                raw_sensor_dict=sample,
                user_id=sample.get("user_id", "U01"),
                apply_calibration=True
            )

            # Append text translation
            pred_gesture = pred_result.get("gesture", "UNKNOWN")
            pred_result["text_en"] = tts_service.get_speech_text(pred_gesture, "en")
            pred_result["text_hi"] = tts_service.get_speech_text(pred_gesture, "hi")

            payload = {
                "type": "sensor_frame",
                "timestamp": sample["timestamp"],
                "mode": system_state["mode"],
                "raw_sample": sample,
                "prediction": pred_result,
                "is_recording": system_state["is_recording"]
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.02)  # 50 Hz = 20ms interval

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


# ==========================================
# Static Files & UI Serving
# ==========================================
frontend_dir = os.path.abspath("frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ISL Glove Backend Running. Please visit /docs for Swagger API documentation."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
