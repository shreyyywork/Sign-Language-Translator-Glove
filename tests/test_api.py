"""
Integration tests for FastAPI REST API endpoints.
"""

from fastapi.testclient import TestClient
from backend.main import app


def test_api_health():
    with TestClient(app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert "mode" in data


def test_api_gestures():
    with TestClient(app) as client:
        res = client.get("/api/gestures")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 20
        assert len(data["gestures"]) == 20


def test_api_mode_switch():
    with TestClient(app) as client:
        res = client.post("/api/mode", json={"mode": "simulated"})
        assert res.status_code == 200
        assert res.json()["mode"] == "simulated"


def test_api_set_simulator_gesture():
    with TestClient(app) as client:
        res = client.post("/api/simulator/set_gesture", json={"gesture": "THANK_YOU", "user_id": "U02"})
        assert res.status_code == 200
        assert res.json()["gesture"] == "THANK_YOU"


def test_api_serial_ports():
    with TestClient(app) as client:
        res = client.get("/api/serial/ports")
        assert res.status_code == 200
        data = res.json()
        assert "ports" in data
        assert "usb_ports" in data
        assert "bluetooth_ports" in data
        assert "other_com_ports" in data
        assert "supported_bauds" in data
        assert 115200 in data["supported_bauds"]


def test_api_serial_status():
    with TestClient(app) as client:
        res = client.get("/api/serial/status")
        assert res.status_code == 200
        data = res.json()
        assert "is_connected" in data
        assert "connection_type" in data
        assert "baud_rate" in data
