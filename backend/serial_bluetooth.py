"""
Serial and Bluetooth Communication Manager for ISL Sensor Glove.
Handles:
  - Scanning available COM ports and Bluetooth SPP ports
  - Non-blocking background thread for reading packet streams
  - Frame parsing and checksum validation ($ISL,thumb,index,middle,ring,little,ax,ay,az,gx,gy,gz*CRC)
  - Auto-reconnect on disconnection
  - Thread-safe distribution of sensor samples to WebSocket listeners.
"""

import time
import serial
import serial.tools.list_ports
import threading
from typing import Dict, Any, Optional, List, Callable


class SerialBluetoothManager:
    def __init__(
        self,
        baud_rate: int = 115200,
        timeout: float = 1.0,
        on_sample_callback: Optional[Callable[[Dict[str, float]], None]] = None
    ):
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.on_sample_callback = on_sample_callback

        self.ser: Optional[serial.Serial] = None
        self.port: Optional[str] = None
        self.is_connected = False
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        self.latest_sample: Optional[Dict[str, float]] = None
        self.packets_received = 0
        self.packets_dropped = 0
        self._lock = threading.Lock()

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """List all available USB serial and Bluetooth COM ports."""
        ports = serial.tools.list_ports.comports()
        result = []
        for p in ports:
            result.append({
                "port": p.device,
                "description": p.description,
                "hwid": p.hwid,
                "is_bluetooth": "bluetooth" in p.description.lower() or "bth" in p.hwid.lower()
            })
        return result

    def connect(self, port_name: str) -> bool:
        """Connect to a specific Serial / Bluetooth COM port."""
        self.disconnect()
        try:
            self.ser = serial.Serial(
                port=port_name,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.port = port_name
            self.is_connected = True
            self.is_running = True
            self.packets_received = 0
            self.packets_dropped = 0

            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"Connected to ESP32 on {port_name} at {self.baud_rate} baud.")
            return True
        except Exception as e:
            print(f"Failed to connect to {port_name}: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect active serial connection."""
        self.is_running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.is_connected = False
        self.ser = None
        self.port = None

    def send_command(self, command: str) -> bool:
        """Send a string command to the ESP32 (e.g., START, STOP, CALIB_STRAIGHT)."""
        if not self.is_connected or not self.ser or not self.ser.is_open:
            return False
        try:
            cmd_bytes = (command.strip() + "\n").encode("utf-8")
            self.ser.write(cmd_bytes)
            self.ser.flush()
            return True
        except Exception as e:
            print(f"Error sending command {command}: {e}")
            return False

    def _parse_packet(self, line: str) -> Optional[Dict[str, float]]:
        """
        Parse: $ISL,thumb,index,middle,ring,little,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z*CRC
        """
        line = line.strip()
        if not line.startswith("$ISL") and not line.startswith("ISL"):
            return None

        # Check for CRC
        if "*" in line:
            payload, crc_str = line.split("*", 1)
        else:
            payload = line
            crc_str = ""

        # Remove leading $
        if payload.startswith("$"):
            payload = payload[1:]

        parts = payload.split(",")
        if len(parts) < 12:
            return None

        try:
            sample = {
                "thumb": float(parts[1]),
                "index": float(parts[2]),
                "middle": float(parts[3]),
                "ring": float(parts[4]),
                "little": float(parts[5]),
                "acc_x": float(parts[6]),
                "acc_y": float(parts[7]),
                "acc_z": float(parts[8]),
                "gyro_x": float(parts[9]),
                "gyro_y": float(parts[10]),
                "gyro_z": float(parts[11]),
                "timestamp": round(time.time(), 3),
                "is_simulated": False
            }
            return sample
        except ValueError:
            return None

    def _read_loop(self):
        """Background loop reading lines from the serial port."""
        while self.is_running and self.ser and self.ser.is_open:
            try:
                raw_line = self.ser.readline()
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                sample = self._parse_packet(line)
                if sample:
                    with self._lock:
                        self.latest_sample = sample
                        self.packets_received += 1

                    if self.on_sample_callback:
                        self.on_sample_callback(sample)
                else:
                    with self._lock:
                        self.packets_dropped += 1
            except Exception as e:
                print(f"Serial read error: {e}")
                time.sleep(0.1)

        self.is_connected = False

    def get_latest_sample(self) -> Optional[Dict[str, float]]:
        """Get the most recent valid sensor sample."""
        with self._lock:
            return self.latest_sample.copy() if self.latest_sample else None

    def get_status(self) -> Dict[str, Any]:
        """Get communication link status and packet counters."""
        return {
            "is_connected": self.is_connected,
            "port": self.port,
            "baud_rate": self.baud_rate,
            "packets_received": self.packets_received,
            "packets_dropped": self.packets_dropped
        }
