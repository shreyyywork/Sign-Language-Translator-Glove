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
        self.connection_type: str = "disconnected"  # "usb", "bluetooth", "com", or "disconnected"
        self.connected_at: Optional[float] = None
        self.last_error: str = ""
        self.is_connected = False
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        self.latest_sample: Optional[Dict[str, float]] = None
        self.packets_received = 0
        self.packets_dropped = 0
        self._lock = threading.Lock()

    @staticmethod
    def identify_port(p) -> Dict[str, Any]:
        """Classify a serial port as USB, Bluetooth, or Standard/Virtual COM with chip identification."""
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        device = (p.device or "").lower()

        is_bt = "bluetooth" in desc or "bth" in hwid or "bth" in device or "bthenum" in hwid

        # Detect USB-UART bridge chips common on ESP32 dev boards
        chip = "Standard Serial"
        is_esp_candidate = False

        if "cp210" in desc or "cp210" in hwid or getattr(p, "vid", None) == 0x10C4:
            chip = "Silicon Labs CP210x (ESP32 NodeMCU)"
            is_esp_candidate = True
        elif "ch340" in desc or "ch341" in desc or "ch9102" in desc or getattr(p, "vid", None) == 0x1A86:
            chip = "WCH CH340 / CH9102 (ESP32 DevKit)"
            is_esp_candidate = True
        elif "ftdi" in desc or "ft232" in desc or getattr(p, "vid", None) == 0x0403:
            chip = "FTDI USB-to-UART"
            is_esp_candidate = True
        elif "espressif" in desc or "jtag" in desc or getattr(p, "vid", None) == 0x303A:
            chip = "Espressif USB JTAG / CDC (ESP32-S2/S3/C3)"
            is_esp_candidate = True
        elif "prolific" in desc or "pl2303" in desc or getattr(p, "vid", None) == 0x067B:
            chip = "Prolific USB Serial"
        elif "usb" in desc or "usb" in hwid or getattr(p, "vid", None) is not None:
            chip = "USB Serial Device"
            is_esp_candidate = True

        is_usb = not is_bt and ("usb" in desc or "usb" in hwid or getattr(p, "vid", None) is not None or is_esp_candidate)

        if is_usb:
            port_type = "usb"
        elif is_bt:
            port_type = "bluetooth"
        else:
            port_type = "com"

        return {
            "port": p.device,
            "description": p.description,
            "hwid": p.hwid,
            "port_type": port_type,
            "is_usb": is_usb,
            "is_bluetooth": is_bt,
            "chip": chip,
            "is_esp32_candidate": is_esp_candidate
        }

    @classmethod
    def list_available_ports(cls) -> List[Dict[str, Any]]:
        """List all available USB serial and Bluetooth COM ports."""
        ports = serial.tools.list_ports.comports()
        return [cls.identify_port(p) for p in ports]

    @classmethod
    def list_categorized_ports(cls) -> Dict[str, Any]:
        """Categorize ports into USB, Bluetooth, and Other COM ports with ESP32 recommendation."""
        all_ports = cls.list_available_ports()
        usb_ports = [p for p in all_ports if p["is_usb"]]
        bluetooth_ports = [p for p in all_ports if p["is_bluetooth"]]
        other_com_ports = [p for p in all_ports if not p["is_usb"] and not p["is_bluetooth"]]

        # Recommend best candidate ESP32 USB port
        recommended_usb = None
        candidates = [p for p in usb_ports if p["is_esp32_candidate"]]
        if candidates:
            recommended_usb = candidates[0]["port"]
        elif usb_ports:
            recommended_usb = usb_ports[0]["port"]

        return {
            "ports": all_ports,  # Preserved for backward compatibility
            "all_ports": all_ports,
            "usb_ports": usb_ports,
            "bluetooth_ports": bluetooth_ports,
            "other_com_ports": other_com_ports,
            "recommended_usb_port": recommended_usb,
            "recommended_baud": 115200,
            "supported_bauds": [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
        }

    @classmethod
    def auto_detect_esp32_usb(cls) -> Optional[str]:
        """Automatically find the best candidate USB port for an ESP32 board."""
        categorized = cls.list_categorized_ports()
        return categorized.get("recommended_usb_port")

    def connect(
        self,
        port_name: str,
        baud_rate: Optional[int] = None,
        connection_type: str = "auto"
    ) -> bool:
        """Connect to a specific USB Serial or Bluetooth COM port."""
        self.disconnect()
        if baud_rate:
            self.baud_rate = baud_rate

        # Determine connection type if auto
        if connection_type == "auto":
            ports = self.list_available_ports()
            matching = [p for p in ports if p["port"].upper() == port_name.upper()]
            if matching:
                self.connection_type = matching[0]["port_type"]
            else:
                self.connection_type = "com"
        else:
            self.connection_type = connection_type

        try:
            self.ser = serial.Serial(
                port=port_name,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.port = port_name
            self.is_connected = True
            self.is_running = True
            self.connected_at = time.time()
            self.last_error = ""
            self.packets_received = 0
            self.packets_dropped = 0

            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            print(f"Connected to ESP32 on {port_name} via {self.connection_type.upper()} at {self.baud_rate} baud.")
            return True
        except Exception as e:
            err_str = str(e)
            if "Access is denied" in err_str or "PermissionError" in err_str:
                self.last_error = (
                    f"Access denied on {port_name}. This port is currently locked by another application "
                    f"(such as the Arduino IDE Serial Monitor or Serial Plotter). "
                    f"Please close the Serial Monitor in Arduino IDE and click Connect again."
                )
            elif "FileNotFoundError" in err_str or "cannot find the file" in err_str.lower():
                self.last_error = f"Port {port_name} not found. Please verify the ESP32 USB cable is securely connected."
            else:
                self.last_error = f"Failed to connect to {port_name}: {err_str}"

            print(f"[SERIAL ERROR] {self.last_error}")
            self.is_connected = False
            self.connection_type = "disconnected"
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
        self.connection_type = "disconnected"
        self.connected_at = None

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

    def set_esp_comm_mode(self, mode: str) -> bool:
        """
        Send command to switch ESP32 between USB and Bluetooth modes.
        mode can be 'usb', 'bluetooth', or 'dual'.
        """
        mode_upper = mode.strip().upper()
        if mode_upper in ["USB", "USB_PORT"]:
            cmd = "MODE_USB"
        elif mode_upper in ["BT", "BLUETOOTH", "BLUETOOTH_COM"]:
            cmd = "MODE_BT"
        elif mode_upper in ["DUAL", "DUAL_USB_BT"]:
            cmd = "MODE_DUAL"
        else:
            return False
        return self.send_command(cmd)

    def get_status(self) -> Dict[str, Any]:
        """Get communication link status and packet counters."""
        uptime = round(time.time() - self.connected_at, 1) if (self.is_connected and self.connected_at) else 0
        return {
            "is_connected": self.is_connected,
            "port": self.port,
            "connection_type": self.connection_type,
            "baud_rate": self.baud_rate,
            "uptime_seconds": uptime,
            "packets_received": self.packets_received,
            "packets_dropped": self.packets_dropped
        }
