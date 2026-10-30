"""
Unit tests for Serial and Bluetooth Manager USB/COM port connection handling.
"""

from unittest.mock import MagicMock, patch
from backend.serial_bluetooth import SerialBluetoothManager


def test_identify_port_usb():
    mock_port = MagicMock()
    mock_port.device = "COM3"
    mock_port.description = "Silicon Labs CP210x USB to UART Bridge (COM3)"
    mock_port.hwid = "USB VID:PID=10C4:EA60"
    mock_port.vid = 0x10C4
    mock_port.pid = 0xEA60

    info = SerialBluetoothManager.identify_port(mock_port)
    assert info["port"] == "COM3"
    assert info["is_usb"] is True
    assert info["is_bluetooth"] is False
    assert info["port_type"] == "usb"
    assert "CP210" in info["chip"]
    assert info["is_esp32_candidate"] is True


def test_identify_port_bluetooth():
    mock_port = MagicMock()
    mock_port.device = "COM4"
    mock_port.description = "Standard Serial over Bluetooth link (COM4)"
    mock_port.hwid = "BTHENUM\\{00001101-0000-1000-8000-00805F9B34FB}"
    mock_port.vid = None
    mock_port.pid = None

    info = SerialBluetoothManager.identify_port(mock_port)
    assert info["port"] == "COM4"
    assert info["is_usb"] is False
    assert info["is_bluetooth"] is True
    assert info["port_type"] == "bluetooth"


def test_list_categorized_ports_mocked():
    mock_p1 = MagicMock(device="COM3", description="Silicon Labs CP210x USB to UART Bridge (COM3)", hwid="USB VID:PID=10C4:EA60", vid=0x10C4, pid=0xEA60)
    mock_p2 = MagicMock(device="COM4", description="Standard Serial over Bluetooth link (COM4)", hwid="BTHENUM", vid=None, pid=None)
    mock_p3 = MagicMock(device="COM1", description="Communications Port (COM1)", hwid="ACPI\\PNP0501", vid=None, pid=None)

    with patch("serial.tools.list_ports.comports", return_value=[mock_p1, mock_p2, mock_p3]):
        data = SerialBluetoothManager.list_categorized_ports()
        assert len(data["usb_ports"]) == 1
        assert data["usb_ports"][0]["port"] == "COM3"
        assert len(data["bluetooth_ports"]) == 1
        assert data["bluetooth_ports"][0]["port"] == "COM4"
        assert len(data["other_com_ports"]) == 1
        assert data["other_com_ports"][0]["port"] == "COM1"
        assert data["recommended_usb_port"] == "COM3"
        assert data["recommended_baud"] == 115200


def test_set_esp_comm_mode():
    manager = SerialBluetoothManager()
    manager.is_connected = True
    manager.ser = MagicMock()
    manager.ser.is_open = True

    assert manager.set_esp_comm_mode("usb") is True
    manager.ser.write.assert_called_with(b"MODE_USB\n")

    assert manager.set_esp_comm_mode("bluetooth") is True
    manager.ser.write.assert_called_with(b"MODE_BT\n")

    assert manager.set_esp_comm_mode("dual") is True
    manager.ser.write.assert_called_with(b"MODE_DUAL\n")

    assert manager.set_esp_comm_mode("invalid_mode") is False
