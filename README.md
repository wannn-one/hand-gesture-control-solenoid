## Hand Gesture Control for Solenoid (OpenCV + MediaPipe + ESP32/Arduino)

Control a solenoid lock using simple hand gestures detected via your webcam. An open palm sends OPEN, a fist sends CLOSE. Auto-lock re-closes after a configurable timeout.

### Features
- **Open palm**: sends `COMMAND_OPEN` (default `b'1'`)
- **Fist**: sends `COMMAND_CLOSE` (default `b'0'`)
- **Auto-lock** after `AUTO_LOCK_SECONDS`
- Lightweight, single-camera, single-hand processing

### Requirements
- Python 3.10+ on Windows
- Webcam
- ESP32/Arduino connected via USB serial

Install Python packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Project Structure
- `gesture_control.py`: main app
- `config/config.py`: runtime configuration (serial, camera, thresholds)
- `.gitignore`: common Python ignores
- `requirements.txt`: minimal runtime deps for this project

### Configuration
Edit `config/config.py`:
- `ESP32_PORT`: your COM port (e.g., `COM9`)
- `BAUD_RATE`: serial speed (default `9600`)
- `CAMERA_INDEX`: default `0` (change if you have multiple cameras)
- `AUTO_LOCK_SECONDS`: seconds before auto-close when open palm no longer detected
- `COMMAND_OPEN` / `COMMAND_CLOSE`: bytes sent to the microcontroller

Example:
```python
ESP32_PORT = 'COM9'
BAUD_RATE = 9600
CAMERA_INDEX = 0
AUTO_LOCK_SECONDS = 5
COMMAND_OPEN = b'1'
COMMAND_CLOSE = b'0'
```

### Running (Windows PowerShell)
```bash
# Optional: create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run
python gesture_control.py
```
Press `q` in the camera window to quit.

### Gestures
- Open palm (index and middle fingertip above their PIP joints) → OPEN
- Fist (index fingertip below its PIP joint) → CLOSE

Status is shown on the video stream: `OPEN_PALM`, `FIST`, `AUTO_LOCKED`, or last state.

### Microcontroller Notes
- Ensure your ESP32/Arduino firmware listens on the serial port at `BAUD_RATE`
- React to single-byte commands: `1` to open, `0` to close (customizable)
- After program exit, the app attempts to send CLOSE and then closes the serial port

### Wiring Diagram
Below is the connection guide between components.

1. Control Circuit (ESP32 to Relay Module)
This circuit uses low voltage (5V/3.3V) to control the relay.

- ESP32 VIN → Relay VCC  
  (Provide 5V power to the relay module)
- ESP32 GND → Relay GND  
  (Share common ground between ESP32 and relay)
- ESP32 GPIO4 → Relay IN  
  (Send ON/OFF signal to the relay)

2. Load Circuit (Relay to Solenoid Lock)
This circuit uses an external power supply (e.g., 12V) and is isolated from the ESP32. Do not connect this circuit to ESP32 pins.

- Power Supply + (Positive) → Relay COM (Common)  
  (Input power to the relay switch)
- Relay NO (Normally Open) → Solenoid + (Positive)  
  (Power flows from the relay to the solenoid when activated)
- Solenoid - (Negative) → Power Supply - (Negative)  
  (Complete the circuit back to the power source)

### Troubleshooting
- Camera not opening: adjust `CAMERA_INDEX` in `config/config.py`
- Serial port error: verify `ESP32_PORT` (check Device Manager → Ports)
- High CPU usage: lower `MODEL_COMPLEXITY`, increase `PROCESS_EVERY_N_FRAMES`, or reduce `FRAME_SCALE`
- Window not rendering: ensure a display is available and no other app locks the camera

### Development Tips
- To tweak performance:
  - `MAX_NUM_HANDS = 1`
  - `MODEL_COMPLEXITY = 0` (faster) or `1/2` (more accurate)
  - `FRAME_SCALE` in `(0,1]` to downscale processing
  - `PROCESS_EVERY_N_FRAMES = 2` or more to skip frames

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.