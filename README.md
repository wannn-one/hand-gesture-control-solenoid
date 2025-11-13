 # Hand Gesture Control for Solenoid Lock: Comprehensive Documentation

This document provides a comprehensive guide to the "Hand Gesture Control for Solenoid Lock" project, detailing its features, setup, usage, and configuration. This project enables the control of solenoid locks using hand gestures detected via computer vision, offering both a straightforward rule-based method and an advanced machine learning approach.

## 1. Project Overview

The `hand-gesture-control-solenoid` project is designed to provide a robust and flexible solution for controlling solenoid locks through real-time hand gesture recognition. It caters to different user needs by offering two distinct methodologies:

*   **Simple Rule-Based Approach**: Ideal for beginners, this method uses predefined hand gestures (open palm for unlock, fist for lock) detected by MediaPipe.
*   **Machine Learning Approach**: For advanced users, this method allows for custom gesture training using a neural network, offering high accuracy and extensibility.

Both approaches support various hardware configurations, including ESP32/Arduino for serial communication and Raspberry Pi for direct GPIO control.

## 2. Features

### 2.1. Simple Approach (`app/gesture_control.py`)

This approach is designed for ease of use and quick deployment.

*   **Predefined Gestures**:
    *   **Open palm**: Sends `COMMAND_OPEN` (default `b'1'`) to unlock.
    *   **Fist**: Sends `COMMAND_CLOSE` (default `b'0'`) to lock.
*   **Auto-lock Functionality**: Automatically locks the solenoid after a configurable timeout if no open palm gesture is detected.
*   **Real-time Detection**: Utilizes MediaPipe for efficient and real-time hand gesture recognition.
*   **Lightweight Processing**: Optimized for single-camera processing.
*   **Serial Communication**: Seamless integration with ESP32/Arduino boards for controlling the solenoid.

### 2.2. Machine Learning Approach (`dataset_take.py`, `train.py`, `predict.py`)

This advanced approach offers customization and higher accuracy.

*   **Custom Gesture Training**: Allows users to define and train their own hand gestures using a neural network model.
*   **Dataset Collection Tool**: Includes `dataset_take.py` for easily gathering custom gesture data.
*   **High Accuracy**: Achieves 90%+ confidence threshold for gesture recognition with trained models.
*   **Raspberry Pi Support**: Direct GPIO control via `lgpio` library for Raspberry Pi devices.
*   **Extensible**: New gesture types can be easily added and integrated into the system.

## 3. Requirements

### 3.1. Hardware

*   **Webcam**: Any standard USB or built-in webcam.
*   **Microcontroller**:
    *   **ESP32/Arduino**: Required for serial communication in the Simple Approach.
    *   **Raspberry Pi**: Required for direct GPIO control in the Machine Learning Approach.
*   **Solenoid Lock**: A 12V solenoid lock.
*   **Relay Module**: An appropriate relay module (5V for ESP32/Arduino, 3.3V/5V compatible for Raspberry Pi) to interface the microcontroller with the solenoid.
*   **Power Supply**: A 12V power supply for the solenoid lock.
*   **Jumper Wires**: For connecting components.

### 3.2. Software

*   **Python**: Version 3.11 or higher (compatible with Windows, Linux, and Raspberry Pi).
*   **Dependencies**: All required Python packages are listed in `requirements.txt`.

## 4. Project Structure

The project is organized into a clear and logical directory structure:

```
hand-gesture-control-solenoid/
├── app/
│   └── gesture_control.py    # Main application for simple rule-based control
├── config/
│   └── config.py             # Runtime configuration (COM port, camera, etc.)
├── dataset_take.py           # Tool to collect custom gesture data for ML
├── train.py                  # Script to train the neural network model
├── predict.py                # Script for ML-based gesture recognition (for RPi)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # Project license
```

## 5. Installation

### 5.1. Python Environment Setup

It is highly recommended to use a virtual environment to manage project dependencies.

1.  **Create a Virtual Environment**:

    *   **Windows PowerShell**:
        ```bash
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
        ```
    *   **Linux/Raspberry Pi**:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

2.  **Install Dependencies**:
    Upgrade `pip` and then install all required packages from `requirements.txt`.

    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

### 5.2. Platform-Specific Dependencies

*   **Raspberry Pi (for GPIO control)**:
    ```bash
    sudo apt update
    sudo apt install python3-lgpio
    # OR
    pip install lgpio
    ```
*   **Windows Only**:
    *   Ensure **Visual C++ Redistributable** is installed (often required for TensorFlow).
    *   Install **USB drivers** for your specific ESP32/Arduino board.

## 6. Quick Start Guide

### 6.1. Option 1: Simple Gesture Control (Recommended for beginners)

This option uses the `gesture_control.py` script for immediate use with predefined gestures.

1.  **Hardware Setup**: Connect your ESP32/Arduino board to your computer via USB.
2.  **Configuration**: Edit `config/config.py` to set your specific COM port (e.g., `ESP32_PORT = 'COM9'` on Windows or `'/dev/ttyUSB0'` on Linux).
3.  **Run Application**:
    ```bash
    python app/gesture_control.py
    ```
4.  **Control**:
    *   Show an **open palm** to unlock the solenoid.
    *   Form a **fist** to lock the solenoid.

### 6.2. Option 2: Machine Learning Approach (Advanced users)

This option involves training a custom model and deploying it, typically on a Raspberry Pi.

1.  **Collect Data**:
    ```bash
    python dataset_take.py
    ```
    Follow the on-screen instructions to record your custom gestures. This will create `dataset_landmarks.csv`.
2.  **Train Model**:
    ```bash
    python train.py
    ```
    This script will train a neural network model and save it as `model_gestur.h5` and the label encoder as `label_encoder.pkl`.
3.  **Deploy**:
    ```bash
    python predict.py
    ```
    This script uses the trained model for real-time gesture recognition and controls GPIO on a Raspberry Pi.

## 7. Detailed Usage Instructions

### 7.1. Simple Gesture Control (`app/gesture_control.py`)

After following the installation and quick start steps:

*   **Controls**:
    *   **Open palm**: Detected when the index and middle fingertips are above their PIP (Proximal Interphalangeal) joints. This triggers the `COMMAND_OPEN` signal, unlocking the solenoid.
    *   **Fist**: Detected when the index fingertip is below its PIP joint. This triggers the `COMMAND_CLOSE` signal, locking the solenoid.
    *   **Auto-lock**: If no open palm gesture is detected for the duration specified by `AUTO_LOCK_SECONDS`, the system will automatically send the `COMMAND_CLOSE` signal to lock the solenoid.
*   **Exiting**: Press the `q` key in the camera window to quit the application.

### 7.2. Machine Learning Approach

#### 7.2.1. Step 1: Collect Training Data (`dataset_take.py`)

This script helps you build a dataset of hand landmarks for your custom gestures.

```bash
python dataset_take.py
```

*   The script will display a camera feed.
*   **Recording**: Press `S` to start recording a sequence of landmarks for a specific gesture. Repeat this for multiple sequences per gesture.
*   **Default Gestures**: The script is configured to collect data for `buka_kunci` (unlock) and `kunci` (lock) by default. You can modify the script to add more gesture types.
*   **Output**: This process generates `dataset_landmarks.csv`, which contains normalized hand landmark coordinates for each recorded gesture.
*   **Quitting**: Press `Q` to quit the data collection tool.

#### 7.2.2. Step 2: Train the Model (`train.py`)

Once you have collected sufficient data, use this script to train your neural network model.

```bash
python train.py
```

*   This script reads `dataset_landmarks.csv`.
*   It trains a neural network model based on the collected data.
*   **Output**:
    *   `model_gestur.h5`: The trained neural network model.
    *   `label_encoder.pkl`: A file containing the mapping of gesture names to numerical labels, essential for prediction.

#### 7.2.3. Step 3: Deploy with Raspberry Pi (`predict.py`)

This script uses the trained model to perform real-time gesture recognition and control GPIO pins on a Raspberry Pi.

```bash
python predict.py
```

*   Ensure `model_gestur.h5` and `label_encoder.pkl` are in the same directory as `predict.py`.
*   The script will use the webcam to detect hands, predict gestures using the loaded model, and activate the configured GPIO pin on the Raspberry Pi accordingly.

## 8. Configuration

All runtime parameters are managed in `config/config.py`.

### 8.1. Basic Configuration (`config/config.py`)

```python
# Serial Communication
ESP32_PORT = 'COM9'          # Your Arduino/ESP32 COM port (e.g., 'COM9' on Windows, '/dev/ttyUSB0' on Linux)
BAUD_RATE = 9600             # Serial communication speed (must match Arduino sketch)
SERIAL_TIMEOUT = 1           # Serial read/write timeout in seconds
SERIAL_STARTUP_DELAY = 2     # Delay in seconds for Arduino/ESP32 to initialize after connection

# Camera Settings
CAMERA_INDEX = 0             # Camera index (0 = default camera, 1 = second camera, etc.)
CAMERA_WIDTH = 640           # Camera resolution width
CAMERA_HEIGHT = 480          # Camera resolution height
WINDOW_TITLE = 'MediaPipe Hand Gesture Control' # Title for the camera display window

# Gesture Detection
MIN_DETECTION_CONFIDENCE = 0.7  # Minimum confidence score for hand detection (0.0-1.0)
MIN_TRACKING_CONFIDENCE = 0.5   # Minimum confidence score for hand tracking (0.0-1.0)
AUTO_LOCK_SECONDS = 5           # Time in seconds after which the solenoid auto-locks if no open palm is detected
COMMAND_OPEN = b'1'             # Byte command sent to unlock the solenoid
COMMAND_CLOSE = b'0'            # Byte command sent to lock the solenoid

# Performance Optimization
MAX_NUM_HANDS = 1               # Maximum number of hands to detect and process (1 or 2)
MODEL_COMPLEXITY = 0            # MediaPipe model complexity: 0=faster, 1/2=more accurate
FRAME_SCALE = 0.75              # Scale factor for processing frames (0.1-1.0, lower for faster processing)
PROCESS_EVERY_N_FRAMES = 1      # Process every N frames (1=every frame, 2=every other frame, etc.)
```

## 9. Hardware Setup

### 9.1. ESP32/Arduino Setup (Simple Approach)

#### 9.1.1. Required Components

*   ESP32 or Arduino board
*   Relay module (5V)
*   Solenoid lock (12V)
*   12V power supply
*   Jumper wires

#### 9.1.2. Arduino Code Example

Upload this sketch to your ESP32/Arduino board.

```cpp
void setup() {
  Serial.begin(9600); // Initialize serial communication at 9600 baud
  pinMode(4, OUTPUT); // Set digital pin 4 as an output for relay control
  digitalWrite(4, LOW); // Ensure the solenoid starts in a locked state (assuming LOW activates relay for lock)
}

void loop() {
  if (Serial.available()) { // Check if data is available from serial port
    char command = Serial.read(); // Read the incoming byte
    if (command == '1') {
      digitalWrite(4, HIGH);  // Unlock: Set pin 4 HIGH (assuming HIGH activates relay for unlock)
    } else if (command == '0') {
      digitalWrite(4, LOW);   // Lock: Set pin 4 LOW
    }
  }
}
```

#### 9.1.3. ESP32/Arduino Wiring Diagram (Text-based)

```
ESP32/Arduino          Relay Module          Solenoid Lock
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│    VIN      │──────▶│    VCC      │       │             │
│    GND      │──────▶│    GND      │       │             │
│   GPIO4     │──────▶│     IN      │       │             │
└─────────────┘       │             │       │             │
                      │    COM      │◀──────│ 12V Supply+ │
                      │     NO      │──────▶│ Solenoid+   │
                      └─────────────┘       │ Solenoid-   │◀──┐
                                           └─────────────┘   │
                                                            │
                                           ┌─────────────┐   │
                                           │ 12V Supply- │───┘
                                           └─────────────┘
```

*   **ESP32/Arduino VIN/5V** to **Relay VCC**
*   **ESP32/Arduino GND** to **Relay GND**
*   **ESP32/Arduino GPIO4** to **Relay IN** (or the signal pin you've chosen)
*   **12V Power Supply Positive** to **Relay COM**
*   **Relay NO (Normally Open)** to **Solenoid Positive**
*   **Solenoid Negative** to **12V Power Supply Negative**

### 9.2. Raspberry Pi Setup (ML Approach)

#### 9.2.1. Required Components

*   Raspberry Pi (any model with GPIO pins)
*   Relay module (3.3V/5V compatible)
*   Solenoid lock (12V)
*   12V power supply
*   Jumper wires

#### 9.2.2. GPIO Configuration

*   **GPIO 17 (BCM)**: This pin is designated for the relay control signal.
*   **5V/3.3V**: Connect to the relay's VCC.
*   **GND**: Connect to the common ground.

#### 9.2.3. Raspberry Pi Wiring Diagram (Text-based)

```
Raspberry Pi           Relay Module          Solenoid Lock
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   5V/3.3V   │──────▶│    VCC      │       │             │
│    GND      │──────▶│    GND      │       │             │
│  GPIO17     │──────▶│     IN      │       │             │
└─────────────┘       │             │       │             │
                      │    COM      │◀──────│ 12V Supply+ │
                      │     NO      │──────▶│ Solenoid+   │
                      └─────────────┘       │ Solenoid-   │◀──┐
                                           └─────────────┘   │
                                                            │
                                           ┌─────────────┐   │
                                           │ 12V Supply- │───┘
                                           └─────────────┘
```

*   **Raspberry Pi 5V/3.3V** to **Relay VCC**
*   **Raspberry Pi GND** to **Relay GND**
*   **Raspberry Pi GPIO17** to **Relay IN**
*   **12V Power Supply Positive** to **Relay COM**
*   **Relay NO (Normally Open)** to **Solenoid Positive**
*   **Solenoid Negative** to **12V Power Supply Negative**

### 9.3. Safety Notes

*   **Disconnect Power**: Always ensure all power supplies are disconnected before making any wiring changes.
*   **Polarity**: Double-check the polarity of all power connections (12V supply, solenoid). Incorrect wiring can damage components.
*   **Wire Gauge**: Use appropriate gauge wires for the 12V power supply to handle the current draw of the solenoid.
*   **Relay Rating**: Verify that your relay module is rated to handle the current and voltage requirements of your solenoid lock.

## 10. Troubleshooting

### 10.1. Common Issues

*   **Camera Problems**:
    *   **Camera not opening**: Try adjusting `CAMERA_INDEX` in `config/config.py` (e.g., 0, 1, 2).
    *   **Poor image quality**: Check `CAMERA_WIDTH` and `CAMERA_HEIGHT` settings in `config/config.py`.
    *   **Camera already in use**: Close any other applications that might be using the webcam.
    *   **No camera detected**: Ensure camera drivers are correctly installed and the camera is properly connected.
*   **Serial Communication (ESP32/Arduino)**:
    *   **Permission denied**: On Linux, you might need to add your user to the `dialout` group (`sudo usermod -a -G dialout $USER`) or run the script with `sudo`. On Windows, ensure you run your terminal as an administrator.
    *   **Incorrect COM port**: Verify `ESP32_PORT` in `config/config.py` matches the port assigned to your ESP32/Arduino.
    *   **Baud rate mismatch**: Ensure `BAUD_RATE` in `config/config.py` matches the `Serial.begin()` rate in your Arduino sketch.
*   **Performance Issues**:
    *   **High CPU usage / Slow detection**: Refer to the "Performance Optimization Tips" section below.
    *   **Window not rendering**: Ensure your display environment is correctly set up and no other application is locking the camera feed.
*   **Machine Learning Issues**:
    *   **Model not loading**: Confirm that `model_gestur.h5` and `label_encoder.pkl` are present in the same directory as `predict.py` (or `train.py` if applicable).

### 10.2. Performance Optimization Tips

These settings are found in `config/config.py`.

*   **For Low-End Hardware (Prioritize Speed)**:
    ```python
    MODEL_COMPLEXITY = 0        # Fastest MediaPipe processing model
    FRAME_SCALE = 0.5           # Process frames at half resolution
    PROCESS_EVERY_N_FRAMES = 3  # Process only every 3rd frame
    MAX_NUM_HANDS = 1           # Limit hand detection to a single hand
    ```
*   **For High Accuracy (Prioritize Precision)**:
    ```python
    MODEL_COMPLEXITY = 1           # More accurate MediaPipe model (can be 2 for even higher)
    MIN_DETECTION_CONFIDENCE = 0.8 # Higher confidence threshold for hand detection
    MIN_TRACKING_CONFIDENCE = 0.7  # Higher confidence threshold for hand tracking
    FRAME_SCALE = 1.0              # Process frames at full resolution
    ```

## 11. Dependencies

### 11.1. Core Dependencies (`requirements.txt`)

*   `tensorflow==2.16.1`: The primary neural network framework used for machine learning models.
*   `mediapipe`: Google's framework for on-device machine learning solutions, used for hand landmark detection.
*   `opencv-python`: The widely used computer vision library for camera interaction and image processing.
*   `numpy`: Fundamental package for numerical computing in Python.
*   `pandas`: Used for data manipulation, especially for handling `dataset_landmarks.csv`.
*   `scikit-learn`: Provides various machine learning utilities, including model training and evaluation tools.

### 11.2. Platform-Specific Dependencies

*   **Raspberry Pi Only**:
    *   `lgpio`: A Python library for controlling GPIO pins on Raspberry Pi. Install via `sudo apt install python3-lgpio` or `pip install lgpio`.
*   **Windows Only**:
    *   **Visual C++ Redistributable**: Essential for TensorFlow to function correctly.
    *   **USB drivers**: Specific drivers for your ESP32/Arduino board to enable serial communication.

### 11.3. Optional Dependencies (for development/debugging)

*   `matplotlib`, `seaborn`: For data visualization and plotting.
*   `jupyter notebook`: For interactive development and experimentation.

## 12. Contributing

Contributions to this project are welcome! Please follow these steps:

1.  **Fork** the repository.
2.  Create a new feature branch: `git checkout -b feature/your-amazing-feature`.
3.  Commit your changes: `git commit -m 'Add your amazing feature'`.
4.  Push to the branch: `git push origin feature/your-amazing-feature`.
5.  Open a **Pull Request** to the main repository.

## 13. License

This project is licensed under the **MIT License**. See the `LICENSE` file for full details.

## 14. Acknowledgments

We extend our gratitude to the following projects and communities:

*   **MediaPipe** team for their excellent and robust hand tracking solutions.
*   **OpenCV** community for providing powerful computer vision tools.
*   **TensorFlow** team for their comprehensive machine learning framework.
*   **Arduino/ESP32** community for their invaluable microcontroller support and resources.
