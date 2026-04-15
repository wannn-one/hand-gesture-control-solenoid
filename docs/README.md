# Raspberry Pi Setup Guide

This document is dedicated to the step-by-step configuration of the Raspberry Pi for running the Hand Gesture Control system.

## 1. Operating System Installation

The OS used on Raspberry Pi 5 is Raspberry Pi OS 64-bit Bookworm. Skip this if you're already installed the OS.

*   **Step 1:** Download Raspberry Pi Imager from [https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)
*   **Step 2:** Flash the OS to the SD card using Raspberry Pi Imager.
*   **Step 3:** Enable SSH and Wi-Fi in Raspberry Pi Imager.

## 2. Hardware Interfaces (Camera & GPIO)

Please use USB Camera instead of Pi Camera.

*   **Step 1:** Connect the USB Camera to the Raspberry Pi.
*   **Step 2:** Connect the Solenoid lock to the Raspberry Pi using configuration on [9.2.3. Raspberry Pi Wiring Diagram](../README.md#923-raspberry-pi-wiring-diagram).

## 3. Virtual Environment & Python Packages

Open terminal and run the following commands:

```bash
source handgesture/bin/activate
cd /home/yuvina/Desktop/hand-gesture-control-solenoid
```

## 4. Testing the Hardware connection
[Brief instructions on how to test the Solenoid lock physically using a simple script before running the main AI pipeline.]

Run on same terminal the following commands. please make sure the diagram is completely connected:

```bash
python3 app/predict.py
```

And that's it. The system will automatically start running.