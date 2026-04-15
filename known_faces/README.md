# Known Faces Directory

This directory serves as the database for storing photos of authorized users who are permitted to unlock the Smart Lock system. For privacy and efficiency reasons, image files (such as `.jpg` or `.png`) are ignored by Git (via `.gitignore`) and will not be pushed to the GitHub repository.

Therefore, when you first clone this repository, this folder will be empty (or only contain this `README.md` file). You **must** populate this folder with valid reference photos for the face recognition system to work.

## Integration with Dynamic Gesture Recognition

With the recent updates to the system (multithreading and LSTM implementation in `app/predict.py`), Face Recognition now acts as the **first layer of security** in a multi-stage pipeline:
1. **`SEARCHING_FACE` State**: The camera continuously scans for faces. If a face matches a photo in this directory, the system starts a session for that user (`SESSION_STANDBY`).
2. **Dynamic Gestures**: Once authorized, the user triggers the command recording with an "OK" gesture, followed by dynamic hand gestures (processed via the trained LSTM model and StandardScaler from the `training/` folder) to execute actions like opening or closing the solenoid lock.
3. **Session Security**: If the recognized user leaves the camera frame, the system automatically drops the session and returns to the `SEARCHING_FACE` state.

## How to Add a New User

1. Prepare a clear, front-facing photo of the user.
2. Ensure there is **only one visible face** in the photo.
3. Rename the image file to the name of the user. This filename will act as the user's ID and will be displayed on the screen during operation (e.g., "Session Started: [Name]").
   - Example: `Ikhwan.jpg`, `Yuvina.png`, or `VIP Guest.jpeg`.
4. Place the renamed photo into this `known_faces/` directory.

## Additional Rules & Troubleshooting

* **Supported Formats:** `.jpg`, `.jpeg`, `.png`.
* If this directory is empty or contains no valid images when you run the system (`app/predict.py`), you will receive a terminal warning, and the system will be permanently blocked at the "SEARCHING_FACE" state (since no one is authorized).
* The system automatically scans the folder on startup. You simply need to restart the main script after adding a new photo for the face to be registered.
