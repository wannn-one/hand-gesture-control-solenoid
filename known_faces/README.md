# Known Faces Directory

This directory serves as the database for storing photos of authorized users who are permitted to unlock the Smart Lock system. For privacy and efficiency reasons, image files (such as `.jpg` or `.png`) are ignored by Git (via `.gitignore`) and will not be pushed to the GitHub repository.

Therefore, when you first clone this repository, this folder will be empty (or only contain this `README.md` file). You **must** populate this folder with valid reference photos for the face recognition system to work.

## How to Add a New User

1. Prepare a clear, front-facing photo of the user.
2. Ensure there is **only one visible face** in the photo.
3. Rename the image file to the name of the user. This filename will act as the user's ID and will be displayed on the screen during operation (e.g., "Session Started: [Name]").
   - Example: `Ikhwan.jpg`, `Yuvina.png`, or `VIP Guest.jpeg`.
4. Place the renamed photo into this `known_faces/` directory.

## Additional Rules & Troubleshooting

* **Supported Formats:** `.jpg`, `.jpeg`, `.png`.
* If this directory is empty or contains no valid images when you run the system (`app/predict.py` or `app/gesture_control.py`), you will receive a terminal warning, and the system will be permanently blocked at the "SEARCHING_FACE" state (since no one is authorized).
* The system automatically scans the folder on startup. You simply need to restart the script after adding a new photo for the face to be registered.
