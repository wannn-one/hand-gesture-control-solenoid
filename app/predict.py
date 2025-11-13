import pickle
import time

import cv2
import lgpio  # Import the new lgpio library
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from config import config


def ekstrak_keypoints(results):
    # (Fungsi ini sama seperti sebelumnya, tidak perlu diubah)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)
    return rh


def init_gpio():
    handle = lgpio.gpiochip_open(config.GPIO_CHIP)
    lgpio.gpio_claim_output(handle, config.SOLENOID_PIN)
    lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)
    return handle


def buka_pintu(handle):
    print("GESTUR BUKA KUNCI TERDETEKSI -> Membuka solenoid...")
    lgpio.gpio_write(handle, config.SOLENOID_PIN, 1)  # Kirim sinyal HIGH (1)
    time.sleep(config.SOLENOID_OPEN_SECONDS)  # Biarkan pintu terbuka
    lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)  # Kirim sinyal LOW (0)
    print("Solenoid terkunci kembali.")


def kunci_pintu(handle):
    print("GESTUR KUNCI TERDETEKSI -> Memastikan solenoid terkunci.")
    lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)  # Kirim sinyal LOW (0)


def main():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan di {config.MODEL_PATH}. Jalankan training terlebih dahulu.")
    if not config.ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder tidak ditemukan di {config.ENCODER_PATH}. Jalankan training terlebih dahulu.")

    print(f"Memuat model dari {config.MODEL_PATH}...")
    model = load_model(config.MODEL_PATH)
    with config.ENCODER_PATH.open('rb') as f:
        label_encoder = pickle.load(f)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Gagal membuka kamera.")

    gpio_handle = None
    try:
        gpio_handle = init_gpio()
        with mp_holistic.Holistic(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                keypoints = ekstrak_keypoints(results)
                prediction = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)  # verbose=0 to clean up output
                confidence = float(np.max(prediction))

                if confidence > config.PREDICTION_CONFIDENCE_THRESHOLD:
                    predicted_index = int(np.argmax(prediction))
                    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

                    cv2.putText(
                        image,
                        f'GESTUR: {predicted_label.upper()}',
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # --- LOGIKA KONTROL GPIO ---
                    if predicted_label == config.GESTURE_OPEN:
                        buka_pintu(gpio_handle)
                        time.sleep(config.OPEN_COOLDOWN_SECONDS)  # Add delay to prevent re-triggering
                    elif predicted_label == config.GESTURE_CLOSE:
                        kunci_pintu(gpio_handle)
                        time.sleep(config.LOCK_COOLDOWN_SECONDS)

                cv2.imshow(config.WINDOW_TITLE, image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    finally:
        print("Membersihkan dan menutup handle GPIO...")
        if gpio_handle is not None:
            try:
                kunci_pintu(gpio_handle)
            except Exception:
                pass
            try:
                lgpio.gpiochip_close(gpio_handle)  # Close the handle
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()