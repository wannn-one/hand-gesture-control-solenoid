import cv2
import mediapipe as mp
import serial
import time

from config import config


def send_command(serial_port, command):
    """Mengirim perintah ke Arduino"""
    try:
        status = 'BUKA' if command == config.COMMAND_OPEN else 'KUNCI'
        print(f"Mengirim perintah: {status}")
        serial_port.write(command)
    except serial.SerialException as e:
        print(f"Peringatan: Gagal mengirim perintah ke Arduino: {e}")
    except Exception as e:
        print(f"Peringatan: Kesalahan tak terduga saat mengirim perintah: {e}")

def main():
    # Inisialisasi Serial untuk komunikasi dengan Arduino
    try:
        ser = serial.Serial(
            config.ESP32_PORT,
            config.BAUD_RATE,
            timeout=config.SERIAL_TIMEOUT,
        )
        print(f"Terhubung ke Arduino di port {config.ESP32_PORT}")
        time.sleep(config.SERIAL_STARTUP_DELAY)  # Beri waktu untuk koneksi stabil
    except serial.SerialException as e:
        print(
            f"Error: Tidak bisa membuka port serial {config.ESP32_PORT}. "
            f"Pastikan port benar dan tidak digunakan program lain."
        )
        print(e)
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    # Set resolusi kamera jika didukung
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    last_gesture = None
    frame_idx = 0
    # Timestamp terakhir ketika status OPEN_PALM terdeteksi
    unlocked_at = None

    try:
        with mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        ) as hands:
            while cap.isOpened():
                try:
                    success, image = cap.read()
                    if not success:
                        print("Gagal membuka kamera.")
                        break

                    image = cv2.flip(image, 1)

                    # Skipping frame untuk beban lebih ringan
                    if frame_idx % config.PROCESS_EVERY_N_FRAMES != 0:
                        # Tetap tampilkan frame terakhir dengan status
                        cv2.putText(
                            image,
                            f"Status: {last_gesture}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow(config.WINDOW_TITLE, image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        frame_idx += 1
                        continue

                    # Downscale untuk pemrosesan (opsional)
                    if 0 < config.FRAME_SCALE < 1.0:
                        proc_image = cv2.resize(image, (0, 0), fx=config.FRAME_SCALE, fy=config.FRAME_SCALE)
                    else:
                        proc_image = image

                    image_rgb = cv2.cvtColor(proc_image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)

                    current_gesture = None

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Gambar landmark pada gambar asli (sesuaikan skala jika perlu)
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )

                            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                            index_finger_tip = hand_landmarks.landmark[
                                mp_hands.HandLandmark.INDEX_FINGER_TIP
                            ]
                            middle_finger_tip = hand_landmarks.landmark[
                                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                            ]
                            thumb_tip = hand_landmarks.landmark[
                                mp_hands.HandLandmark.THUMB_TIP
                            ]

                            if (
                                index_finger_tip.y
                                < hand_landmarks.landmark[
                                    mp_hands.HandLandmark.INDEX_FINGER_PIP
                                ].y
                                and middle_finger_tip.y
                                < hand_landmarks.landmark[
                                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP
                                ].y
                            ):
                                current_gesture = "OPEN_PALM"
                            elif (
                                index_finger_tip.y
                                > hand_landmarks.landmark[
                                    mp_hands.HandLandmark.INDEX_FINGER_PIP
                                ].y
                            ):
                                current_gesture = "FIST"

                    # Refresh timer jika OPEN_PALM terdeteksi pada frame ini
                    if current_gesture == "OPEN_PALM":
                        unlocked_at = time.time()

                    if current_gesture != last_gesture:
                        if current_gesture == "OPEN_PALM":
                            send_command(ser, config.COMMAND_OPEN)
                            last_gesture = "OPEN_PALM"
                            # Mulai/ulang timer auto-lock saat dibuka
                            unlocked_at = time.time()
                        elif current_gesture == "FIST":
                            send_command(ser, config.COMMAND_CLOSE)
                            last_gesture = "FIST"

                    # Auto-lock: jika terakhir terbuka dan sudah melewati AUTO_LOCK_SECONDS
                    if last_gesture == "OPEN_PALM" and unlocked_at is not None:
                        if (time.time() - unlocked_at) >= config.AUTO_LOCK_SECONDS:
                            send_command(ser, config.COMMAND_CLOSE)
                            last_gesture = "AUTO_LOCKED"
                            unlocked_at = None

                    cv2.putText(
                        image,
                        f"Status: {last_gesture}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow(config.WINDOW_TITLE, image)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frame_idx += 1

                except Exception as e:
                    print(f"Peringatan: Terjadi error dalam loop utama: {e}")
                    # Lanjutkan ke frame berikutnya untuk menghindari crash total
                    frame_idx += 1
                    continue

    finally:
        try:
            send_command(ser, config.COMMAND_CLOSE)
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()