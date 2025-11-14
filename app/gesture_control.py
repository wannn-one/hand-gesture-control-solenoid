import cv2
import mediapipe as mp
import serial
import time
import os
import face_recognition

from loguru import logger
from config import config

def send_command(serial_port, command):
    """Mengirim perintah ke Arduino"""
    try:
        status = 'BUKA' if command == config.COMMAND_OPEN else 'KUNCI'
        logger.info(f"Mengirim perintah: {status}")
        serial_port.write(command)
    except serial.SerialException as e:
        logger.warning(f"Gagal mengirim perintah ke Arduino: {e}")
    except Exception as e:
        logger.warning(f"Kesalahan tak terduga saat mengirim perintah: {e}")

def load_known_faces():
    """Memuat semua encoding wajah dari folder 'known_faces'."""
    known_encodings = []
    known_names = []
    logger.info("Memuat wajah master...")
    
    if not os.path.exists(config.KNOWN_FACES_DIR):
        logger.error(f"Folder 'known_faces' tidak ditemukan di {config.KNOWN_FACES_DIR}")
        logger.info("Silakan buat folder 'known_faces' dan tambahkan foto Anda.")
        return [], []

    for file_name in os.listdir(config.KNOWN_FACES_DIR):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            try:
                image_path = os.path.join(config.KNOWN_FACES_DIR, file_name)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(file_name)[0])
                    logger.success(f"Berhasil memuat wajah: {file_name}")
                else:
                    logger.warning(f"Tidak ada wajah ditemukan di {file_name}")
            except Exception as e:
                logger.error(f"Error saat memuat {file_name}: {e}")

    if not known_encodings:
        logger.warning("Tidak ada wajah master yang berhasil dimuat.")
        
    logger.info(f"Total wajah dimuat: {len(known_encodings)}")
    return known_encodings, known_names

def main():
    # --- 1. Inisialisasi Wajah ---
    known_face_encodings, known_face_names = load_known_faces()
    if not known_face_encodings:
        logger.error("Tidak ada wajah master untuk dibandingkan. Program berhenti.")
        return

    # --- 2. Inisialisasi Serial ---
    ser = None
    try:
        ser = serial.Serial(
            config.ESP32_PORT,
            config.BAUD_RATE,
            timeout=config.SERIAL_TIMEOUT,
        )
        logger.info(f"Terhubung ke Arduino di port {config.ESP32_PORT}")
        time.sleep(config.SERIAL_STARTUP_DELAY)
    except serial.SerialException as e:
        logger.error(f"Tidak bisa membuka port serial {config.ESP32_PORT}.")
        logger.exception(e)
        return

    # --- 3. Inisialisasi Kamera & MediaPipe ---
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # --- 4. Inisialisasi Variabel Status (State Machine) ---
    system_state = "SEARCHING_FACE"
    state_changed_at = time.time()
    last_gesture = None
    current_gesture = None # <-- [ MODIFIKASI ] Pindahkan inisialisasi ke sini
    last_known_name = None
    unlocked_at = None

    try:
        with mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            model_complexity=config.MODEL_COMPLEXITY,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        ) as hands:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    logger.warning("Gagal membuka kamera.")
                    break
                
                image = cv2.flip(image, 1)

                # --- START STATE MACHINE ---

                if system_state == "SEARCHING_FACE":
                    # --- MODE 1: MENCARI WAJAH ---
                    small_frame = cv2.resize(image, (0, 0), fx=config.FACE_REC_SCALE, fy=config.FACE_REC_SCALE)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_name = "Unknown"
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        
                        if True in matches:
                            first_match_index = matches.index(True)
                            face_name = known_face_names[first_match_index]
                            
                            logger.success(f"Wajah dikenali: {face_name}")
                            system_state = "WAITING_FOR_GESTURE"
                            state_changed_at = time.time()
                            last_known_name = face_name
                            
                            top, right, bottom, left = face_location
                            top = int(top / config.FACE_REC_SCALE)
                            right = int(right / config.FACE_REC_SCALE)
                            bottom = int(bottom / config.FACE_REC_SCALE)
                            left = int(left / config.FACE_REC_SCALE)
                            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(image, face_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            break
                    
                    cv2.putText(image, "Status: Mencari Wajah", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                elif system_state == "WAITING_FOR_GESTURE":
                    # --- MODE 2: MENUNGGU GESTUR ---
                    time_elapsed = time.time() - state_changed_at
                    
                    # 1. Cek Timeout
                    if time_elapsed > config.FACE_REC_TIMEOUT:
                        logger.info("Waktu habis. Kembali ke mode pencarian wajah.")
                        system_state = "SEARCHING_FACE"
                        last_gesture = None
                        current_gesture = None
                        if unlocked_at:
                            send_command(ser, config.COMMAND_CLOSE)
                            unlocked_at = None
                        continue

                    # 2. Jalankan Deteksi Tangan
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    current_gesture = None

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                            middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

                            if (index_finger_tip.y < index_finger_pip.y and middle_finger_tip.y < middle_finger_pip.y):
                                current_gesture = "OPEN_PALM"
                            elif (index_finger_tip.y > index_finger_pip.y):
                                current_gesture = "FIST"

                    # 3. Proses Aksi Gestur
                    if current_gesture != last_gesture and current_gesture is not None:
                        if current_gesture == "OPEN_PALM":
                            send_command(ser, config.COMMAND_OPEN)
                            logger.info("Aksi: BUKA. Kembali ke mode pencarian wajah.")
                            unlocked_at = time.time()
                            system_state = "SEARCHING_FACE"
                        elif current_gesture == "FIST":
                            send_command(ser, config.COMMAND_CLOSE)
                            logger.info("Aksi: KUNCI. Kembali ke mode pencarian wajah.")
                            unlocked_at = None
                            system_state = "SEARCHING_FACE"
                        
                        last_gesture = current_gesture
                        continue
                    
                    # Tampilkan status
                    remaining_time = int(config.FACE_REC_TIMEOUT - time_elapsed)
                    cv2.putText(image, f"Wajah: {last_known_name} | Beri Gestur ({remaining_time}s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Tampilkan status GESTUR SAAT INI
                    gesture_text = f"Gestur: {current_gesture if current_gesture else '...'}"
                    cv2.putText(image, gesture_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 2) 
                    
                    last_gesture = current_gesture

                # --- END STATE MACHINE ---

                # Tampilkan gambar
                cv2.imshow(config.WINDOW_TITLE, image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        logger.info("Menutup program...")
        if ser:
            try:
                send_command(ser, config.COMMAND_CLOSE)
                ser.close()
                logger.info("Koneksi serial ditutup.")
            except Exception as e:
                logger.error(f"Error saat menutup serial: {e}")
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Program selesai.")

if __name__ == "__main__":
    main()