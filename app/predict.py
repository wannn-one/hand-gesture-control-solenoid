import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False

# Tambahkan root directory ke path agar folder config dikenali
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import face_recognition
import pickle
from datetime import datetime
from keras.models import load_model
from loguru import logger
from config import config

# --- FUNGSI BANTUAN ---

def init_gpio():
    if not LGPIO_AVAILABLE:
        logger.warning("Library lgpio tidak tersedia (mungkin di Windows). Mode simulasi GPIO.")
        return None
    try:
        handle = lgpio.gpiochip_open(config.GPIO_CHIP)
        lgpio.gpio_claim_output(handle, config.SOLENOID_PIN)
        lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)
        logger.info(f"Pin Solenoid {config.SOLENOID_PIN} (GPIO) Siap!")
        return handle
    except Exception as e:
        logger.warning(f"Gagal inisialisasi GPIO: {e}. Mode simulasi.")
        return None

def buka_pintu(handle):
    if handle is not None and LGPIO_AVAILABLE:
        try:
            logger.info("Membuka Pintu (Solenoid HIGH)...")
            lgpio.gpio_write(handle, config.SOLENOID_PIN, 1)
            # Blokir script sesaat untuk memberikan jeda relay listrik (opsional, sesuaikan dengan hardware asli)
            time.sleep(config.SOLENOID_OPEN_SECONDS)
            lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)
            logger.info("Mengunci Pintu Kembali (Solenoid LOW).")
        except Exception as e:
            logger.error(f"Gagal kontrol GPIO (Buka): {e}")

def kunci_pintu(handle):
    if handle is not None and LGPIO_AVAILABLE:
        try:
            logger.info("Memastikan Pintu Terkunci (Solenoid LOW).")
            lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)
        except Exception as e:
            logger.error(f"Gagal kontrol GPIO (Kunci): {e}")

def load_known_faces():
    """Memuat semua encoding wajah dari folder 'known_faces'."""
    known_encodings = []
    known_names = []
    logger.info("Memuat wajah master...")
    
    if not os.path.exists(config.KNOWN_FACES_DIR):
        os.makedirs(config.KNOWN_FACES_DIR) # Buat folder jika belum ada
        logger.warning(f"Folder '{config.KNOWN_FACES_DIR}' baru saja dibuat. Silakan isi dengan foto wajah Anda.")
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

    logger.info(f"Total wajah dimuat: {len(known_encodings)}")
    return known_encodings, known_names

def ekstrak_keypoints(results):
    """Mengekstrak landmarks tangan untuk LSTM."""
    rh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    elif results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    return rh

def is_ok_gesture(hand_landmarks):
    """Deteksi gestur 'OK' sederhana (Jempol & Telunjuk menyatu)."""
    if not hand_landmarks: return False
    
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    
    # Hitung jarak Euclidean
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    return distance < 0.05 # Threshold jarak dekat

def main():
    # --- 1. SETUP LOGGING ---
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_filename = f"logs/predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_filename, rotation="5 MB")
    logger.info("Sistem Smart Lock (LGPIO) Dimulai...")

    # --- 2. LOAD RESOURCES ---
    known_face_encodings, known_face_names = load_known_faces()
    
    if not os.path.exists(config.MODEL_LSTM_PATH):
        logger.error(f"Model LSTM tidak ditemukan di {config.MODEL_LSTM_PATH}")
        return
    model_lstm = load_model(config.MODEL_LSTM_PATH, compile=False)

    if not os.path.exists(config.LABEL_ENCODER_PATH):
        logger.error("Label encoder tidak ditemukan.")
        return
    with open(config.LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    # --- 3. SETUP GPIO LGPIO ---
    gpio_handle = init_gpio()

    # --- 4. SETUP KAMERA & MEDIAPIPE ---
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    # --- 5. STATE MACHINE VARIABLES ---
    # States: SEARCHING_FACE -> SESSION_STANDBY -> SESSION_COUNTDOWN -> SESSION_RECORDING
    system_state = "SEARCHING_FACE" 
    
    current_user = None
    last_face_check_time = 0
    state_start_time = 0
    
    sequence = [] 
    current_result_message = ""
    current_result_color = (0, 255, 0)
    
    try:
        with mp_holistic.Holistic(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE, 
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        ) as holistic:
            
            prev_frame_time = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                # Mirror frame untuk UX yang natural
                frame = cv2.flip(frame, 1)
                debug_image = frame.copy()
                
                # Waktu sekarang
                now = time.time()
                fps = 1 / (now - prev_frame_time) if prev_frame_time > 0 else 0
                prev_frame_time = now
                
                # Tampilkan FPS di pojok kanan atas
                cv2.putText(debug_image, f"FPS: {int(fps)}", (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # --- PROSES UMUM (Wajah & Holistic) ---
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # Gambar Skeleton (Visualisasi)
                mp_drawing.draw_landmarks(debug_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
                mp_drawing.draw_landmarks(debug_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # --- LOGIKA STATE MACHINE ---

                if system_state == "SEARCHING_FACE":
                    # Scan Wajah
                    small_frame = cv2.resize(frame, (0, 0), fx=config.FACE_REC_SCALE, fy=config.FACE_REC_SCALE)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_detected = False
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"
                        if True in matches:
                            name = known_face_names[matches.index(True)]
                            system_state = "SESSION_STANDBY"
                            current_user = name
                            last_face_check_time = now
                            logger.success(f"Sesi Dimulai: {name}")
                            face_detected = True
                        
                        # Visualisasi
                        top, right, bottom, left = face_location
                        scale = 1/config.FACE_REC_SCALE
                        cv2.rectangle(debug_image, (int(left*scale), int(top*scale)), (int(right*scale), int(bottom*scale)), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)
                        cv2.putText(debug_image, name, (int(left*scale), int(top*scale)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        if face_detected: break

                    cv2.putText(debug_image, "SCAN WAJAH...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                elif system_state == "SESSION_STANDBY":
                    # Cek Wajah Berkala (Setiap 10 detik)
                    if now - last_face_check_time > config.FACE_RECHECK_INTERVAL:
                        small_frame = cv2.resize(frame, (0, 0), fx=config.FACE_REC_SCALE, fy=config.FACE_REC_SCALE)
                        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_small_frame)
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        
                        user_still_here = False
                        for face_encoding in face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            if True in matches and known_face_names[matches.index(True)] == current_user:
                                user_still_here = True
                                break
                        
                        if user_still_here:
                            last_face_check_time = now
                        else:
                            logger.warning("User hilang. Logout.")
                            system_state = "SEARCHING_FACE"
                            continue

                    # Cek Trigger: Pose "OK" di tangan manapun
                    trigger_active = False
                    if is_ok_gesture(results.left_hand_landmarks) or is_ok_gesture(results.right_hand_landmarks):
                        trigger_active = True
                    
                    if trigger_active:
                        system_state = "SESSION_COUNTDOWN"
                        state_start_time = now
                        logger.info("Trigger OK dideteksi! Memulai countdown.")

                    # UI Standby
                    cv2.rectangle(debug_image, (0, 0), (640, 80), (50, 50, 50), -1)
                    cv2.putText(debug_image, f"User: {current_user}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(debug_image, "Beri Pose 'OK' untuk Perintah", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                elif system_state == "SESSION_COUNTDOWN":
                    elapsed = now - state_start_time
                    countdown = 3 - int(elapsed)
                    
                    if countdown <= 0:
                        system_state = "SESSION_RECORDING"
                        sequence = [] # Reset buffer
                        logger.info("Mulai Merekam Gestur Dinamis...")
                    
                    # UI Countdown Besar
                    cv2.putText(debug_image, str(countdown), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
                    cv2.putText(debug_image, "Siapkan Gestur Kanan!", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif system_state == "SESSION_RECORDING":
                    # Rekam Gestur Dinamis (Tangan User)
                    keypoints = ekstrak_keypoints(results)
                    sequence.append(keypoints)
                    
                    # UI Recording
                    cv2.circle(debug_image, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(debug_image, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Progress Bar
                    progress = len(sequence) / config.TIME_STEPS
                    cv2.rectangle(debug_image, (0, 470), (int(progress*640), 480), (0, 0, 255), -1)

                    if len(sequence) >= config.TIME_STEPS:
                        # Prediksi Asli
                        res = model_lstm.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        
                        predicted_index = np.argmax(res)
                        confidence = res[predicted_index]
                        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
                        
                        logger.info(f"Hasil Prediksi: {predicted_label} ({confidence:.2f})")
                        
                        result_message = "?"
                        result_color = (0, 255, 255)

                        if confidence > config.LSTM_THRESHOLD:
                            if predicted_label == 'buka_kunci':
                                logger.success("AKSI: BUKA KUNCI")
                                buka_pintu(gpio_handle)
                                result_message = "AKSI: BUKA KUNCI"
                                result_color = (0, 255, 0)
                            elif predicted_label == 'kunci':
                                logger.success("AKSI: TUTUP KUNCI")
                                kunci_pintu(gpio_handle)
                                result_message = "AKSI: TUTUP KUNCI"
                                result_color = (0, 255, 0)
                        else:
                            logger.warning("Gestur tidak dikenali / kurang yakin.")
                            result_message = "?"
                            result_color = (0, 255, 255)

                        # Masuk ke State Result Display (Non-blocking wait)
                        system_state = "SESSION_RESULT_DISPLAY"
                        state_start_time = time.time()  # Reset now for display
                        current_result_message = result_message
                        current_result_color = result_color
                        sequence = []

                elif system_state == "SESSION_RESULT_DISPLAY":
                    # Tampilkan hasil selama 2 detik
                    elapsed = now - state_start_time
                    
                    cv2.putText(debug_image, current_result_message, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, current_result_color, 3)
                    
                    if elapsed > 2.0: # Tahan 2 detik sebelum standby
                        system_state = "SESSION_STANDBY"

                # Tampilkan Frame
                cv2.imshow(config.WINDOW_TITLE, debug_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        logger.exception(f"Terjadi crash: {e}")
    finally:
        if 'gpio_handle' in locals() and gpio_handle is not None and LGPIO_AVAILABLE:
            try:
                kunci_pintu(gpio_handle)
                lgpio.gpiochip_close(gpio_handle)
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Sistem Berhenti.")

if __name__ == "__main__":
    main()