import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
import pickle
import face_recognition
import tensorflow as tf
from datetime import datetime
from loguru import logger
import threading
import queue

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False

# Tambahkan root directory ke path agar folder config dikenali
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
        os.makedirs(config.KNOWN_FACES_DIR)
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
                    known_names.append(os.path.splitext(file_name))
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
    
    # Ambil index ke-4 (Ujung Jempol / Thumb Tip)
    tx = hand_landmarks.landmark[4].x
    ty = hand_landmarks.landmark[4].y
    
    # Ambil index ke-8 (Ujung Telunjuk / Index Finger Tip)
    ix = hand_landmarks.landmark[8].x
    iy = hand_landmarks.landmark[8].y
    
    # Hitung jarak Euclidean
    distance = np.sqrt((tx - ix)**2 + (ty - iy)**2)
    return distance < 0.05

# SYSTEM THREAD CLASS

class CameraThread:
    """Thread mandiri untuk menampung input kamera terus menerus tanpa lag IO"""
    def __init__(self, camera_index, width, height):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None
            
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

class GpioThread:
    """Thread mandiri untuk menjalankan Solenoid agar time.sleep tidak pause video"""
    def __init__(self, handle):
        self.handle = handle
        self.q = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        
    def execute(self, command):
        self.q.put(command)
        
    def run(self):
        while self.running:
            try:
                cmd = self.q.get(timeout=0.1)
                if cmd == 'BUKA':
                    buka_pintu(self.handle)
                elif cmd == 'KUNCI':
                    kunci_pintu(self.handle)
            except queue.Empty:
                pass
                
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class MLThread:
    """Thread yang khusus mengeksekusi sistem (Facerec, Mediapipe, TFLite LSTM), beroperasi terpisah dari antarmuka GUI"""
    def __init__(self, camera_thread, gpio_thread, known_faces_data, interpreter, input_details, output_details, label_encoder, scaler):
        self.camera = camera_thread
        self.gpio = gpio_thread
        self.known_face_encodings, self.known_face_names = known_faces_data
        
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        self.label_encoder = label_encoder
        self.scaler = scaler
        
        self.display_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def get_display_frame(self):
        with self.lock:
            if self.display_frame is not None:
                return self.display_frame.copy()
            return None

    def run(self):
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        
        system_state = "SEARCHING_FACE"
        current_user = None
        last_face_check_time = 0
        state_start_time = 0
        sequence = []
        
        current_result_message = ""
        current_result_color = (0, 255, 0)
        
        # Throttle Face Recognition
        FACE_SEARCH_INTERVAL = 0.3
        last_face_search_time = 0
        last_face_result = []

        prev_frame_time = time.time()
        
        with mp_holistic.Holistic(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        ) as holistic:
            while self.running:
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                debug_image = frame
                now = time.time()
                fps = 1.0 / (now - prev_frame_time) if (now - prev_frame_time) > 0 else 0
                prev_frame_time = now
                
                cv2.putText(debug_image, f"ML FPS: {int(fps)}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Rule 2 & 3: Hanya jalankan Model Gestur (Mediapipe) pada state membutuhkan user input
                need_hand = system_state in ("SESSION_STANDBY", "SESSION_COUNTDOWN", "SESSION_RECORDING")
                results = None
                if need_hand:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = holistic.process(image_rgb)
                    
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(debug_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(debug_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # --- STATE MACHINE RULES ---
                if system_state == "SEARCHING_FACE":
                    # Rule 1: Hanya menjalankan face recog saat mencari muka
                    if now - last_face_search_time >= FACE_SEARCH_INTERVAL:
                        last_face_search_time = now
                        small_frame = cv2.resize(frame, (0, 0), fx=config.FACE_REC_SCALE, fy=config.FACE_REC_SCALE)
                        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        locs = face_recognition.face_locations(rgb_small_frame, model="hog")
                        encs = face_recognition.face_encodings(rgb_small_frame, locs)
                        last_face_result = list(zip(encs, locs))
                    
                    face_detected = False
                    for face_encoding, face_location in last_face_result:
                        name = "Unknown"
                        if len(self.known_face_encodings) > 0:
                            known_arr = np.array(self.known_face_encodings)
                            distances = np.linalg.norm(known_arr - face_encoding, axis=1)
                            best_idx = np.argmin(distances)
                            
                            if distances[best_idx] <= 0.6:
                                name = str(self.known_face_names[best_idx])
                                system_state = "SESSION_STANDBY"
                                current_user = name
                                last_face_check_time = now
                                last_face_result = []
                                logger.success(f"Sesi Dimulai: {name}")
                                face_detected = True
                        
                        top, right, bottom, left = face_location
                        scale = 1/config.FACE_REC_SCALE
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(debug_image, (int(left*scale), int(top*scale)), (int(right*scale), int(bottom*scale)), color, 2)
                        cv2.putText(debug_image, name, (int(left*scale), int(top*scale)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        if face_detected: break
                    
                    cv2.putText(debug_image, "SCAN WAJAH...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                elif system_state == "SESSION_STANDBY":
                    # Sesekali memastikan orangnya masih ada
                    if now - last_face_check_time > config.FACE_RECHECK_INTERVAL:
                        small_frame = cv2.resize(frame, (0, 0), fx=config.FACE_REC_SCALE, fy=config.FACE_REC_SCALE)
                        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        locs = face_recognition.face_locations(rgb_small_frame)
                        encs = face_recognition.face_encodings(rgb_small_frame, locs)
                        
                        user_still_here = False
                        if len(self.known_face_encodings) > 0:
                            known_arr = np.array(self.known_face_encodings)
                            for enc in encs:
                                dist = np.linalg.norm(known_arr - enc, axis=1)
                                if np.min(dist) <= 0.6:
                                    match_name = str(self.known_face_names[np.argmin(dist)])
                                    if match_name == current_user:
                                        user_still_here = True
                                        break
                        if user_still_here:
                            last_face_check_time = now
                        else:
                            logger.warning("User hilang. Logout.")
                            system_state = "SEARCHING_FACE"
                            continue
                            
                    trigger_active = False
                    if results and is_ok_gesture(results.left_hand_landmarks):
                        trigger_active = True

                    if trigger_active:
                        system_state = "SESSION_COUNTDOWN"
                        state_start_time = now
                        logger.info("Trigger OK dideteksi! Memulai countdown.")

                    cv2.rectangle(debug_image, (0, 0), (640, 80), (50, 50, 50), -1)
                    cv2.putText(debug_image, f"User: {current_user}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(debug_image, "Beri Pose 'OK' untuk Perintah", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                elif system_state == "SESSION_COUNTDOWN":
                    elapsed = now - state_start_time
                    countdown = 3 - int(elapsed)
                    
                    if countdown <= 0:
                        system_state = "SESSION_RECORDING"
                        sequence = [] 
                        logger.info("Mulai Merekam Gestur Dinamis...")
                    
                    cv2.putText(debug_image, str(countdown), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5)
                    cv2.putText(debug_image, "Siapkan Gestur!", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif system_state == "SESSION_RECORDING":
                    if results:
                        keypoints = ekstrak_keypoints(results)
                        sequence.append(keypoints)
                    
                    cv2.circle(debug_image, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(debug_image, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    progress = len(sequence) / config.TIME_STEPS
                    cv2.rectangle(debug_image, (0, 470), (int(progress*640), 480), (0, 0, 255), -1)

                    # Rule 3: Menjalankan model TFLite HANYA ketika buffer data (sequence) telah siap
                    if len(sequence) >= config.TIME_STEPS:
                        input_data = np.array(sequence).astype(np.float32)  # shape: (TIME_STEPS, 63)
                        
                        # Normalisasi dengan StandardScaler (sama seperti saat training)
                        if self.scaler is not None:
                            input_data = self.scaler.transform(input_data)
                        
                        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)  # shape: (1, TIME_STEPS, 63)
                        
                        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                        self.interpreter.invoke()
                        res = self.interpreter.get_tensor(self.output_details[0]['index'])
                        
                        predicted_index = np.argmax(res[0])
                        confidence = float(res[0][predicted_index])
                        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]
                        
                        logger.info(f"Hasil Prediksi: {predicted_label} ({confidence:.2f})")
                        
                        result_msg = "?"
                        result_col = (0, 255, 255)

                        if confidence > config.LSTM_THRESHOLD:
                            if predicted_label == 'buka_kunci':
                                logger.success("AKSI: BUKA KUNCI")
                                self.gpio.execute('BUKA')
                                result_msg = "AKSI: BUKA KUNCI"
                                result_col = (0, 255, 0)
                            elif predicted_label == 'kunci':
                                logger.success("AKSI: TUTUP KUNCI")
                                self.gpio.execute('KUNCI')
                                result_msg = "AKSI: TUTUP KUNCI"
                                result_col = (0, 255, 0)
                        else:
                            logger.warning("Gestur tidak dikenali / kurang yakin.")

                        system_state = "SESSION_RESULT_DISPLAY"
                        state_start_time = time.time()  
                        current_result_message = result_msg
                        current_result_color = result_col
                        sequence = []

                elif system_state == "SESSION_RESULT_DISPLAY":
                    elapsed = now - state_start_time
                    cv2.putText(debug_image, current_result_message, (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, current_result_color, 3)
                    
                    if elapsed > 2.0: 
                        system_state = "SESSION_STANDBY"

                with self.lock:
                    self.display_frame = debug_image
                
                time.sleep(0.005)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

def main():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_filename = f"logs/predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_filename, rotation="5 MB")
    logger.info("Sistem Smart Lock (Multithreading) Dimulai...")

    known_faces_data = load_known_faces()
    
    tflite_path = config.MODEL_LSTM_PATH
    if not os.path.exists(tflite_path):
        logger.error(f"Model TFLite tidak ditemukan di {tflite_path}")
        return
        
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Berhasil memuat model TFLite untuk inference!")

    if not os.path.exists(config.LABEL_ENCODER_PATH):
        logger.error("Label encoder tidak ditemukan.")
        return
    with open(config.LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)

    # Memuat StandardScaler untuk normalisasi input (WAJIB sama dengan saat training)
    scaler = None
    if os.path.exists(config.SCALER_PATH):
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Berhasil memuat StandardScaler untuk normalisasi input!")
    else:
        logger.warning(f"StandardScaler tidak ditemukan di {config.SCALER_PATH}! Prediksi mungkin tidak akurat.")

    gpio_handle = init_gpio()
    
    logger.info("Inisialisasi Thread Kamera...")
    camera_thread = CameraThread(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
    
    logger.info("Inisialisasi Thread GPIO...")
    gpio_thread = GpioThread(gpio_handle)
    
    logger.info("Inisialisasi Thread Sistem ML...")
    ml_thread = MLThread(
        camera_thread, gpio_thread, known_faces_data, 
        interpreter, input_details, output_details, label_encoder, scaler
    )
    
    logger.info("Semua thread berjalan. Memasuki loop antarmuka GUI Murni (Thread Utama)...")
    
    prev_ui_time = time.time()
    try:
        while True:
            display = ml_thread.get_display_frame()
            if display is not None:
                # Menghitung UI Frame rate
                now = time.time()
                ui_fps = 1.0 / (now - prev_ui_time) if (now - prev_ui_time) > 0 else 0
                prev_ui_time = now
                cv2.putText(display, f"UI FPS: {int(ui_fps)}", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.imshow(config.WINDOW_TITLE, display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Keluar ditekan (q)...")
                break
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("Dihentikan manual.")
    except Exception as e:
        logger.exception(f"Terjadi crash di loop antarmuka: {e}")
    finally:
        logger.info("Menghentikan sistem ML...")
        ml_thread.stop()
        logger.info("Menghentikan Camera...")
        camera_thread.release()
        logger.info("Menghentikan GPIO...")
        gpio_thread.stop()
        
        if gpio_handle is not None and LGPIO_AVAILABLE:
            try:
                kunci_pintu(gpio_handle)
                lgpio.gpiochip_close(gpio_handle)
            except Exception:
                pass
        
        cv2.destroyAllWindows()
        logger.info("Sistem Berhenti Secara Penuh.")

if __name__ == "__main__":
    main()
