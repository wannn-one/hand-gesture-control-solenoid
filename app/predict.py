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
from typing import List, Tuple, Any, Optional

try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    LGPIO_AVAILABLE = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

# --- HELPER FUNCTIONS ---

def init_gpio() -> Optional[Any]:
    """Inisialisasi koneksi GPIO via lgpio bila berjalan di Raspberry Pi."""
    if not LGPIO_AVAILABLE:
        logger.warning("Library lgpio tidak tersedia. Mode simulasi GPIO.")
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

def buka_pintu(handle: Optional[Any]) -> None:
    """Mengaktifkan Solenoid agar pintu terbuka."""
    if handle is not None and LGPIO_AVAILABLE:
        try:
            logger.info("Membuka Pintu (Solenoid HIGH)...")
            lgpio.gpio_write(handle, config.SOLENOID_PIN, 1)
            time.sleep(config.SOLENOID_OPEN_SECONDS)
            lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)
            logger.info("Mengunci Pintu Kembali (Solenoid LOW).")
        except Exception as e:
            logger.error(f"Gagal kontrol GPIO (Buka): {e}")

def kunci_pintu(handle: Optional[Any]) -> None:
    """Menonaktifkan Solenoid agar pintu terkunci secara default."""
    if handle is not None and LGPIO_AVAILABLE:
        try:
            logger.info("Memastikan Pintu Terkunci (Solenoid LOW).")
            lgpio.gpio_write(handle, config.SOLENOID_PIN, 0)
        except Exception as e:
            logger.error(f"Gagal kontrol GPIO (Kunci): {e}")

def load_known_faces() -> Tuple[List[np.ndarray], List[str]]:
    """Memuat semua encoding wajah dari folder 'known_faces'."""
    known_encodings: List[np.ndarray] = []
    known_names: List[str] = []
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
                    known_names.append(os.path.splitext(file_name)[0])
                    logger.success(f"Berhasil memuat wajah: {file_name}")
                else:
                    logger.warning(f"Tidak ada wajah ditemukan di {file_name}")
            except Exception as e:
                logger.error(f"Error saat memuat {file_name}: {e}")

    logger.info(f"Total wajah dimuat: {len(known_encodings)}")
    return known_encodings, known_names

def ekstrak_keypoints(results: Any) -> np.ndarray:
    """Mengekstrak landmarks tangan untuk LSTM."""
    rh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    elif results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    return rh

def is_ok_gesture(hand_landmarks: Any) -> bool:
    """Deteksi gestur 'OK' sederhana (Jempol & Telunjuk menyatu)."""
    if not hand_landmarks: 
        return False
    tx, ty = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y
    ix, iy = hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y
    distance = np.sqrt((tx - ix)**2 + (ty - iy)**2)
    return distance < 0.05

# --- SYSTEM THREAD CLASSES ---

class CameraThread:
    """Thread mandiri untuk menampung input kamera terus menerus tanpa lag IO."""
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame: Optional[np.ndarray] = None
        self.running: bool = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        
    def update(self) -> None:
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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None
            
    def release(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

class GpioThread:
    """Thread mandiri untuk menjalankan Solenoid agar time.sleep tidak pause video."""
    def __init__(self, handle: Optional[Any]) -> None:
        self.handle = handle
        self.q: queue.Queue = queue.Queue()
        self.running: bool = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        
    def execute(self, command: str) -> None:
        self.q.put(command)
        
    def run(self) -> None:
        while self.running:
            try:
                cmd = self.q.get(timeout=0.1)
                if cmd == 'BUKA':
                    buka_pintu(self.handle)
                elif cmd == 'KUNCI':
                    kunci_pintu(self.handle)
            except queue.Empty:
                pass
                
    def stop(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class MLThread:
    """Thread yang khusus mengeksekusi sistem (Facerec, Mediapipe, TFLite LSTM)."""
    def __init__(self, camera_thread: CameraThread, gpio_thread: GpioThread, known_faces_data: Tuple[List[np.ndarray], List[str]], interpreter: tf.lite.Interpreter, input_details: List[dict], output_details: List[dict], label_encoder: Any, scaler: Any) -> None:
        self.camera = camera_thread
        self.gpio = gpio_thread
        self.known_face_encodings, self.known_face_names = known_faces_data
        
        self.interpreter = interpreter
        self.input_details = input_details
        self.output_details = output_details
        self.label_encoder = label_encoder
        self.scaler = scaler
        
        self.display_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running: bool = True
        
        # State Machine variables
        self.system_state: str = "SEARCHING_FACE"
        self.current_user: Optional[str] = None
        self.last_face_check_time: float = 0
        self.state_start_time: float = 0
        self.sequence: List[np.ndarray] = []
        
        self.ok_gesture_start_time: float = 0.0
        self.ok_gesture_duration_required: float = 2
        
        self.current_result_message: str = ""
        self.current_result_color: Tuple[int, int, int] = (0, 255, 0)
        
        self.face_search_interval: float = 0.3
        self.last_face_search_time: float = 0
        self.last_face_result: List[Any] = []
        
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def get_display_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.display_frame is not None:
                return self.display_frame.copy()
            return None

    def _draw_text(self, img: np.ndarray, text: str, pos: Tuple[int, int], font_scale: float = 0.7, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
        """Fungsi pembantu (DRY) untuk menggambar teks di layar."""
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _handle_searching_face(self, debug_image: np.ndarray, frame: np.ndarray, now: float) -> None:
        if now - self.last_face_search_time >= self.face_search_interval:
            self.last_face_search_time = now
            small_frame = cv2.resize(frame, (0, 0), fx=config.FACE_REC_SCALE, fy=config.FACE_REC_SCALE)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb_small_frame, model="hog")
            encs = face_recognition.face_encodings(rgb_small_frame, locs)
            self.last_face_result = list(zip(encs, locs))
        
        face_detected = False
        for face_encoding, face_location in self.last_face_result:
            name = "Unknown"
            if len(self.known_face_encodings) > 0:
                known_arr = np.array(self.known_face_encodings)
                distances = np.linalg.norm(known_arr - face_encoding, axis=1)
                best_idx = int(np.argmin(distances))
                
                if distances[best_idx] <= 0.6:
                    name = str(self.known_face_names[best_idx])
                    self.system_state = "SESSION_STANDBY"
                    self.current_user = name
                    self.last_face_check_time = now
                    self.last_face_result = []
                    logger.success(f"Sesi Dimulai: {name}")
                    face_detected = True
            
            top, right, bottom, left = face_location
            scale = 1 / config.FACE_REC_SCALE
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(debug_image, (int(left*scale), int(top*scale)), (int(right*scale), int(bottom*scale)), color, 2)
            self._draw_text(debug_image, name, (int(left*scale), int(top*scale)-10), 0.6, color)
            
            if face_detected: 
                break
        
        self._draw_text(debug_image, "SCAN WAJAH...", (10, 30), 0.8, (255, 255, 0))

    def _handle_session_standby(self, debug_image: np.ndarray, frame: np.ndarray, results: Any, now: float) -> None:
        if now - self.last_face_check_time > config.FACE_RECHECK_INTERVAL:
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
                        match_name = str(self.known_face_names[int(np.argmin(dist))])
                        if match_name == self.current_user:
                            user_still_here = True
                            break
            if user_still_here:
                self.last_face_check_time = now
            else:
                logger.warning("User hilang. Logout.")
                self.system_state = "SEARCHING_FACE"
                return
                
        trigger_active = False
        if results:
            if is_ok_gesture(results.left_hand_landmarks) or is_ok_gesture(results.right_hand_landmarks):
                trigger_active = True

        if trigger_active:
            if self.ok_gesture_start_time == 0:
                self.ok_gesture_start_time = now
            elif now - self.ok_gesture_start_time >= self.ok_gesture_duration_required:
                self.system_state = "SESSION_COUNTDOWN"
                self.state_start_time = now
                self.ok_gesture_start_time = 0
                logger.info("Trigger OK dideteksi (ditahan)! Memulai countdown.")
        else:
            self.ok_gesture_start_time = 0

        cv2.rectangle(debug_image, (0, 0), (640, 80), (50, 50, 50), -1)
        self._draw_text(debug_image, f"User: {self.current_user}", (10, 30))
        
        if self.ok_gesture_start_time > 0:
            hold_time = now - self.ok_gesture_start_time
            progress = min(1.0, hold_time / self.ok_gesture_duration_required)
            cv2.rectangle(debug_image, (10, 70), (int(10 + progress * 200), 80), (0, 255, 0), -1)
            self._draw_text(debug_image, "Tahan 'OK'...", (10, 60), 0.6, (0, 255, 0), 1)
        else:
            self._draw_text(debug_image, "Beri Pose 'OK' untuk Perintah", (10, 60), 0.6, (255, 255, 255), 1)

    def _handle_session_countdown(self, debug_image: np.ndarray, now: float) -> None:
        elapsed = now - self.state_start_time
        countdown = 3 - int(elapsed)
        
        if countdown <= 0:
            self.system_state = "SESSION_RECORDING"
            self.sequence = [] 
            logger.info("Mulai Merekam Gestur Dinamis...")
        
        self._draw_text(debug_image, str(countdown), (280, 240), 4, (0, 0, 255), 5)
        self._draw_text(debug_image, "Siapkan Gestur!", (180, 300), 0.7, (0, 0, 255))

    def _handle_session_recording(self, debug_image: np.ndarray, results: Any) -> None:
        if results:
            keypoints = ekstrak_keypoints(results)
            self.sequence.append(keypoints)
        
        cv2.circle(debug_image, (30, 30), 10, (0, 0, 255), -1)
        self._draw_text(debug_image, "REC", (50, 35), 0.6, (0, 255, 255))
        
        progress = len(self.sequence) / config.TIME_STEPS
        cv2.rectangle(debug_image, (0, 470), (int(progress*640), 480), (0, 0, 255), -1)

        if len(self.sequence) >= config.TIME_STEPS:
            self._predict_gesture()

    def _predict_gesture(self) -> None:
        """DRY: Fungsi khusus untuk menjalankan inferensi ML pada sequence."""
        input_data = np.array(self.sequence).astype(np.float32)
        
        if self.scaler is not None:
            input_data = self.scaler.transform(input_data)
        
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        res = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predicted_index = int(np.argmax(res[0]))
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

        self.system_state = "SESSION_RESULT_DISPLAY"
        self.state_start_time = time.time()  
        self.current_result_message = result_msg
        self.current_result_color = result_col
        self.sequence = []

    def _handle_session_result_display(self, debug_image: np.ndarray, now: float) -> None:
        elapsed = now - self.state_start_time
        self._draw_text(debug_image, self.current_result_message, (200, 240), 1, self.current_result_color, 3)
        if elapsed > 2.0: 
            self.system_state = "SESSION_STANDBY"

    def run(self) -> None:
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
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
                
                self._draw_text(debug_image, f"ML FPS: {int(fps)}", (480, 30), 0.7, (0, 255, 255))
                
                need_hand = self.system_state in ("SESSION_STANDBY", "SESSION_COUNTDOWN", "SESSION_RECORDING")
                results = None
                
                if need_hand:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = holistic.process(image_rgb)
                    
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(debug_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(debug_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # State Machine Delegation
                if self.system_state == "SEARCHING_FACE":
                    self._handle_searching_face(debug_image, frame, now)
                elif self.system_state == "SESSION_STANDBY":
                    self._handle_session_standby(debug_image, frame, results, now)
                elif self.system_state == "SESSION_COUNTDOWN":
                    self._handle_session_countdown(debug_image, now)
                elif self.system_state == "SESSION_RECORDING":
                    self._handle_session_recording(debug_image, results)
                elif self.system_state == "SESSION_RESULT_DISPLAY":
                    self._handle_session_result_display(debug_image, now)

                with self.lock:
                    self.display_frame = debug_image
                
                time.sleep(0.005)

    def stop(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

# --- APP CONTROLLER ---

class SmartLockApp:
    """Class utama penampung alur hidup Smart Lock System."""
    def __init__(self) -> None:
        self._setup_logging()
        self.gpio_handle = None
        self.camera_thread: Optional[CameraThread] = None
        self.gpio_thread: Optional[GpioThread] = None
        self.ml_thread: Optional[MLThread] = None

    def _setup_logging(self) -> None:
        if not os.path.exists("logs"):
            os.makedirs("logs")
        log_filename = f"logs/predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_filename, rotation="5 MB")
        logger.info("Sistem Smart Lock (Multithreading) Dimulai...")

    def setup(self) -> bool:
        """Menginisialisasi seluruh dependencies dan threads."""
        known_faces_data = load_known_faces()
        
        tflite_path = config.MODEL_LSTM_PATH
        if not os.path.exists(tflite_path):
            logger.error(f"Model TFLite tidak ditemukan di {tflite_path}")
            return False
            
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info("Berhasil memuat model TFLite untuk inference!")

        if not os.path.exists(config.LABEL_ENCODER_PATH):
            logger.error("Label encoder tidak ditemukan.")
            return False
            
        with open(config.LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)

        scaler = None
        if os.path.exists(config.SCALER_PATH):
            with open(config.SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("Berhasil memuat StandardScaler untuk normalisasi input!")
        else:
            logger.warning(f"StandardScaler tidak ditemukan di {config.SCALER_PATH}! Prediksi mungkin tidak akurat.")

        self.gpio_handle = init_gpio()
        
        logger.info("Inisialisasi Thread Kamera...")
        self.camera_thread = CameraThread(config.CAMERA_INDEX, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        
        logger.info("Inisialisasi Thread GPIO...")
        self.gpio_thread = GpioThread(self.gpio_handle)
        
        logger.info("Inisialisasi Thread Sistem ML...")
        self.ml_thread = MLThread(
            self.camera_thread, self.gpio_thread, known_faces_data, 
            interpreter, input_details, output_details, label_encoder, scaler
        )
        return True

    def run(self) -> None:
        """Menjalankan main loop UI thread."""
        logger.info("Semua thread berjalan. Memasuki loop antarmuka GUI (Thread Utama)...")
        if not self.ml_thread:
            return
            
        prev_ui_time = time.time()
        try:
            while True:
                display = self.ml_thread.get_display_frame()
                if display is not None:
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
            self.teardown()

    def teardown(self) -> None:
        """Menutup seluruh threads dengan aman."""
        logger.info("Menghentikan sistem ML...")
        if self.ml_thread:
            self.ml_thread.stop()
        logger.info("Menghentikan Camera...")
        if self.camera_thread:
            self.camera_thread.release()
        logger.info("Menghentikan GPIO...")
        if self.gpio_thread:
            self.gpio_thread.stop()
            
        if self.gpio_handle is not None and LGPIO_AVAILABLE:
            try:
                import lgpio
                kunci_pintu(self.gpio_handle)
                lgpio.gpiochip_close(self.gpio_handle)
            except Exception:
                pass
        
        cv2.destroyAllWindows()
        logger.info("Sistem Berhenti Secara Penuh.")

def main() -> None:
    app = SmartLockApp()
    if app.setup():
        app.run()

if __name__ == "__main__":
    main()
