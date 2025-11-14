from pathlib import Path

# --- Jalur Proyek ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'model_gestur.h5'
ENCODER_PATH = ARTIFACTS_DIR / 'label_encoder.pkl'

# --- Data Capture ---
DATASET_DIR = DATA_DIR / 'Dataset_Gestur'
CSV_FILE_PATH = DATA_DIR / 'dataset_landmarks.csv'
GESTURE_OPEN = 'buka_kunci'
GESTURE_CLOSE = 'kunci'
ACTIONS = (GESTURE_OPEN, GESTURE_CLOSE)
NUM_SEQUENCES = 20
FRAMES_PER_SEQUENCE = 20
RECORD_COUNTDOWN_SECONDS = 3
DATA_CAPTURE_CAMERA_INDEX = 1

# --- Training ---
TRAIN_EPOCHS = 50
TRAIN_BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.2
TRAIN_RANDOM_STATE = 42
TRAIN_STRATIFY = True

# --- Perangkat/Kamera Umum ---
CAMERA_INDEX = 0  # indeks kamera default untuk aplikasi
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_TITLE = 'MediaPipe Hand Gesture Control'  # judul window

# --- MediaPipe ---
MIN_DETECTION_CONFIDENCE = 0.7  # minimum confidence untuk deteksi tangan
MIN_TRACKING_CONFIDENCE = 0.5  # minimum confidence untuk tracking tangan
MAX_NUM_HANDS = 1  # batasi jumlah tangan yang diproses
MODEL_COMPLEXITY = 0  # 0 = lebih ringan/cepat, 1/2 = lebih akurat
FRAME_SCALE = 0.75  # skala frame untuk diproses (1.0 = asli)
PROCESS_EVERY_N_FRAMES = 1  # proses setiap N frame (1 = setiap frame)

# --- Serial / ESP32 ---
ESP32_PORT = 'COM9'  # port Arduino
BAUD_RATE = 9600  # baud rate untuk komunikasi serial
SERIAL_TIMEOUT = 1  # timeout untuk komunikasi serial
SERIAL_STARTUP_DELAY = 2  # delay untuk menunggu Arduino siap
COMMAND_OPEN = b'1'  # perintah untuk membuka
COMMAND_CLOSE = b'0'  # perintah untuk menutup

# --- Kontrol Solenoid / Auto-Lock ---
AUTO_LOCK_SECONDS = 5  # waktu tunggu sebelum otomatis terkunci (detik)
SOLENOID_PIN = 17  # GPIO pin BCM 17 (Raspberry Pi)
GPIO_CHIP = 0  # Default GPIO chip untuk Raspberry Pi
SOLENOID_OPEN_SECONDS = 3
OPEN_COOLDOWN_SECONDS = 5
LOCK_COOLDOWN_SECONDS = 2

# --- Inferensi ---
PREDICTION_CONFIDENCE_THRESHOLD = 0.90

# --- Face Recognition ---
KNOWN_FACES_DIR = ROOT_DIR / 'known_faces' # Folder untuk foto master
FACE_REC_TIMEOUT = 10 # Detik untuk memberi gestur setelah wajah dikenali
FACE_REC_SCALE = 0.5 # Skala untuk mempercepat deteksi (0.5 = 50% lebih kecil)