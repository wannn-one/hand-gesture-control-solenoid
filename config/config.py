ESP32_PORT = 'COM9' # port Arduino
BAUD_RATE = 9600 # baud rate untuk komunikasi serial
SERIAL_TIMEOUT = 1 # timeout untuk komunikasi serial
SERIAL_STARTUP_DELAY = 2 # delay untuk menunggu Arduino siap

MIN_DETECTION_CONFIDENCE = 0.7 # minimum confidence untuk deteksi tangan
MIN_TRACKING_CONFIDENCE = 0.5 # minimum confidence untuk tracking tangan

CAMERA_INDEX = 0 # indeks kamera default
WINDOW_TITLE = 'MediaPipe Hand Gesture Control' # judul window

COMMAND_OPEN = b'1' # perintah untuk membuka
COMMAND_CLOSE = b'0' # perintah untuk menutup

# --- Opsi Auto-Lock ---
AUTO_LOCK_SECONDS = 5  # waktu tunggu sebelum otomatis terkunci (detik)

# --- Opsi Kinerja ---
MAX_NUM_HANDS = 1  # batasi jumlah tangan yang diproses
MODEL_COMPLEXITY = 0  # 0 = lebih ringan/cepat, 1/2 = lebih akurat
FRAME_SCALE = 0.75  # skala frame untuk diproses (1.0 = asli)
PROCESS_EVERY_N_FRAMES = 1  # proses setiap N frame (1 = setiap frame)

# --- Opsi Kamera ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480