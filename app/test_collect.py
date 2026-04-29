import cv2
import csv
import numpy as np
import mediapipe as mp
import time
import os
import sys
import pickle
import face_recognition
import tensorflow as tf
from datetime import datetime
from typing import List, Tuple, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

DAFTAR_SUBJEK = [
    "Vina",
    "Zafa",
    "Nopal",
    "Raka",
    "Orang_Lain",
]

DAFTAR_STATUS = ["Terdaftar", "Tidak Terdaftar"]
DAFTAR_CAHAYA = [
    ("Terang", "300-1000 lux"),
    ("Lampu",  "50-200 lux"),
    ("Gelap",  "0-20 lux"),
]
DAFTAR_RESOLUSI = ["720", "480"]
DAFTAR_JENIS_UJI = ["Face Recognition", "Gesture Recognition"]
DAFTAR_MODE = ["Idle", "Face Only", "Gesture Only", "Full System"]

CSV_FILENAME = "hasil_pengujian.csv"
CSV_HEADER = [
    "no", "nama", "status", "kondisi_cahaya", "lux", "resolusi",
    "jenis_uji", "mode_sistem", "prediksi", "hasil", "confidence",
    "fps", "latency_ms", "timestamp",
]

def load_known_faces() -> Tuple[List[np.ndarray], List[str]]:
    known_encodings, known_names = [], []
    print("[INFO] Memuat wajah terdaftar...")
    if not os.path.exists(config.KNOWN_FACES_DIR):
        os.makedirs(config.KNOWN_FACES_DIR)
        print(f"[WARN] Folder '{config.KNOWN_FACES_DIR}' kosong.")
        return [], []
    for fn in os.listdir(config.KNOWN_FACES_DIR):
        if fn.endswith(('.jpg', '.png', '.jpeg')):
            try:
                img = face_recognition.load_image_file(
                    os.path.join(config.KNOWN_FACES_DIR, fn))
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(os.path.splitext(fn)[0])
                    print(f"  [OK] {fn}")
            except Exception as e:
                print(f"  [ERR] {fn}: {e}")
    print(f"[INFO] Total wajah dimuat: {len(known_encodings)}")
    return known_encodings, known_names

def ekstrak_keypoints(results: Any) -> np.ndarray:
    rh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        rh = np.array([[r.x, r.y, r.z]
                        for r in results.left_hand_landmarks.landmark]).flatten()
    elif results.right_hand_landmarks:
        rh = np.array([[r.x, r.y, r.z]
                        for r in results.right_hand_landmarks.landmark]).flatten()
    return rh

def load_lstm_model():
    path = str(config.MODEL_LSTM_PATH)
    if not os.path.exists(path):
        print(f"[ERR] Model TFLite tidak ditemukan: {path}")
        return None, None, None, None, None
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print("[INFO] Model TFLite berhasil dimuat.")
    if not os.path.exists(config.LABEL_ENCODER_PATH):
        print("[ERR] Label encoder tidak ditemukan.")
        return None, None, None, None, None
    with open(config.LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    scaler = None
    if os.path.exists(config.SCALER_PATH):
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("[INFO] Scaler berhasil dimuat.")
    return interpreter, inp, out, le, scaler

def open_camera(resolusi: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if resolusi == "720":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)
    return cap

def draw_text(img, text, pos, scale=0.7, color=(0, 255, 0), thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def countdown_display(cap, seconds: int):
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        remaining = seconds - int(time.time() - start)
        if remaining <= 0:
            return frame
        h, w = frame.shape[:2]
        draw_text(frame, str(remaining), (w // 2 - 30, h // 2 + 20),
                  4, (0, 0, 255), 5)
        draw_text(frame, "Bersiap...", (w // 2 - 80, h // 2 + 60),
                  0.8, (0, 0, 255))
        cv2.imshow("Test Collection", frame)
        if cv2.waitKey(30) & 0xFF == 27:
            return None

def ensure_results_dir():
    os.makedirs(config.TEST_RESULTS_DIR, exist_ok=True)

def append_csv(filepath: str, row: list):
    write_header = not os.path.exists(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADER)
        w.writerow(row)

def pick_menu(title: str, options: list) -> int:
    """Tampilkan sub-menu dan kembalikan index pilihan (0-based)."""
    print(f"\n  {title}")
    for i, opt in enumerate(options, 1):
        if isinstance(opt, tuple):
            print(f"  {i}. {opt[0]}  ({opt[1]})")
        else:
            print(f"  {i}. {opt}")
    while True:
        try:
            c = int(input("  Masukkan pilihan: "))
            if 1 <= c <= len(options):
                return c - 1
        except ValueError:
            pass
        print("  [!] Pilihan tidak valid.")

# STATE: semua parameter pengujian
class TestConfig:
    def __init__(self):
        self.nama: str = DAFTAR_SUBJEK[0]
        self.status: str = DAFTAR_STATUS[0]
        self.cahaya: str = DAFTAR_CAHAYA[0][0]
        self.lux: int = 0
        self.resolusi: str = DAFTAR_RESOLUSI[0]
        self.jenis_uji: str = DAFTAR_JENIS_UJI[0]
        self.mode_sistem: str = DAFTAR_MODE[1]
        self.jumlah: int = config.TEST_DEFAULT_TRIALS

    def summary(self) -> str:
        lines = [
            f"  Nama Subjek   : {self.nama}",
            f"  Status        : {self.status}",
            f"  Cahaya        : {self.cahaya}",
            f"  Lux           : {self.lux}",
            f"  Resolusi      : {self.resolusi}p",
            f"  Jenis Uji     : {self.jenis_uji}",
            f"  Mode Sistem   : {self.mode_sistem}",
            f"  Jumlah        : {self.jumlah}x",
        ]
        return "\n".join(lines)

def run_test(tc: TestConfig):
    """Jalankan loop pengujian sesuai konfigurasi."""
    ensure_results_dir()
    csv_path = str(config.TEST_RESULTS_DIR / CSV_FILENAME)

    # Load resources sesuai kebutuhan
    known_enc, known_names = [], []
    need_face = tc.mode_sistem in ("Face Only", "Full System")
    need_gesture = tc.mode_sistem in ("Gesture Only", "Full System")
    is_idle = tc.mode_sistem == "Idle"

    if need_face:
        known_enc, known_names = load_known_faces()

    interpreter = inp_d = out_d = le = scaler = None
    if need_gesture:
        interpreter, inp_d, out_d, le, scaler = load_lstm_model()
        if interpreter is None:
            print("[ERR] Model LSTM gagal dimuat. Batal.")
            return

    cap = open_camera(tc.resolusi)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = None
    if need_gesture or is_idle:
        holistic = mp_holistic.Holistic(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    trial = 0
    print(f"\n[INFO] Tekan SPASI untuk mulai percobaan. ESC untuk berhenti.")
    print(f"[INFO] Target: {tc.jumlah} percobaan\n")

    while trial < tc.jumlah:
        # Live feed, tunggu SPASI
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # Tampilkan overlay tracking jika perlu
            if holistic is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res_mp = holistic.process(rgb)
                if res_mp.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, res_mp.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)
                if res_mp.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, res_mp.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)

            h, w = frame.shape[:2]
            draw_text(frame, f"{tc.nama} | {tc.cahaya} | {tc.resolusi}p",
                      (10, 30), 0.6, (255, 255, 0))
            draw_text(frame, f"Trial {trial+1}/{tc.jumlah} — Tekan SPASI",
                      (10, h - 20), 0.6, (255, 255, 255))
            cv2.imshow("Test Collection", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 32:   # SPASI
                break
            if key == 27:   # ESC
                _cleanup(cap, holistic)
                print(f"[INFO] Dihentikan. {trial} percobaan tercatat.")
                return

        # Countdown
        frame = countdown_display(cap, config.TEST_COUNTDOWN_SECONDS)
        if frame is None:
            _cleanup(cap, holistic)
            return

        # Variabel hasil
        prediksi = "-"
        hasil = "-"
        confidence = "-"
        latency = 0.0
        fps_val = 0.0

        # FACE RECOGNITION
        if need_face:
            t0 = time.time()
            small = cv2.resize(frame, (0, 0),
                               fx=config.FACE_REC_SCALE,
                               fy=config.FACE_REC_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb_small, model="hog")
            encs = face_recognition.face_encodings(rgb_small, locs)
            face_latency = (time.time() - t0) * 1000

            face_pred = "Unknown"
            face_dist = 1.0
            if len(known_enc) > 0 and len(encs) > 0:
                known_arr = np.array(known_enc)
                for enc in encs:
                    dists = np.linalg.norm(known_arr - enc, axis=1)
                    best = int(np.argmin(dists))
                    if dists[best] <= 0.6:
                        face_pred = str(known_names[best])
                        face_dist = float(dists[best])
                        break

            if tc.jenis_uji == "Face Recognition":
                prediksi = face_pred
                confidence = f"{1 - face_dist:.4f}" if face_pred != "Unknown" else "0"
                if tc.status == "Terdaftar":
                    hasil = "TP" if face_pred != "Unknown" else "FN"
                else:
                    hasil = "TN" if face_pred == "Unknown" else "FP"
                latency = face_latency
                fps_val = 1000.0 / face_latency if face_latency > 0 else 0

        # GESTURE RECOGNITION
        if need_gesture and tc.jenis_uji == "Gesture Recognition":
            sequence: List[np.ndarray] = []
            fps_list: List[float] = []
            prev_t = time.time()

            while len(sequence) < config.TIME_STEPS:
                ret, fr = cap.read()
                if not ret:
                    continue
                fr = cv2.flip(fr, 1)

                now = time.time()
                dt = now - prev_t
                fps_list.append(1.0 / dt if dt > 0 else 0)
                prev_t = now

                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res_mp = holistic.process(rgb)
                sequence.append(ekstrak_keypoints(res_mp))

                # Overlay
                if res_mp.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        fr, res_mp.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)
                if res_mp.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        fr, res_mp.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS)

                hh, ww = fr.shape[:2]
                cv2.circle(fr, (30, 30), 10, (0, 0, 255), -1)
                draw_text(fr, "REC", (50, 38), 0.6, (0, 255, 255))
                prog = len(sequence) / config.TIME_STEPS
                cv2.rectangle(fr, (0, hh - 10),
                              (int(prog * ww), hh), (0, 0, 255), -1)
                cv2.imshow("Test Collection", fr)
                cv2.waitKey(1)

            # Inference
            t0 = time.time()
            inp_data = np.array(sequence).astype(np.float32)
            if scaler is not None:
                inp_data = scaler.transform(inp_data)
            inp_data = np.expand_dims(inp_data, axis=0).astype(np.float32)
            interpreter.set_tensor(inp_d[0]['index'], inp_data)
            interpreter.invoke()
            res = interpreter.get_tensor(out_d[0]['index'])
            pred_idx = int(np.argmax(res[0]))
            conf = float(res[0][pred_idx])
            pred_label = le.inverse_transform([pred_idx])[0]
            gest_latency = (time.time() - t0) * 1000

            prediksi = pred_label
            confidence = f"{conf:.4f}"
            # Tentukan ground truth dari jenis gestur berdasarkan status
            # Untuk gesture test: "Terdaftar" berarti gestur seharusnya dikenali
            # Label gestur ditentukan dari nama (bisa juga ditambahkan menu terpisah)
            # Sederhananya: correct jika prediksi sesuai gestur yang dilakukan
            hasil = "correct" if pred_label == tc.nama else "wrong"
            latency = gest_latency
            fps_val = np.mean(fps_list) if fps_list else 0

        # IDLE MODE
        if is_idle:
            t0 = time.time()
            ret, fr = cap.read()
            if ret:
                fr = cv2.flip(fr, 1)
            idle_lat = (time.time() - t0) * 1000
            latency = idle_lat
            fps_val = 1000.0 / idle_lat if idle_lat > 0 else 0
            prediksi = "-"
            hasil = "-"

        # Log & Save
        trial += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            trial, tc.nama, tc.status, tc.cahaya, tc.lux,
            tc.resolusi, tc.jenis_uji, tc.mode_sistem,
            prediksi, hasil, confidence,
            f"{fps_val:.1f}", f"{latency:.1f}", ts,
        ]
        append_csv(csv_path, row)

        # Terminal log
        print(f"  [{trial}/{tc.jumlah}] {tc.nama} | {tc.cahaya} | "
              f"{tc.resolusi}p")
        print(f"  Prediction: {prediksi}")
        print(f"  Result    : {hasil}")
        print(f"  FPS       : {fps_val:.0f}")
        print(f"  Latency   : {latency:.1f}ms")
        print(f"  ---------------------------")

        # Tampilkan di layar
        color = (0, 255, 0) if hasil in ("TP", "TN", "correct", "-") \
                else (0, 0, 255)
        draw_text(frame, f"Hasil: {hasil} | {prediksi}",
                  (10, 30), 0.8, color)
        cv2.imshow("Test Collection", frame)
        cv2.waitKey(1000)

    _cleanup(cap, holistic)
    print(f"\n[DONE] {trial} percobaan selesai. CSV: {csv_path}")

def _cleanup(cap, holistic):
    if holistic is not None:
        holistic.close()
    cap.release()
    cv2.destroyAllWindows()

# MENU UTAMA
def main():
    tc = TestConfig()

    while True:
        print("\n" + "=" * 50)
        print("      PENGUJIAN SMART LOCK")
        print("=" * 50)
        print(tc.summary())
        print("-" * 50)
        print("  1. Pilih Nama Subjek")
        print("  2. Pilih Status Subjek")
        print("  3. Pilih Kondisi Pencahayaan")
        print("  4. Masukkan Nilai Lux")
        print("  5. Pilih Resolusi Kamera")
        print("  6. Pilih Jenis Pengujian")
        print("  7. Pilih Mode Sistem")
        print("  8. Jumlah Percobaan")
        print("  9. Mulai Pengujian")
        print("  0. Keluar")
        print("=" * 50)

        c = input("  Pilih menu: ").strip()

        if c == "1":
            idx = pick_menu("Pilih Nama Subjek:", DAFTAR_SUBJEK)
            tc.nama = DAFTAR_SUBJEK[idx]

        elif c == "2":
            idx = pick_menu("Status Subjek:", DAFTAR_STATUS)
            tc.status = DAFTAR_STATUS[idx]

        elif c == "3":
            idx = pick_menu("Kondisi Pencahayaan:", DAFTAR_CAHAYA)
            tc.cahaya = DAFTAR_CAHAYA[idx][0]

        elif c == "4":
            while True:
                try:
                    val = int(input("  Masukkan nilai lux (contoh: 350): "))
                    tc.lux = val
                    break
                except ValueError:
                    print("  [!] Masukkan angka.")

        elif c == "5":
            idx = pick_menu("Resolusi Kamera:", ["720p", "480p"])
            tc.resolusi = DAFTAR_RESOLUSI[idx]

        elif c == "6":
            idx = pick_menu("Jenis Pengujian:", DAFTAR_JENIS_UJI)
            tc.jenis_uji = DAFTAR_JENIS_UJI[idx]

        elif c == "7":
            idx = pick_menu("Mode Sistem:", DAFTAR_MODE)
            tc.mode_sistem = DAFTAR_MODE[idx]

        elif c == "8":
            while True:
                try:
                    val = int(input(f"  Jumlah percobaan [{tc.jumlah}]: ")
                              or str(tc.jumlah))
                    tc.jumlah = val
                    break
                except ValueError:
                    print("  [!] Masukkan angka.")

        elif c == "9":
            run_test(tc)

        elif c == "0":
            print("[INFO] Selesai.")
            break

        else:
            print("  [!] Pilihan tidak valid.")

if __name__ == "__main__":
    main()
