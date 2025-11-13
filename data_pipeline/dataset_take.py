import csv
import os

import cv2
import mediapipe as mp
import numpy as np

from config import config

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(config.ACTIONS)
jumlah_sequence = config.NUM_SEQUENCES
jumlah_frame = config.FRAMES_PER_SEQUENCE

def buat_folder():
    """Membuat struktur folder yang dibutuhkan."""
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for action in actions:
        (config.DATASET_DIR / action).mkdir(parents=True, exist_ok=True)

def ekstrak_keypoints(results):
    """Mengekstrak koordinat landmarks dari hasil deteksi MediaPipe."""
    # Jika tidak ada tangan kanan, hasilkan array nol
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3) # 21 landmarks, 3 koordinat (x,y,z)
    return rh

def siapkan_file_csv():
    """Membuat file CSV dan menulis header jika file belum ada."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = config.CSV_FILE_PATH.is_file()
    if not file_exists:
        # Buat header untuk CSV
        header = ['label', 'sequence', 'frame_idx', 'image_path']
        # Tambahkan nama kolom untuk setiap landmark
        for i in range(21):
            header += [f'rh_x_{i}', f'rh_y_{i}', f'rh_z_{i}']

        with config.CSV_FILE_PATH.open(mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def main():
    """Fungsi utama untuk menjalankan proses perekaman."""
    buat_folder()
    siapkan_file_csv()
    
    cap = cv2.VideoCapture(config.DATA_CAPTURE_CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    with mp_holistic.Holistic(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    ) as holistic:
        for action in actions:
            # Cari tahu sequence terakhir yang sudah direkam agar bisa melanjutkan
            try:
                # Dapatkan daftar folder sequence yang sudah ada untuk gestur ini
                dir_list = os.listdir(config.DATASET_DIR / action)
                # Ambil nomor sequence terakhir dan tambahkan 1
                start_sequence = max([int(d) for d in dir_list if d.isdigit()]) + 1
            except (ValueError, FileNotFoundError):
                start_sequence = 0 # Mulai dari 0 jika belum ada sama sekali

            print(f"Memulai dari sequence ke-{start_sequence} untuk gestur '{action}'")

            for sequence in range(start_sequence, start_sequence + jumlah_sequence):
                # Buat folder untuk sequence ini
                (config.DATASET_DIR / action / str(sequence)).mkdir(parents=True, exist_ok=True)

                # Tunggu pengguna siap
                while True:
                    ret, frame = cap.read()
                    cv2.putText(frame, 'SIAP? Tekan "S"', (180, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f'Merekam: {action.upper()} | Urutan ke: {sequence}', (15, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Perekam Dataset', frame)
                    if cv2.waitKey(10) & 0xFF == ord('s'):
                        break
                
                # Hitung mundur
                for t in range(config.RECORD_COUNTDOWN_SECONDS, 0, -1):
                    ret, frame = cap.read()
                    cv2.putText(frame, str(t), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.imshow('Perekam Dataset', frame)
                    cv2.waitKey(1000)

                # Loop perekaman frame
                for frame_num in range(jumlah_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Deteksi dengan MediaPipe
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Tampilkan status MEREKAM
                    cv2.putText(image, 'MEREKAM...', (15, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Gambar landmarks (opsional, untuk visualisasi)
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                    cv2.imshow('Perekam Dataset', image)

                    # --- SIMPAN DATA GAMBAR & KOORDINAT ---
                    # 1. Tentukan path untuk menyimpan gambar
                    image_path = config.DATASET_DIR / action / str(sequence) / f"{frame_num}.jpg"
                    # 2. Simpan gambar
                    cv2.imwrite(str(image_path), frame)
                    
                    # 3. Ekstrak keypoints
                    keypoints = ekstrak_keypoints(results)
                    
                    # 4. Buat satu baris data untuk CSV
                    csv_row = [action, sequence, frame_num, str(image_path)] + list(keypoints)
                    
                    # 5. Tulis baris tersebut ke file CSV
                    with config.CSV_FILE_PATH.open(mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(csv_row)

                    # Keluar jika menekan 'q'
                    if cv2.waitKey(50) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                print(f"Selesai merekam sequence ke-{sequence} untuk gestur '{action}'")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()