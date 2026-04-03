import cv2
import os
import time
import numpy as np
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- KONFIGURASI ---
DATA_PATH = os.path.join(os.getcwd(), 'Dataset_Gestur')
CSV_FILE_NAME = 'dataset_landmarks.csv'
actions = np.array(['buka_kunci', 'kunci'])

jumlah_sequence = 20
jumlah_frame = 40 

def buat_folder():
    for action in actions:
        try:
            os.makedirs(os.path.join(DATA_PATH, action))
        except FileExistsError:
            pass

def siapkan_file_csv():
    file_exists = os.path.isfile(CSV_FILE_NAME)
    if not file_exists:
        header = ['label', 'sequence', 'frame_idx', 'image_path']
        for i in range(21):
            header += [f'rh_x_{i}', f'rh_y_{i}', f'rh_z_{i}']
        with open(CSV_FILE_NAME, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

def get_right_hand_landmarks(results):
    """
    Mencari landmarks untuk TANGAN KANAN FISIK pengguna.
    """
    if not results.multi_hand_landmarks:
        return None
    
    # Loop semua tangan yang terdeteksi
    for idx, hand_handedness in enumerate(results.multi_handedness):
        label = hand_handedness.classification[0].label
        if label == 'Right':
            return results.multi_hand_landmarks[idx]
            
    return None

def main():
    buat_folder()
    siapkan_file_csv()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    # Setting resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Gunakan mp_hands.Hands seperti di mediapipe_test.py
    # Tingkatkan confidence agar lebih akurat
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2 # Deteksi 2 tangan agar tidak rebutan
    ) as hands:
        
        for action in actions:
            try:
                dir_list = os.listdir(os.path.join(DATA_PATH, action))
                start_sequence = max([int(d) for d in dir_list if d.isdigit()]) + 1
            except (ValueError, FileNotFoundError):
                start_sequence = 0 

            print(f"Memulai dari sequence ke-{start_sequence} untuk gestur '{action}'")

            for sequence in range(start_sequence, start_sequence + jumlah_sequence):
                try:
                    os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
                except FileExistsError:
                    pass

                # Tentukan instruksi posisi awal
                if action == 'buka_kunci':
                    instruksi_awal = "Posisi Awal: MENGEPAL (Fist)"
                    instruksi_aksi = "Lakukan: BUKA TANGAN"
                elif action == 'kunci':
                    instruksi_awal = "Posisi Awal: TERBUKA (Palm)"
                    instruksi_aksi = "Lakukan: KEPAL TANGAN"
                else:
                    instruksi_awal = "Siapkan Posisi Awal"
                    instruksi_aksi = "Lakukan Gestur"

                # --- FASE PERSIAPAN ---
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # 1. Flip Frame (Mirror)
                    frame = cv2.flip(frame, 1)
                    
                    # 2. Deteksi Tangan (untuk visualisasi saat persiapan)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # 3. Gambar Landmarks jika ada
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # 4. Teks UI
                    cv2.putText(image, 'Gunakan Tangan KANAN', (180, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                    
                    # Tampilkan Instruksi Khusus
                    cv2.putText(image, instruksi_awal, (100, 200),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'Tekan "S" jika SIAP', (180, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, f'Gestur: {action.upper()} | Seq: {sequence}', (15, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    cv2.imshow('Perekam Dataset', image)
                    if cv2.waitKey(10) & 0xFF == ord('s'):
                        break
                
                # --- FASE HITUNG MUNDUR ---
                for t in range(3, 0, -1):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    
                    # Visualisasi tetap jalan saat hitung mundur
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cv2.putText(image, str(t), (280, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(image, instruksi_aksi, (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.imshow('Perekam Dataset', image)
                    cv2.waitKey(1000)

                # --- FASE PEREKAMAN ---
                for frame_num in range(jumlah_frame):
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # 1. Flip
                    frame = cv2.flip(frame, 1)

                    # 2. Proses MediaPipe
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # 3. Gambar Landmarks (SEMUA TANGAN) ke 'image'
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 4. Teks Status
                    cv2.putText(image, 'MEREKAM...', (15, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, instruksi_aksi, (100, 400),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.imshow('Perekam Dataset', image)

                    # 5. Ambil Data Tangan Kanan
                    right_hand_landmarks = get_right_hand_landmarks(results)
                    
                    if right_hand_landmarks:
                        # Ekstrak koordinat
                        rh = np.array([[res.x, res.y, res.z] for res in right_hand_landmarks.landmark]).flatten()
                        if frame_num == 0: print("  > Tangan Kanan terdeteksi!")
                    else:
                        # Isi nol jika tidak terdeteksi
                        rh = np.zeros(21 * 3)
                        if frame_num == 0: 
                            if results.multi_handedness:
                                labels = [h.classification[0].label for h in results.multi_handedness]
                                print(f"  ! Tangan Kanan TIDAK terdeteksi. Terdeteksi: {labels}")
                            else:
                                print("  ! Tidak ada tangan terdeteksi.")

                    # 6. Simpan Gambar
                    image_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.jpg")
                    cv2.imwrite(image_path, image)
                    
                    # 7. Simpan CSV
                    csv_row = [action, sequence, frame_num, image_path] + list(rh)
                    with open(CSV_FILE_NAME, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(csv_row)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                print(f"Selesai sequence {sequence}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()