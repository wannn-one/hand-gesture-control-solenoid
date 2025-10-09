import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import RPi.GPIO as GPIO # Import library GPIO
import time

# --- PENGATURAN GPIO ---
GPIO.setmode(GPIO.BCM) # Gunakan penomoran pin BCM
GPIO.setwarnings(False)
SOLENOID_PIN = 17 # Ganti jika Anda menggunakan pin GPIO lain
GPIO.setup(SOLENOID_PIN, GPIO.OUT)
GPIO.output(SOLENOID_PIN, GPIO.LOW) # Pastikan solenoid mati di awal

# --- MUAT MODEL & ENCODER ---
print("Memuat model...")
model = load_model('model_gestur.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# --- INISIALISASI MEDIAPIPE & KAMERA ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

def ekstrak_keypoints(results):
    # (Fungsi ini sama seperti sebelumnya, tidak perlu diubah)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)
    return rh

# --- FUNGSI UNTUK MENGONTROL SOLENOID ---
def buka_pintu():
    print("GESTUR BUKA KUNCI TERDETEKSI -> Membuka solenoid...")
    GPIO.output(SOLENOID_PIN, GPIO.HIGH) # Kirim sinyal HIGH untuk menyalakan relay
    time.sleep(3) # Biarkan pintu terbuka selama 3 detik
    GPIO.output(SOLENOID_PIN, GPIO.LOW) # Matikan relay
    print("Solenoid terkunci kembali.")

def kunci_pintu():
    print("GESTUR KUNCI TERDETEKSI -> Memastikan solenoid terkunci.")
    GPIO.output(SOLENOID_PIN, GPIO.LOW) # Kirim sinyal LOW untuk mematikan relay

# --- LOOP UTAMA ---
try:
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = ekstrak_keypoints(results)
            prediction = model.predict(np.expand_dims(keypoints, axis=0))
            confidence = np.max(prediction)
            
            if confidence > 0.8: # Naikkan threshold agar lebih yakin
                predicted_index = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]

                # Tampilkan label di layar
                cv2.putText(image, f'GESTUR: {predicted_label.upper()}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # --- LOGIKA KONTROL GPIO ---
                if predicted_label == 'buka_kunci':
                    buka_pintu()
                    # Tambahkan jeda agar tidak terus menerus membuka
                    time.sleep(5) 
                elif predicted_label == 'kunci':
                    kunci_pintu()
                    time.sleep(2)

            cv2.imshow('Deteksi Gestur + Kontrol GPIO', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
finally:
    # Pastikan pin GPIO dibersihkan saat program berhenti
    print("Membersihkan pin GPIO...")
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()