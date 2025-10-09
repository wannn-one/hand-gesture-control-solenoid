import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import lgpio # Import the new lgpio library
import time

# --- PENGATURAN LGPIO ---
SOLENOID_PIN = 17 # GPIO pin BCM 17
CHIP = 0 # Default GPIO chip for Raspberry Pi

# Open a handle to the GPIO chip
h = lgpio.gpiochip_open(CHIP)

# Claim the GPIO pin for output
lgpio.gpio_claim_output(h, SOLENOID_PIN)

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
    lgpio.gpio_write(h, SOLENOID_PIN, 1) # Kirim sinyal HIGH (1)
    time.sleep(3) # Biarkan pintu terbuka selama 3 detik
    lgpio.gpio_write(h, SOLENOID_PIN, 0) # Kirim sinyal LOW (0)
    print("Solenoid terkunci kembali.")

def kunci_pintu():
    print("GESTUR KUNCI TERDETEKSI -> Memastikan solenoid terkunci.")
    lgpio.gpio_write(h, SOLENOID_PIN, 0) # Kirim sinyal LOW (0)

# --- LOOP UTAMA ---
try:
    # Ensure solenoid is off at the start
    lgpio.gpio_write(h, SOLENOID_PIN, 0)
    
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
            prediction = model.predict(np.expand_dims(keypoints, axis=0), verbose=0) # verbose=0 to clean up output
            confidence = np.max(prediction)
            
            if confidence > 0.90:
                predicted_index = np.argmax(prediction)
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]

                cv2.putText(image, f'GESTUR: {predicted_label.upper()}', (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # --- LOGIKA KONTROL GPIO ---
                if predicted_label == 'buka_kunci':
                    buka_pintu()
                    time.sleep(5) # Add delay to prevent re-triggering
                elif predicted_label == 'kunci':
                    kunci_pintu()
                    time.sleep(2)

            cv2.imshow('Deteksi Gestur + Kontrol LGPIO', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
finally:
    # This block ensures GPIO resources are freed when the program exits
    print("Membersihkan dan menutup handle GPIO...")
    lgpio.gpio_write(h, SOLENOID_PIN, 0) # Ensure pin is off
    lgpio.gpiochip_close(h) # Close the handle
    cap.release()
    cv2.destroyAllWindows()