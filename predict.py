import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle

# Muat model yang sudah dilatih
print("Memuat model...")
model = load_model('model_gestur.h5')

# Muat label encoder
print("Memuat label encoder...")
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Inisialisasi MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def ekstrak_keypoints(results):
    """Mengekstrak dan meratakan koordinat landmarks tangan kanan."""
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        # Jika tidak ada tangan kanan terdeteksi, kembalikan array nol
        rh = np.zeros(21 * 3)
    return rh

# Inisialisasi Video Capture
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Atur resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up model MediaPipe
with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame dari kamera.")
            break

        # Balik frame secara horizontal agar seperti cermin
        frame = cv2.flip(frame, 1)

        # Proses deteksi
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Gambar landmarks tangan
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- Logika Prediksi ---
        # 1. Ekstrak keypoints dari hasil deteksi
        keypoints = ekstrak_keypoints(results)
        
        # 2. Lakukan prediksi dengan model
        # Model Keras mengharapkan input dalam bentuk batch, jadi kita perlu mengubah
        # bentuk array dari (63,) menjadi (1, 63)
        prediction = model.predict(np.expand_dims(keypoints, axis=0))

        # 3. Dapatkan hasil prediksi
        confidence = np.max(prediction) # Ambil nilai probabilitas tertinggi
        predicted_index = np.argmax(prediction) # Ambil indeks dari probabilitas tertinggi
        
        # 4. Tampilkan hasil hanya jika confidence di atas ambang batas
        threshold = 0.80 # Anda bisa sesuaikan nilai threshold ini (misal: 0.7, 0.9)
        
        if confidence > threshold:
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]
            
            # Tampilkan label prediksi dan confidence di layar
            cv2.putText(image, f'GESTUR: {predicted_label.upper()}', (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'CONF: {int(confidence * 100)}%', (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Tampilkan frame ke jendela
        cv2.imshow('Deteksi Gestur Real-time', image)

        # Keluar dari loop jika menekan tombol 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Lepaskan semua resource
    cap.release()
    cv2.destroyAllWindows()