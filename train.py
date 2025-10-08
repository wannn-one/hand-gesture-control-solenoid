import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. Muat Dataset
print("Memuat dataset dari 'dataset_landmarks.csv'...")
df = pd.read_csv('dataset_landmarks.csv')

# 2. Persiapan Data
# Pisahkan antara fitur (koordinat) dan label (nama gestur)
# Kita akan abaikan kolom 'sequence', 'frame_idx', dan 'image_path' untuk training
X = df.drop(columns=['label', 'sequence', 'frame_idx', 'image_path']).values
y = df['label'].values

print(f"Jumlah total data: {len(df)}")
print(f"Jumlah fitur (koordinat): {X.shape[1]}")
print(f"Daftar gestur unik: {np.unique(y)}")

# 3. Encoding Label
# Mengubah label teks (misal: 'kunci') menjadi angka (misal: 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Mengubah angka menjadi format "one-hot encoding" yang dibutuhkan model
y_categorical = to_categorical(y_encoded)

# 4. Membagi Data
# Bagi data menjadi 80% untuk training dan 20% untuk testing/validasi
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

print(f"Data training: {X_train.shape[0]} sampel")
print(f"Data testing: {X_test.shape[0]} sampel")

# 5. Membangun Arsitektur Model Neural Network (MLP)
model = Sequential([
    # Input layer: jumlah neuron harus sama dengan jumlah fitur (63)
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    # Hidden layer 1
    Dense(64, activation='relu'),
    # Hidden layer 2
    Dense(32, activation='relu'),
    # Output layer: jumlah neuron harus sama dengan jumlah gestur
    # Aktivasi 'softmax' untuk masalah klasifikasi multi-kelas
    Dense(y_categorical.shape[1], activation='softmax')
])

# Menampilkan ringkasan arsitektur model
model.summary()

# 6. Kompilasi Model
# Mengatur optimizer, fungsi loss, dan metrik yang ingin ditampilkan
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Melatih Model
print("\n--- Memulai Proses Training ---")
# 'epochs' adalah berapa kali model akan "melihat" keseluruhan data training
# 'batch_size' adalah berapa banyak sampel data yang diproses dalam satu waktu
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
print("--- Proses Training Selesai ---")

# 8. Evaluasi Model
print("\n--- Mengevaluasi Model ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi pada data testing: {accuracy * 100:.2f}%")
print(f"Loss pada data testing: {loss:.4f}")

# 9. Menyimpan Model yang Sudah Dilatih
model_save_path = 'model_gestur.h5'
model.save(model_save_path)
print(f"\nModel berhasil disimpan di: {model_save_path}")

# Simpan juga label encoder untuk digunakan saat inferensi nanti
import pickle
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder berhasil disimpan di: label_encoder.pkl")