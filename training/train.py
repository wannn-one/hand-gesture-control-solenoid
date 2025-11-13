import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from config import config

DATASET_PATH = config.CSV_FILE_PATH
MODEL_PATH = config.MODEL_PATH
ENCODER_PATH = config.ENCODER_PATH


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {DATASET_PATH}. "
            "Pastikan sudah menjalankan perekaman dataset."
        )

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Memuat dataset dari '{DATASET_PATH}'...")
    df = pd.read_csv(DATASET_PATH)

    # Pisahkan antara fitur (koordinat) dan label (nama gestur)
    # Kita akan abaikan kolom 'sequence', 'frame_idx', dan 'image_path' untuk training
    X = df.drop(columns=['label', 'sequence', 'frame_idx', 'image_path']).values
    y = df['label'].values

    print(f"Jumlah total data: {len(df)}")
    print(f"Jumlah fitur (koordinat): {X.shape[1]}")
    print(f"Daftar gestur unik: {np.unique(y)}")

    # Encoding label
    # Mengubah label teks (misal: 'kunci') menjadi angka (misal: 1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Mengubah angka menjadi format "one-hot encoding" yang dibutuhkan model
    y_categorical = to_categorical(y_encoded)

    # Bagi data menjadi 80% untuk training dan 20% untuk testing/validasi
    stratify = y_encoded if config.TRAIN_STRATIFY else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_categorical,
        test_size=config.TRAIN_TEST_SPLIT,
        random_state=config.TRAIN_RANDOM_STATE,
        stratify=stratify,
    )

    print(f"Data training: {X_train.shape[0]} sampel")
    print(f"Data testing: {X_test.shape[0]} sampel")

    # Membangun arsitektur model Neural Network (MLP)
    model = Sequential([
        # Input layer: jumlah neuron harus sama dengan jumlah fitur (63)
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        # Hidden layer 1
        Dense(64, activation='relu'),
        # Hidden layer 2
        Dense(32, activation='relu'),
        # Output layer: jumlah neuron harus sama dengan jumlah gestur
        # Aktivasi 'softmax' untuk masalah klasifikasi multi-kelas
        Dense(y_categorical.shape[1], activation='softmax'),
    ])

    # Menampilkan ringkasan arsitektur model
    model.summary()

    # Kompilasi model
    # Mengatur optimizer, fungsi loss, dan metrik yang ingin ditampilkan
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Melatih model
    print("\n--- Memulai Proses Training ---")
    # 'epochs' adalah berapa kali model akan "melihat" keseluruhan data training
    # 'batch_size' adalah berapa banyak sampel data yang diproses dalam satu waktu
    model.fit(
        X_train,
        y_train,
        epochs=config.TRAIN_EPOCHS,
        batch_size=config.TRAIN_BATCH_SIZE,
        validation_data=(X_test, y_test),
    )
    print("--- Proses Training Selesai ---")

    # Evaluasi model
    print("\n--- Mengevaluasi Model ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Akurasi pada data testing: {accuracy * 100:.2f}%")
    print(f"Loss pada data testing: {loss:.4f}")

    # Menyimpan model yang sudah dilatih
    model.save(MODEL_PATH)
    print(f"\nModel berhasil disimpan di: {MODEL_PATH}")

    # Simpan juga label encoder untuk digunakan saat inferensi nanti
    with ENCODER_PATH.open('wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder berhasil disimpan di: {ENCODER_PATH}")


if __name__ == '__main__':
    main()