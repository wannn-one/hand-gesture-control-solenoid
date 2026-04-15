"""
Script untuk menyimpan StandardScaler yang sama persis dengan yang digunakan saat training.
Jalankan script ini setelah training selesai, atau kapan saja selama dataset_landmarks.csv tersedia.

Usage: python training/save_scaler.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Tambahkan root directory ke path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

def main():
    # Step 1: Baca dataset
    data_path = config.CSV_FILE_PATH
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset tidak ditemukan di {data_path}")
        return

    print(f"Membaca dataset dari {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset: {len(df)} baris, {len(df.columns)} kolom")

    # Step 2: Buat sequences (sama persis dengan notebook)
    df['global_seq'] = df['label'] + '_' + df['sequence'].astype(str)
    grouped = df.groupby('global_seq')

    sequences = []
    labels = []

    for name, group in grouped:
        if len(group) == config.TIME_STEPS:  # 40 frames per sequence
            sequences.append(group.iloc[:, 4:-1].values)  # Skip sequence ids, paths, global_seq
            labels.append(group['label'].iloc[0])

    X = np.array(sequences)
    y = np.array(labels)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Step 3: Encode labels & split (sama persis dengan notebook)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training set: {X_train.shape}")

    # Step 4: Fit StandardScaler pada data training
    scaler = StandardScaler()
    X_train_res = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_res)
    print(f"Scaler fit selesai. Mean shape: {scaler.mean_.shape}, Scale shape: {scaler.scale_.shape}")

    # Step 5: Simpan scaler
    scaler_path = config.SCALER_PATH
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler berhasil disimpan ke: {scaler_path}")

    # Verifikasi
    with open(scaler_path, 'rb') as f:
        loaded_scaler = pickle.load(f)
    print(f"Verifikasi: mean match = {np.allclose(scaler.mean_, loaded_scaler.mean_)}")
    print("DONE!")


if __name__ == "__main__":
    main()
