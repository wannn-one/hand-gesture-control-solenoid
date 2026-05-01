# Panduan Pengujian Sistem — `test_collect.py`

Script untuk **pengambilan data uji** Smart Lock System secara sistematis.
Merupakan replika pipeline `predict.py` yang disesuaikan untuk pencatatan hasil otomatis ke CSV.

---

## Cara Menjalankan

```bash
cd /path/to/hand-gesture-control-solenoid
python app/test_collect.py
```

---

## Menu Utama

```
══════════════════════════════════════════════════
   SISTEM PENGUJIAN SMART LOCK
══════════════════════════════════════════════════
  Nama Subjek   : Vina
  Status        : Terdaftar
  Cahaya        : Terang
  Lux           : 350
  Resolusi      : 720p
  Jenis Uji     : Face Recognition
  Mode Sistem   : Face Only
  Jumlah        : 30x
──────────────────────────────────────────────────
  1. Pilih Nama Subjek
  2. Pilih Status Subjek
  3. Pilih Kondisi Pencahayaan
  4. Masukkan Nilai Lux
  5. Pilih Resolusi Kamera
  6. Pilih Jenis Pengujian
  7. Pilih Mode Sistem
  8. Jumlah Percobaan
  9. Mulai Pengujian
  0. Keluar
══════════════════════════════════════════════════
```

Set parameter (menu 1–8), lalu tekan **9** untuk mulai.

---

## Menu Detail

### Menu 1 — Nama Subjek
```
  1. Vina
  2. Zafa
  3. Nopal
  4. Raka
  5. Orang_Lain (Unknown)
```

### Menu 2 — Status Subjek
```
  1. Terdaftar
  2. Tidak Terdaftar
```
> Ini adalah **ground truth** — jangan salah isi!

### Menu 3 — Kondisi Pencahayaan
```
  1. Terang  (300-1000 lux)
  2. Lampu   (50-200 lux)
  3. Gelap   (0-20 lux)
```

### Menu 4 — Nilai Lux
Input angka manual sesuai pengukuran lux meter.

### Menu 5 — Resolusi Kamera
```
  1. 720p
  2. 480p
```

### Menu 6 — Jenis Pengujian
```
  1. Face Recognition
  2. Gesture Recognition
```

### Menu 7 — Mode Sistem
```
  1. Idle (tanpa deteksi)
  2. Face Only
  3. Gesture Only
  4. Full System (Face + Gesture)
```

### Menu 8 — Jumlah Percobaan
Input angka (default: 30).

---

## Alur Pengujian (Setelah Tekan 9)

1. Live kamera ditampilkan di layar
2. Tekan **SPASI** untuk mulai percobaan
3. **Hitung mundur** (default 3 detik — bisa diubah di `config.py`)
4. Sistem otomatis berjalan sesuai mode:
   - **Face Only** → capture frame → face recognition → TP/TN/FP/FN
   - **Gesture Only** → record 40 frame → LSTM inference → correct/wrong
   - **Full System** → face rec + gesture rec
   - **Idle** → hanya ukur FPS
5. Hasil tampil di terminal & disimpan ke CSV
6. Tekan **SPASI** untuk lanjut, **ESC** untuk berhenti

### Terminal Log

```
  [1/50] Vina | Terang | 720p
  Prediction: Vina
  Result    : TP
  FPS       : 28
  Latency   : 142.3ms
  ---------------------------
```

---

## Output CSV

Semua hasil disimpan di: **`test_results/hasil_pengujian.csv`**

Data bersifat **auto-append** — aman untuk menjalankan berkali-kali.

### Kolom CSV

| Kolom | Contoh | Keterangan |
|-------|--------|------------|
| `no` | 1 | Nomor percobaan dalam sesi |
| `nama` | Vina | Nama subjek |
| `status` | Terdaftar | Ground truth |
| `kondisi_cahaya` | Terang | Label cahaya |
| `lux` | 350 | Nilai lux |
| `resolusi` | 720 | Resolusi kamera |
| `jenis_uji` | Face Recognition | Jenis pengujian |
| `mode_sistem` | Face Only | Mode pipeline |
| `prediksi` | Vina | Hasil deteksi sistem |
| `hasil` | TP | TP/TN/FP/FN atau correct/wrong |
| `confidence` | 0.8521 | Tingkat keyakinan |
| `fps` | 28.3 | Frame per second |
| `latency_ms` | 142.3 | Waktu proses (ms) |
| `timestamp` | 2026-04-28 20:01:01 | Waktu percobaan |

### Contoh CSV

```csv
no,nama,status,kondisi_cahaya,lux,resolusi,jenis_uji,mode_sistem,prediksi,hasil,confidence,fps,latency_ms,timestamp
1,Vina,Terdaftar,Terang,350,720,Face Recognition,Face Only,Vina,TP,0.8521,28.3,142.3,2026-04-28 20:01:01
2,Orang_Lain,Tidak Terdaftar,Gelap,15,480,Face Recognition,Face Only,Unknown,TN,0,24.1,155.7,2026-04-28 20:02:15
```

---

## Konfigurasi

Semua parameter default bisa diubah di [`config/config.py`](../config/config.py):

```python
# --- Pengujian / Data Collection ---
TEST_COUNTDOWN_SECONDS = 3      # Hitung mundur sebelum capture (detik)
TEST_DEFAULT_TRIALS = 30        # Jumlah default percobaan per sesi
TEST_FPS_DURATION = 60          # Durasi default uji FPS (detik)
TEST_RESULTS_DIR = ROOT_DIR / 'test_results'  # Folder output CSV
```

---

## Kontrol Keyboard

| Tombol | Fungsi |
|--------|--------|
| **SPASI** | Mulai percobaan / lanjut ke berikutnya |
| **ESC** | Berhenti & simpan data yang sudah tercatat |

---

## Catatan Penting

- **Nama subjek harus match** dengan file di `known_faces/` (untuk yang terdaftar)
- **Status subjek = ground truth** — pastikan benar sebelum mulai
- **Lux** diisi manual sesuai pembacaan lux meter
- Script **tidak mengaktifkan solenoid** — aman tanpa hardware aktuator
- Pipeline face recognition & gesture recognition **sama persis** dengan `predict.py`

---

## Apa Itu Ground Truth?

**Ground truth = kenyataan** tentang orang yang berdiri di depan kamera.
Ini diisi di **Menu 2 (Status Subjek)** dan menjadi acuan untuk menentukan TP/TN/FP/FN.

- Jika orang yang di depan kamera **wajahnya ada** di folder `known_faces/` → pilih **Terdaftar**
- Jika orang yang di depan kamera **wajahnya tidak ada** di folder `known_faces/` → pilih **Tidak Terdaftar**

> ⚠️ Jangan salah isi! Kalau ground truth salah, semua hasil TP/TN/FP/FN jadi tidak valid.

### Arti Hasil (TP / TN / FP / FN)

| Hasil | Status (Ground Truth) | Sistem Mengenali? | Artinya |
|-------|-----------------------|-------------------|---------|
| **TP** | Terdaftar | ✅ Ya | Sistem **benar** mengenali |
| **TN** | Tidak Terdaftar | ❌ Tidak (Unknown) | Sistem **benar** menolak |
| **FN** | Terdaftar | ❌ Tidak (Unknown) | Sistem **gagal** mengenali |
| **FP** | Tidak Terdaftar | ✅ Ya (salah orang) | Sistem **salah** mengenali |

---

## Panduan Langkah-demi-Langkah

### Skenario 1: Uji Orang Terdaftar (menghasilkan TP atau FN)

1. Pastikan `known_faces/` sudah ada foto `vina.jpg`
2. **Vina** berdiri di depan kamera
3. Setting menu:
   - Menu 1 → Vina
   - Menu 2 → **Terdaftar**
   - Menu 3 → (sesuai kondisi, misal Terang)
   - Menu 4 → (isi angka lux dari lux meter)
   - Menu 5 → 720
   - Menu 6 → Face Recognition
   - Menu 7 → Face Only
   - Menu 8 → 30
4. Tekan **9** (Mulai)
5. Tekan **SPASI** → countdown → capture → hasil muncul
6. Tekan **SPASI lagi** untuk percobaan ke-2, ke-3, ... sampai 30
7. Setelah 30x, sesi selesai otomatis

Jika sistem kenali Vina → **TP**. Jika gagal → **FN**.

### Skenario 2: Uji Orang Tidak Terdaftar (menghasilkan TN atau FP)

1. **Raka** (yang wajahnya TIDAK ada di `known_faces/`) berdiri di depan kamera
2. Setting menu:
   - Menu 1 → Raka
   - Menu 2 → **Tidak Terdaftar**
   - (sisanya sama)
3. Jalankan 30 percobaan

Jika sistem tolak (Unknown) → **TN**. Jika salah kenali → **FP**.

### Checklist Pengujian Lengkap (untuk Bab 4)

Ulangi skenario di atas untuk setiap kombinasi:

- **4 subjek**: Vina, Zafa, Nopal + 1 orang asing (Raka/Orang_Lain)
- **3 kondisi cahaya**: Terang, Lampu, Gelap
- **2 resolusi**: 720p, 480p
- **Masing-masing 3-5 percobaan** (minimal 30 data total per jenis uji)

Semua data otomatis masuk ke **1 file CSV** (`hasil_pengujian.csv`) karena auto-append.
