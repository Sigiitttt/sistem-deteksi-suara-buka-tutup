# Sistem Deteksi Suara "Buka" dan "Tutup"

Proyek ini merupakan sistem machine learning untuk mendeteksi perintah suara sederhana: **"buka"** dan **"tutup"**, yang hanya dikenali dari **suara pemilik dataset** (speaker-dependent). Sistem dibuat menggunakan Python dan Streamlit sebagai antarmuka web.

## 🎯 Tujuan Proyek

* Mengembangkan sistem klasifikasi suara yang mampu mengenali dua kata kunci: *buka* dan *tutup*.
* Mengimplementasikan ekstraksi fitur audio berbasis **spectral**, **temporal**, dan **statistik**.
* Melatih model machine learning dan menyimpannya dalam file `.pkl` untuk digunakan pada aplikasi.
* Membangun aplikasi deteksi suara berbasis web yang memungkinkan pengguna untuk **mengunggah file audio** atau **merekam suara secara langsung**.
* Membuat sistem yang hanya mengenali suara pemilik dataset sehingga jika orang lain berbicara, hasilnya tidak akurat atau tidak dikenali.

## 📁 Struktur Dataset

Dataset terdiri dari dua kategori utama:

* **buka**  → berisi seluruh rekaman suara kata "buka"
* **tutup** → berisi seluruh rekaman suara kata "tutup"

Dataset dibagi menjadi:

* **train** → untuk melatih model
* **test** → untuk evaluasi model

Format file menggunakan `.wav`.

## 🔍 Ekstraksi Fitur

Proses ekstraksi fitur dilakukan menggunakan tiga jenis karakteristik suara:

### 1. **Fitur Spectral**

* MFCC
* Mel Spectrogram
* Spectral Centroid
* Spectral Bandwidth
* Spectral Contrast

### 2. **Fitur Temporal**

* Zero Crossing Rate (ZCR)
* Energy

### 3. **Fitur Statistik**

* Mean
* Standard Deviation
* Skewness

Semua fitur kemudian digabungkan menjadi vektor fitur final untuk melatih model.

## 🤖 Model Machine Learning

Model machine learning dilatih menggunakan algoritma **Random Forest Classifier**. Model tersebut kemudian disimpan sebagai:

```
voice_model.pkl
```

Model ini digunakan oleh aplikasi web untuk melakukan prediksi pada audio baru.

## 🌐 Aplikasi Web

Aplikasi dibangun menggunakan **Streamlit**, dengan dua fitur utama:

### 1. **Upload File Audio (.wav)**

Pengguna dapat mengunggah file suara untuk dianalisis oleh model.

### 2. **Rekam Suara Langsung (jika browser mendukung)**

Menggunakan komponen `st_audiorec`, pengguna dapat merekam suara langsung dari browser.
Jika perangkat tidak mendukung fitur ini, pengguna akan diarahkan untuk menggunakan metode upload.

Setelah suara diproses, aplikasi akan menampilkan hasil prediksi: **"buka"** atau **"tutup"**.

## ⚠️ Catatan

* Sistem hanya dioptimalkan untuk suara pemilik dataset, sehingga performa tidak dijamin pada speaker lain.
* Beberapa perangkat mobile mungkin tidak mendukung fitur rekaman browser.
* Aplikasi bergantung pada koneksi HTTPS agar akses mikrofon dapat berjalan.

## 📦 File Penting dalam Proyek

* `app.py` → aplikasi Streamlit
* `extract_features.py` → fungsi ekstraksi fitur audio
* `voice_model.pkl` → model machine learning yang sudah dilatih
* `requirements.txt` → daftar library yang digunakan

## ✨ Hasil Akhir

Proyek ini menghasilkan sistem deteksi suara sederhana yang mampu mengklasifikasikan perintah **"buka"** dan **"tutup"** menggunakan model machine learning serta menyediakan antarmuka web yang interaktif untuk pengujian secara real-time.
<img width="1917" height="977" alt="image" src="https://github.com/user-attachments/assets/d976b631-a7c9-4d05-9266-17d88cfdf399" />
