# 🚀 Automated Data Preprocessing - Telco Customer Churn

Repository ini merupakan bagian dari submission kelas **Membangun Sistem Machine Learning**. Fokus utama dari repository ini adalah melakukan otomatisasi tahap *data preprocessing* menggunakan **GitHub Actions**.

## 📌 Deskripsi Proyek
Dalam pipeline Machine Learning, *data preprocessing* adalah tahapan krusial. Proyek ini mendemonstrasikan bagaimana sebuah *script* Python dieksekusi secara otomatis setiap kali ada pembaruan (*push*) ke *branch* utama. 

Sistem akan membaca dataset mentah (`telco_raw`), memprosesnya (seperti memisahkan fitur dan target, menangani *missing values*, dan membagi data *train/test*), lalu secara otomatis menyimpan (*commit* & *push*) hasil pemrosesannya ke dalam folder `preprocessing/telco_preprocessing`.

## ⚙️ Teknologi yang Digunakan
* **Python 3.12**
* **Pandas & Scikit-Learn**
* **GitHub Actions** (CI Pipeline)

## 📂 Struktur Repository
* `telco_raw/`: Berisi dataset mentah awal.
* `preprocessing/`: Berisi *script* otomasi (`automate_Muhammad-Mishbahul-Muflihin.py`) dan folder `telco_preprocessing` yang menyimpan hasil data bersih (`X_train.csv`, `y_train.csv`, dll).
* `.github/workflows/`: Berisi file konfigurasi YAML untuk menjalankan GitHub Actions.
