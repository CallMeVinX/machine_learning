# Proyek Machine Learning: Prediksi Persetujuan Pinjaman (Loan Approval)
### 🏷️ Project ID: `A10 - Persetujuan Pinjaman`
**Dokumentasi Teknis & Panduan Repositori (UTS Milestone 1)**

| Informasi Proyek | Keterangan |
| :--- | :--- |
| **Project ID** | **A10 - Persetujuan Pinjaman** |
| **Mata Kuliah** | Machine Learning (Semester Ganjil TA 2026/2027) |
| **Judul Proyek** | Prediksi Persetujuan Pinjaman Berbasis Classical Machine Learning dan Rekayasa Rasio Risiko Finansial |
| **Notebook Eksperimen** | [`machinelearning.ipynb`](machinelearning.ipynb) |
| **Dataset Sumber** | [`Loan Dataset.csv`](Loan%20Dataset.csv) ([Kaggle Source Link](https://www.kaggle.com/datasets/arbaaztamboli/loan-approval-dataset)) |
| **Repositori GitHub** | [https://github.com/CallMeVinX/machine_learning](https://github.com/CallMeVinX/machine_learning) |

#### 👥 Anggota Kelompok:
1. **Alvin Dinata** — NIM: `231111420`
2. **James** — NIM: `231111512`
3. **Vera Nataria** — NIM: `231110030`

---

## 📌 Deskripsi Masalah & Problem Formulation

### 1. Konteks Bisnis & Urgensi Masalah
Penyaluran fasilitas pinjaman (*kredit*) merupakan pilar bisnis utama sekaligus sumber risiko finansial terbesar bagi institusi perbankan. Evaluasi kelayakan kredit secara manual terhadap puluhan ribu berkas pengajuan berpotensi mengalami bias subjektivitas, waktu pemrosesan yang lambat, serta kegagalan mendeteksi hubungan non-linear antar-atribut finansial pemohon. Dalam konteks keputusan persetujuan kredit, kesalahan prediksi memiliki konsekuensi bisnis yang sangat asimetris:
* **Risiko Kredit Macet (False Positive / Type I Error):** Meloloskan calon debitur yang sebenarnya berisiko gagal bayar (aktual kelas 0 / Ditolak, namun diprediksi kelas 1 / Disetujui). Kesalahan ini berakibat fatal karena menyebabkan kerugian modal langsung berupa kredit bermasalah (*Non-Performing Loan* / NPL).
* **Kehilangan Peluang Pendapatan (False Negative / Type II Error):** Menolak calon debitur yang sebenarnya memiliki kapasitas finansial layak (aktual kelas 1 / Disetujui, namun diprediksi kelas 0 / Ditolak). Kesalahan ini menyebabkan hilangnya potensi pendapatan bunga (*opportunity loss*) dan menurunkan kepuasan nasabah.

### 2. Tujuan Proyek
Membangun sistem pendukung keputusan otomatis (*credit underwriting decision support system*) berbasis **Classical Machine Learning** untuk memprediksi status persetujuan pinjaman secara objektif, berkeadilan (*fairness*), transparan, dan terbebas dari kebocoran data (*data leakage*).

### 3. Pendekatan Machine Learning
* **Paradigma:** Supervised Learning (Pembelajaran Terawasi).
* **Tugas Prediktif:** Binary Classification (Klasifikasi Biner) dengan variabel target diskrit `Loan_Approval_Status`:
  * `1` = Disetujui (*Approved*) — 33.366 observasi (64,17%)
  * `0` = Ditolak (*Rejected*) — 18.634 observasi (35,83%)
* **Rasionalisasi Pendekatan Klasik:** Algoritma berbasis pohon dan regresi terstandarisasi menawarkan efisiensi komputasi tinggi pada data tabular heterogen, resisten terhadap multikolinearitas (pada pohon), serta memiliki tingkat transparansi (*white-box*) yang dapat dipertanggungjawabkan kepada regulator perbankan.

---

## 🛠️ Petunjuk Penggunaan Repository (How to Run & Reproduce)

Seluruh eksperimen telah dirancang secara modular dan dapat direproduksi secara mandiri (*end-to-end reproducible*).

> [!IMPORTANT]
> **Rekomendasi Lingkungan Eksekusi (IDE Lokal vs Cloud):**  
> **Sangat direkomendasikan untuk menjalankan proyek ini menggunakan IDE lokal seperti Visual Studio Code (VS Code)** atau JupyterLab lokal.  
> **Mengapa tidak disarankan di Google Colab?** Lingkungan cloud virtual machine seperti Google Colab berpotensi menghasilkan variasi numerik minor pada desimal metrik (akibat perbedaan arsitektur thread/multiprocessing library `loky`, optimasi aljabar linear BLAS/LAPACK pada kernel Linux cloud, serta perbedaan default integer bitwidth antara runtime OS). Menjalankan secara lokal di VS Code menjamin seluruh tabel metrik, koefisien korelasi, dan visualisasi menghasilkan angka yang **konsisten** dengan dokumentasi laporan ini.

### 1. Struktur File & Artefak Repositori
```text
├── Loan Dataset.csv         # Dataset historis profil kredit pemohon (52.000 baris, 27 kolom)
├── machinelearning.ipynb    # Jupyter Notebook eksperimen lengkap (EDA, Preprocessing, Modeling, CV, Evaluasi)
└── README.md                # Dokumentasi komprehensif proyek, panduan eksekusi, & hasil eksperimen
```

### 2. Prasyarat Lingkungan (*Environment Prerequisites*)
* Python versi 3.10 atau 3.11 (direkomendasikan).
* IDE: **Visual Studio Code (VS Code)** dengan ekstensi resmi **Python** dan **Jupyter** terpasang.
* Instalasi pustaka pendukung via terminal:
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn ipykernel
  ```

### 3. Panduan Menjalankan Eksperimen di VS Code (*Step-by-Step*)
1. **Kloning Repositori:**
   Buka terminal lokal Anda dan unduh repositori:
   ```bash
   git clone https://github.com/CallMeVinX/machine_learning.git
   cd machine_learning
   ```
2. **Buka Proyek di VS Code:**
   Buka folder proyek secara langsung melalui terminal:
   ```bash
   code .
   ```
   *Atau melalui menu GUI VS Code: `File` -> `Open Folder...` -> pilih folder `machine_learning`.*
3. **Buka File Notebook Eksperimen:**
   Pada panel *Explorer* di sisi kiri VS Code, klik dan buka file **[`machinelearning.ipynb`](machinelearning.ipynb)**.
4. **Pilih Python Kernel:**
   Di pojok kanan atas editor notebook VS Code, klik **Select Kernel** -> pilih lingkungan Python lokal Anda (misalnya `Python 3.10` atau `Python 3.11`).
5. **Eksekusi Eksperimen Secara Sekuensial:**
   Klik tombol **"Run All"** pada bilah atas editor notebook (atau gunakan menu `Clear All Outputs` lalu `Run All`).
   * Seluruh alur eksperimen dari Bagian 1 (Setup & EDA), Bagian 4 (Pipeline Preprocessing), Bagian 5 (Feature Engineering), Bagian 8 (Stratified 5-Fold CV), hingga Bagian 9–11 (Error Analysis & Model Card) akan dieksekusi secara otomatis dari atas ke bawah menghasilkan output yang konsisten.

---

## Daftar Isi Dokumentasi
- [Identitas Proyek & Anggota Kelompok](#-project-id-a10---persetujuan-pinjaman)
- [Deskripsi Masalah & Problem Formulation](#-deskripsi-masalah--problem-formulation)
- [Petunjuk Penggunaan Repository](#️-petunjuk-penggunaan-repository-how-to-run--reproduce)
1. [Karakteristik & Ringkasan Dataset (Data Understanding & Setup)](#1-karakteristik--ringkasan-dataset-data-understanding--setup)
2. [Temuan Eksplorasi Data & Driver Utama Keputusan Kredit (EDA)](#2-temuan-eksplorasi-data--driver-utama-keputusan-kredit-eda)
3. [Temuan Kualitas Data & Anomali Bisnis (Bukti Data Sintetis)](#3-temuan-kualitas-data--anomali-bisnis-bukti-data-sintetis)
4. [Desain Preprocessing & Proteksi Kebocoran Data (Data Leakage)](#4-desain-preprocessing--proteksi-kebocoran-data-data-leakage)
5. [Rekayasa Fitur Rasio Finansial (Feature Engineering)](#5-rekayasa-fitur-rasio-finansial-feature-engineering)
6. [Baseline Modeling (Logistic Regression & Random Forest)](#6-baseline-modeling-logistic-regression--random-forest)
7. [Eksplorasi Kandidat Model Tambahan (Decision Tree)](#7-eksplorasi-kandidat-model-tambahan-decision-tree)
8. [Audit Strategi Validasi & Uji Kestabilan Model (Milestone Pertemuan 5)](#8-audit-strategi-validasi--uji-kestabilan-model-milestone-pertemuan-5)
9. [Diagnosis Model, Seleksi Fitur, & Analisis Galat](#9-diagnosis-model-seleksi-fitur--analisis-galat)
10. [Evaluasi Komprehensif: Benchmark Train vs Test Seluruh Model](#10-evaluasi-komprehensif-benchmark-train-vs-test-seluruh-model)
11. [Tata Kelola Pemodelan (Model Governance), Kesimpulan Akhir, & Rencana Lanjutan](#11-tata-kelola-pemodelan-model-governance-kesimpulan-akhir--rencana-lanjutan)

---

## 1. Karakteristik & Ringkasan Dataset (Data Understanding & Setup)

* **Sumber Dataset Publik:** Kaggle — [Loan Approval Dataset by Arbaaz Tamboli](https://www.kaggle.com/datasets/arbaaztamboli/loan-approval-dataset)
* **File Dataset:** `Loan Dataset.csv`
* **Dimensi Awal:** 52.000 baris observasi pemohon dan 27 kolom atribut (15 kolom int64, 2 kolom float64, dan 10 kolom kategorikal teks). Pasca-rekayasa fitur dan encoding bertransformasi menjadi 52 dimensi prediktor (23 numerik + 29 kategori one-hot).
* **Unit Observasi:** Satu baris merepresentasikan satu pengajuan aplikasi pinjaman oleh seorang pemohon individu (`Applicant_ID`).
* **Target Biner (Loan_Approval_Status):**
  * **Kelas 1 (Disetujui / Approved):** 33.366 sampel (**64.17%**)
  * **Kelas 0 (Ditolak / Rejected):** 18.634 sampel (**35.83%**)
  * **Karakteristik Ketimpangan:** Moderat (*Mild Imbalance*, rasio ~1.79 : 1).
  * **Bahaya Metrik Tunggal:** Model naif mayoritas yang menebak 'Disetujui' pada seluruh data akan menghasilkan akurasi semu 64.17%, namun mengalami kegagalan fatal (*zero recall*) dalam mendeteksi nasabah berisiko kredit macet.

---

## 2. Temuan Eksplorasi Data & Driver Utama Keputusan Kredit (EDA)

1. **Prediktor Finansial Positif Dominan:**
   * Credit_Score ($r = 0.617$): Sinyal terkuat penentu kelayakan pinjaman.
   * Loan_Amount_Requested ($r = 0.508$): Pemohon dengan plafon pinjaman besar cenderung disetujui (berkorelasi kuat dengan pendapatan tinggi).
   * Annual_Income ($r = 0.472$): Kapasitas pendapatan merupakan penopang utama persetujuan kredit.
2. **Multikolinearitas Tinggi Antarfitur Finansial:**
   * Korelasi linear sangat kuat antara Loan_Amount_Requested dan Annual_Income ($r = 0.849$), serta Credit_Score dengan Loan_Amount_Requested ($r = 0.703$) dan Annual_Income ($r = 0.651$).
   * Mengindikasikan bahwa pemohon dengan pendapatan besar mengajukan nominal pinjaman yang jauh lebih tinggi.
3. **Prediktor Negatif Nyata:**
   * Age ($r = -0.184$): Pemohon berusia lebih tua memiliki tingkat penolakan yang lebih tinggi pada dataset ini.
4. **Variabel Non-Prediktif (Noise):**
   * Fitur mentah Interest_Rate ($r = 0.009$) dan Default_Risk ($r = -0.005$) memiliki korelasi linear mendekati nol terhadap target persetujuan pinjaman.
   * Demikian pula variabel demografi seperti jumlah tanggungan (Dependents, $r = -0.003$) dan frekuensi transaksi tahunan ($r = -0.003$).

---

## 3. Temuan Kualitas Data & Anomali Bisnis (Bukti Data Sintetis)

1. **Kelengkapan Teknis:**
   * 0 *missing value* dan 0 baris duplikat penuh pada seluruh 52.000 baris data.
   * Applicant_ID bernilai 100% unik ($r = -0.0019$) dan wajib dieliminasi dari matriks pemodelan.
2. **Disparitas Skala Ekstrem:**
   * Nilai fitur bervariasi dari ratusan ribu (Annual_Income hingga 149.998) hingga satuan/desimal (Interest_Rate dan Default_Risk), membuktikan pentingnya standardisasi skala (StandardScaler).
3. **Outlier Metode IQR (1.5x):**
   * Ditemukan outlier pada Age sebanyak 862 data (1.66%, usia > 66 tahun) dan Loan_Amount_Requested sebanyak 896 data (1.72%, plafon > 39.110).
4. **Anomali Logika Bisnis (Bukti Kuat Data Sintetis / Synthetic Artefacts):**
   * 17.350 dari 17.350 pemohon (100%) dengan Existing_Loans == 0 tercatat memiliki saldo utang atau plafon pinjaman aktif.
   * 15.618 baris (30.03%) memiliki sisa utang melebihi plafon pinjaman awal (Outstanding_Debt > Total_Existing_Loan_Amount).
   * 4.836 baris (9.30%) mengalami defisit arus kas tahunan (Monthly_Expenses * 12 > Annual_Income).
   * 10.376 dari 10.376 pemohon (100%) berstatus Unemployed tercatat memiliki Occupation_Type formal (*Salaried*, *Professional*, *Business*, *Freelancer*).
   * Anomali Demografi: 216 pemohon usia $\le 21$ tahun bergelar S2 (Postgraduate) dan 307 pemohon usia $< 20$ tahun memiliki tanggungan $\ge 2$.

---

## 4. Desain Preprocessing & Proteksi Kebocoran Data (Data Leakage)

Pencegahan kebocoran data (*data leakage*) dijamin secara ketat dengan memisahkan tahap fitting transformasi murni hanya pada data latih:

```
                      DATASET UTUH (52.000 Sampel)
                                   │
               ┌───────────────────┴───────────────────┐
               ▼ (Stratified Split 80%)                 ▼ (Stratified Hold-out 20%)
       TRAINING SET (41.600)                    TEST SET (10.400)
               │                                [Disimpan Murni untuk Evaluasi Akhir]
   ┌───────────┴───────────┐
   ▼ Stratified 5-Fold CV  ▼
┌───────┬───────┬───────┬───────┬───────┐
│Fold 1 │Fold 2 │Fold 3 │Fold 4 │Fold 5 │
└───────┴───────┴───────┴───────┴───────┘
   │
   ├─ 4 Folds: Preprocessor FIT + Classifier FIT
   └─ 1 Fold : Preprocessor TRANSFORM + VALIDATION
```

* **Skema Split:** 80% Data Latih (41.600 data) dan 20% Data Uji (10.400 data) dengan `stratify=y` dan `random_state=42`.
* **ColumnTransformer Architecture:**
  * **Fitur Numerik:** SimpleImputer(strategy='median') + StandardScaler().
  * **Fitur Kategorikal:** SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore', sparse_output=False).
* **Jaminan Bebas Bocor:** Seluruh estimator hanya dipelajari (*fit*) dari X_train. Data uji (X_test) hanya di-transformasi murni.

---

## 5. Rekayasa Fitur Rasio Finansial (Feature Engineering)

Diciptakan 9 rasio keuangan deterministik per baris (*row-wise deterministic operations*) yang mencerminkan likuiditas dan kapasitas beban utang riil pemohon:

1. **Debt_to_Income_Ratio** = `Outstanding_Debt / (Annual_Income + 1)` — ($r = -0.319$)
2. **Loan_to_Income_Ratio** = `Loan_Amount_Requested / (Annual_Income + 1)` — ($r = 0.000$)
3. **Expense_to_Income_Ratio** = `(Monthly_Expenses * 12) / (Annual_Income + 1)` — ($r = -0.357$)
4. **Monthly_Disposable_Income** = `(Annual_Income / 12) - Monthly_Expenses` — ($r = 0.431$)
5. **Estimated_Monthly_Installment** = `Loan_Amount_Requested * (1 + (Interest_Rate / 100) * (Loan_Term / 12)) / Loan_Term` — ($r = 0.247$)
6. **Installment_to_Income_Ratio** = `(Estimated_Monthly_Installment * 12) / (Annual_Income + 1)` — ($r = -0.001$)
7. **Is_Financial_Deficit** = `((Monthly_Expenses * 12) > Annual_Income).astype(int)` — ($r = -0.297$)
8. **Debt_Utilization_Ratio** = `Outstanding_Debt / (Total_Existing_Loan_Amount + 1)` — ($r = -0.004$)
9. **Credit_Risk_Factor** = `Default_Risk * (1000 - Credit_Score)` — ($r = -0.379$)

---

## 6. Baseline Modeling (Logistic Regression & Random Forest)

### Evaluasi Baseline pada Data Uji (10.400 Sampel)
* **Logistic Regression (Linear Baseline):**
  * Akurasi: **85.12%** | F1-Score (Weighted): **84.77%** | ROC-AUC: **0.8158**
  * Confusion Matrix: TN = 2.640, FP = 1.087, FN = 461, TP = 6.212
  * Karakteristik: Model linear stabil, transparan, namun terbatas dalam menangkap interaksi batas keputusan non-linear tanpa rekayasa fitur.
* **Random Forest (Ensemble Bagging Baseline):**
  * Akurasi: **85.11%** | F1-Score (Weighted): **84.76%** | ROC-AUC: **0.8200**
  * Confusion Matrix: TN = 2.639, FP = 1.088, FN = 461, TP = 6.212
  * Karakteristik: Menggabungkan 100 pohon keputusan acak, mereduksi variansi secara efektif.

---

## 7. Eksplorasi Kandidat Model Tambahan (Decision Tree)

* **Decision Tree Classifier (max_depth=8, class_weight='balanced'):**
  * Akurasi: **84.75%** | F1-Score (Weighted): **84.42%** | ROC-AUC: **0.8202**
  * Confusion Matrix: TN = 2.642, FP = 1.085, FN = 501, TP = 6.172
* **Komparasi Bisnis:** Decision Tree meningkatkan False Negative (FN) sebesar **+40 nasabah layak yang keliru ditolak** (FN = 501 vs 461 pada LR/RF). Ini menyebabkan hilangnya potensi pendapatan bunga (*opportunity loss*) akibat batas pemisah orthogonal pohon tunggal yang kaku.

---

## 8. Audit Strategi Validasi & Uji Kestabilan Model (Milestone Pertemuan 5)

Eksperimen validasi silang Stratified 5-Fold Cross-Validation dijalankan pada data latih (41.600 sampel) untuk membuktikan kestabilan model lintas lipatan (*fold*):

### Tabel Kestabilan Lintas 5 Lipatan:
| Metrik Validasi | Logistic Regression | Decision Tree (depth=8) | Random Forest (100 pohon) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC (Fold 1 - 5)** | 0.8103 s.d. 0.8214 | 0.8066 s.d. 0.8213 | 0.8137 s.d. 0.8222 |
| **Rata-rata ROC-AUC** | **0.8164** | **0.8124** | **0.8173** |
| **Standar Deviasi ($\sigma$)** | **0.0043** | **0.0050** | **0.0032** |
| **Rata-rata Akurasi** | **0.8497** | **0.8434** | **0.8497** |
| **Standar Deviasi Akurasi** | **0.0024** | **0.0037** | **0.0024** |
| **Rata-rata F1-Weighted** | **0.8461** | **0.8401** | **0.8461** |
| **Standar Deviasi F1** | **0.0027** | **0.0037** | **0.0027** |

### Temuan Kestabilan & Ensemble:
1. **Stabilitas Sangat Tinggi:** Seluruh model memiliki fluktuasi skor yang sangat kecil ($\sigma \le 0.0050$ atau $\le 0.50\%$), mengonfirmasi bahwa performa model konsisten dan tidak bergantung pada partisi data tertentu.
2. **Keunggulan Ensemble:** Random Forest menunjukkan variansi terendah pada ROC-AUC ($\sigma = 0.0032$).
3. **Audit Imbalance:** Penerapan `class_weight='balanced'` meningkatkan Recall Kelas 0 (nasabah berisiko) hingga $70.81\% - 70.83\%$, membuktikan keefektifan *Cost-Sensitive Learning* tanpa perlu manipulasi resampling sintetis (SMOTE).

---

## 9. Diagnosis Model, Seleksi Fitur, & Analisis Galat

### 9.1 Model Comparison & Pembuktian Bahaya *Accuracy Paradox*
* **DummyClassifier (Most Frequent):** Menghasilkan **Akurasi 64.16%** dengan **ROC-AUC murni acak (0.5000)** dan **Recall Kelas 0 = 0.00%**.
* **Implikasi Finansial:** Membuktikan secara empiris bahaya *Accuracy Paradox* di perbankan: model naif tampak memiliki akurasi lumayan (64.16%), namun meloloskan 100% nasabah berisiko gagal bayar yang berakibat pada kebangkrutan modal perbankan akibat kredit macet (*Non-Performing Loan*).

### 9.2 Kerangka Seleksi Fitur: Gini Importance vs Permutation Feature Importance (PFI)

Pengujian Permutation Feature Importance (PFI) dijalankan dengan 10 repetisi pengacakan pada data uji (*hold-out test set*, 10.400 sampel) untuk memverifikasi kontribusi objektif ke-52 fitur (23 numerik + 29 kategori one-hot):

#### Tabel Top 10 Fitur Hasil Evaluasi Feature Importance:
| Peringkat | Fitur | Gini Importance (MDI) | Permutation Importance (PFI) | Permutation Std | Kategori Sinyal & Peran |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Credit_Score** | 0.3197 | **0.1557** | ± 0.0037 | Driver Utama Dominan (~15.57% degradasi performa) |
| **2** | **Loan_Amount_Requested** | 0.1796 | **0.0078** | ± 0.0007 | Driver Sekunder (~0.78% degradasi performa) |
| **3** | **Age** | 0.0451 | **0.0002** | ± 0.0002 | Sinyal Demografis Minor (~0.02% degradasi) |
| **4** | **x5_Other (Residential_Status)** | 0.0007 | **0.0000** | ± 0.0000 | Kontribusi Marginal Mendekati Nol |
| **5** | **x1_Single (Marital_Status)** | 0.0008 | **0.0000** | ± 0.0000 | Bebas Bias Status Pernikahan |
| **6** | **x2_Graduate (Education)** | 0.0008 | **0.0000** | ± 0.0000 | Bebas Bias Tingkat Pendidikan |
| **7** | **x2_High School (Education)** | 0.0009 | **0.0000** | ± 0.0000 | Bebas Bias Tingkat Pendidikan |
| **8** | **x2_Postgraduate (Education)** | 0.0007 | **0.0000** | ± 0.0000 | Bebas Bias Tingkat Pendidikan |
| **9** | **x3_Employed (Employment_Status)** | 0.0007 | **0.0000** | ± 0.0000 | Kontribusi Marginal Mendekati Nol |
| **10** | **x3_Unemployed (Employment_Status)** | 0.0007 | **0.0000** | ± 0.0000 | Kontribusi Marginal Mendekati Nol |

#### Temuan Analisis & Interpretasi:
1. **Credit_Score sebagai Pilar Penentu Absolut:** Nilai **PFI = 0.1557** (Gini = 0.3197) menunjukkan bahwa pengacakan nilai `Credit_Score` meruntuhkan skor diskriminasi model secara drastis sebesar ~15.57%. Atribut ini adalah sinyal kredit paling kredibel pada *unseen data*.
2. **Loan_Amount_Requested sebagai Penyeimbang Plafon:** Berada di posisi kedua (**PFI = 0.0078** | Gini = 0.1796), membuktikan besaran plafon pinjaman yang diajukan menjadi pertimbangan material kedua setelah riwayat kelayakan kredit pemohon.
3. **Efek Penalti Usia (Age):** Menghasilkan **PFI = 0.0002** (Gini = 0.0451) yang merefleksikan tren penalti usia produktif vs usia lanjut.
4. **Efek Masking Multikolinearitas:** Fitur-fitur rasio likuiditas hasil *feature engineering* (seperti `Monthly_Disposable_Income` dan `Credit_Risk_Factor`) memperoleh bobot Gini yang substansial saat pohon dibangun, namun memiliki nilai PFI yang mendekati nol saat diacak secara individual karena informasinya telah terwakili secara redundan (*masked*) oleh `Credit_Score` dan `Loan_Amount_Requested`.
5. **Independensi Keadilan Demografis (*Model Fairness*):** Seluruh atribut demografis sensitif seperti Gender, Status Pernikahan (`x1_Single`), Pendidikan (`x2_Graduate`, dll.), dan Status Tempat Tinggal (`x5_Other`) memiliki nilai PFI sebesar **0.0000**, membuktikan model secara empiris tidak mendiskriminasi nasabah berdasarkan kelompok demografis terlindungi.

### 9.3 Diagnosis Bias-Variance (Learning Curve & Validation Curve)
1. **Learning Curve:** Train ROC-AUC berada di kisaran ~0.975–0.995 sementara Validation ROC-AUC mendatar di ~0.818 dari 2.500 hingga 27.000 sampel latih. Pola ini mengindikasikan karakteristik **High Variance (Overfitting)** di mana pohon acak menghafal data latih.
2. **Validation Curve (max_depth):** Peningkatan kedalaman pohon di atas 8 melesatkan performa latih mendekati 1.0, namun skor validasi mengalami saturasi (*plateau*) di kisaran **~0.820**.
* **Rekomendasi Arsitektur:** Titik optimal regulasi kedalaman pohon (*sweet spot*) berada di rentang **max_depth=6 hingga max_depth=8**.

### 9.4 Pembedahan Galat & Laporan Error Analysis
Dari total **10.400 data uji**, Random Forest menghasilkan **1.549 galat prediksi (Error Rate = 14.89%)**:
* **False Positives (FP): 1.088 kasus** (Nasabah macet yang keliru disetujui, risiko kerugian kredit).
* **False Negatives (FN): 461 kasus** (Nasabah layak yang keliru ditolak, risiko kehilangan bunga).

#### Distribusi Taksonomi Galat Berdasarkan Akar Masalah:
| Kategori Akar Masalah | Jumlah Kasus | Persentase | Karakteristik Pola & Tindakan Rekayasa |
| :--- | :---: | :---: | :--- |
| **1. Ketidakcukupan Interaksi Likuiditas Kas** | **1.241** | **80.12%** | Beban pengeluaran bulanan sangat tinggi terhadap sisa dana bebas dengan tenor pendek. Diperlukan fitur interaksi likuiditas multi-tahap. |
| **2. Kontradiksi Label Sintetis (Noise)** | **247** | **15.95%** | Profil finansial prima (Credit_Score tinggi & utang rendah) namun ditolak secara acak oleh generator data sintetis. |
| **3. Outlier Finansial Ekstrem** | **60** | **3.87%** | Nilai pendapatan atau usia lanjut (> 66 tahun) di luar batas IQR yang mengacaukan estimasi probabilitas pohon keputusan. |
| **4. Kasus Perbatasan (Borderline)** | **1** | **0.06%** | Distribusi probabilitas prediksi model terpolarisasi bimodal ekstrem ($P < 0.20$ atau $P > 0.70$); hanya terdapat 1 sampel galat di ambang batas ketidakpastian [$0.42, 0.58$]. |

---

## 10. Evaluasi Komprehensif: Benchmark Train vs Test Seluruh Model

Pengujian komparatif menyeluruh (*apple-to-apple*) dijalankan menggunakan data latih (41.600 sampel) dan data uji independen (10.400 sampel) dengan Stratified 5-Fold Cross-Validation:

| Model Architecture | Train Accuracy | Test Accuracy | Train ROC-AUC | Test ROC-AUC | CV ROC-AUC (Mean ± Std) | Test F1-Weighted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.6420 | 0.6416 | 0.5000 | 0.5000 | 0.5000 ± 0.0000 | 0.5016 |
| **Logistic Regression** | 0.8500 | **0.8512** | 0.8163 | 0.8158 | 0.8164 ± 0.0043 | **0.8477** |
| **Decision Tree (depth 8)** | 0.8520 | 0.8475 | 0.8471 | 0.8202 | 0.8124 ± 0.0050 | 0.8442 |
| **Random Forest (depth 12)** | 0.8510 | **0.8511** | 0.8660 | 0.8200 | 0.8173 ± 0.0032 | **0.8476** |
| **Hist Gradient Boosting** | 0.8502 | 0.8510 | 0.8436 | **0.8203** | **0.8196 ± 0.0046** | 0.8474 |

### Analisis Evaluasi:
1. **Generalization Gap yang Ketat:** Selisih akurasi latih dan uji seluruh model berada di bawah **0.45%**, membuktikan bahwa restriksi regularisasi berhasil mencegah memorisasi data.
2. **Efektivitas Model Boosting:** Hist Gradient Boosting mencatatkan skor validasi silang tertinggi (**CV ROC-AUC = 0.8196 ± 0.0046**) dan **Test ROC-AUC = 0.8203**.
3. **Sensitivitas Risiko Tinggi:** Seluruh model ML dengan `class_weight='balanced'` sukses mendeteksi $> 70.8\%$ nasabah berisiko gagal bayar (Recall Kelas 0 = 70.81% s.d. 70.83%).

---

## 11. Tata Kelola Pemodelan (Model Governance), Kesimpulan Akhir, & Rencana Lanjutan

### 11.1 Format Baku Experiment Tracking Log (EXP-001 s.d. EXP-005)

| Exp ID | Architecture | Hyperparameter Utama | CV ROC-AUC (Mean ± Std) | Test Acc | Test AUC | Test F1 | Status & Keputusan |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **EXP-001** | DummyClassifier | `strategy='most_frequent'` | 0.5000 ± 0.0000 | 0.6416 | 0.5000 | 0.5016 | Baseline Naif (*Rejected - Accuracy Paradox*) |
| **EXP-002** | Logistic Regression | `max_iter=1000, balanced` | 0.8164 ± 0.0043 | 0.8512 | 0.8158 | 0.8477 | Linear Baseline (*Accepted Benchmark*) |
| **EXP-003** | Decision Tree | `max_depth=8, balanced` | 0.8124 ± 0.0050 | 0.8475 | 0.8202 | 0.8442 | Single Tree (*High FN - Whitebox Reference*) |
| **EXP-004** | Random Forest | `n_estimators=100, depth=12, balanced` | 0.8173 ± 0.0032 | 0.8511 | 0.8200 | 0.8476 | Ensemble Bagging (*Production Candidate 1*) |
| **EXP-005** | HistGradientBoosting | `max_iter=100, depth=6, balanced` | 0.8196 ± 0.0046 | 0.8510 | 0.8203 | 0.8474 | Ensemble Boosting (*Production Candidate 2*) |

---

### 11.2 Model Card Standar Industri

#### Model Details
- **Project ID:** A10 - Persetujuan Pinjaman
- **Pengembang / Peneliti:** Tim Proyek A10:
  1. Alvin Dinata (`231111420`)
  2. James (`231111512`)
  3. Vera Nataria (`231110030`)
- **Institusi / Semester:** Universitas Mikroskil / Semester Ganjil TA 2026/2027
- **Tanggal Rilis:** November 2026
- **Versi Model:** 1.0.0 (Tahap UTS Milestone 1)
- **Arsitektur Model:** Random Forest Classifier & Hist Gradient Boosting Classifier (`scikit-learn 1.3+`)
- **Lisensi & Dependensi:** Open Source / Python 3.10+, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

#### Intended Use & Users
- **Kegunaan Utama (*Primary Use*):** Sistem pendukung keputusan underwriting kredit perbankan (*Credit Underwriting Decision Support System*) untuk penyaringan awal kelayakan pemohon pinjaman personal.
- **Pengguna yang Dituju (*Target Users*):** Credit Risk Officers, Analis Kredit Perbankan, dan Pengelola Portofolio Pembiayaan.
- **Di Luar Lingkup (*Out-of-Scope*):** Tidak diperkenankan untuk eksekusi penolakan atau persetujuan pinjaman 100% otomatis tanpa supervisi manusia (*Human-in-the-Loop*), khususnya pada profil yang mengandung anomali data kerja atau outlier pendapatan.

#### Quantitative Factors & Fairness
- **Faktor Prediktif Utama:** Credit_Score, Loan_Amount_Requested, Annual_Income, dan rasio likuiditas kas.
- **Aspek Keadilan (*Fairness*):** Atribut sensitif terlindungi (*Gender*, *Marital_Status*) terbukti memiliki *Permutation Feature Importance* bernilai nol ($\le 0.0001$), membuktikan bahwa model tidak melakukan diskriminasi demografis.

---

### 11.3 Kesimpulan Akhir & Rencana Aksi Lanjutan

1. **Kesimpulan Teknis:**
   - Rekayasa rasio likuiditas finansial terbukti berhasil mengekstrak sinyal prediktif penting dari dataset perbankan.
   - Model ensemble (*Random Forest* dan *Hist Gradient Boosting*) menunjukkan performa generalisasi dan kestabilan paling unggul dengan variansi minimal lintas 5-fold CV ($\sigma \le 0.0046$).
2. **Strategic Action Items untuk Milestone Selanjutnya:**
   - **Threshold Tuning Berbasis Biaya Finansial (*Cost-Sensitive Cut-off*):** Menggeser ambang batas persetujuan dari 0.50 menjadi $\ge 0.65$ untuk memangkas 1.088 kasus False Positive (risiko kerugian modal akibat kredit macet).
   - **Hyperparameter Tuning Terfokus:** Menjalankan optimasi Bayesian / RandomizedSearchCV pada parameter min_samples_leaf, l2_regularization, dan learning_rate pada Hist Gradient Boosting.
   - **Implementasi Explainable AI (XAI):** Menerapkan integrasi SHAP (*SHapley Additive exPlanations*) untuk menyediakan laporan alasan penolakan/persetujuan kredit individual yang memenuhi regulasi keterbukaan perbankan (*Adverse Action Notices*).
