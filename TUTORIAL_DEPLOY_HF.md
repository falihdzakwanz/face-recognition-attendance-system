# 🚀 Tutorial Lengkap Deploy ke Hugging Face Spaces

## 📋 Daftar Isi

1. [Persiapan](#persiapan)
2. [Setup Hugging Face](#setup-hugging-face)
3. [Deploy Aplikasi](#deploy-aplikasi)
4. [Testing & Troubleshooting](#testing--troubleshooting)

---

## 🎯 Persiapan

### 1. File Deployment Sudah Siap! ✅

Script `prepare_hf_deployment.py` sudah membuat folder `hf_deployment/` berisi:

```
hf_deployment/
├── app.py                  # Entry point Gradio
├── README.md              # Dokumentasi Space
├── config.yaml            # Konfigurasi
├── requirements.txt       # Dependencies
├── .gitattributes         # Git LFS config
├── class_names.json       # List mahasiswa
├── src/                   # Source code
│   ├── models/
│   ├── preprocessing/
│   ├── app/
│   └── ...
├── dataset/Train/         # Folder kosong (70 kelas)
└── models/                # Model weights (~127 MB)
    └── cnn_arcface_.../
        └── best_model.pth
```

**Penting:**

- ✅ Model sudah included (~127 MB)
- ✅ Tidak ada foto mahasiswa (privacy!)
- ✅ Hanya folder kosong untuk get class names

---

## 🔑 Setup Hugging Face

### Step 1: Install Hugging Face CLI

```powershell
pip install huggingface_hub
```

### Step 2: Buat Account Hugging Face

1. Buka: https://huggingface.co/join
2. Daftar dengan email/GitHub
3. Verifikasi email

### Step 3: Generate Access Token

1. Login ke HF: https://huggingface.co/settings/tokens
2. Klik **"New token"**
3. Pilih:
   - Name: `deployment-token`
   - Type: **Write** (bisa push code)
4. Copy token (simpan aman!)

### Step 4: Login CLI

```powershell
huggingface-cli login
```

Paste token yang tadi di-copy, tekan Enter.

Output:

```
Token is valid (permission: write).
Your token has been saved to C:\Users\YourName\.cache\huggingface\token
Login successful
```

---

## 🚀 Deploy Aplikasi

### Step 1: Buat Space Baru

1. Buka: https://huggingface.co/spaces
2. Klik **"Create new Space"**
3. Isi form:
   ```
   Owner: [username-anda]
   Space name: face-recognition-presensi
   License: MIT
   Select SDK: Gradio
   Hardware: CPU Basic (free)
   Space visibility: Public / Private (pilih sesuai kebutuhan)
   ```
4. Klik **"Create Space"**

### Step 2: Clone Space Repository

```powershell
# Ganti YOUR_USERNAME dengan username HF Anda
git clone https://huggingface.co/spaces/YOUR_USERNAME/face-recognition-presensi
cd face-recognition-presensi
```

### Step 3: Setup Git LFS (untuk model file)

```powershell
# Install Git LFS jika belum
git lfs install

# Track file model
git lfs track "*.pth"
git lfs track "*.pkl"
git lfs track "*.bin"
```

### Step 4: Copy File Deployment

```powershell
# Copy semua file dari hf_deployment ke Space repo
# Ganti path sesuai lokasi project Anda

# Windows (PowerShell):
Copy-Item -Path "D:\Deep Learning\face recognition mahasiswa\hf_deployment\*" -Destination . -Recurse -Force

# Atau manual: copy paste folder isi hf_deployment/ ke folder Space
```

### Step 5: Commit & Push

```powershell
# Add all files
git add .

# Commit
git commit -m "Initial deployment: Face recognition system with CNN+ArcFace"

# Push ke Hugging Face
git push
```

**Note:** Upload model 127MB akan memakan waktu beberapa menit tergantung koneksi internet.

### Step 6: Tunggu Build

1. Buka Space Anda: `https://huggingface.co/spaces/YOUR_USERNAME/face-recognition-presensi`
2. HF akan otomatis:
   - Install dependencies dari `requirements.txt`
   - Build container
   - Launch aplikasi
3. Build memakan waktu **3-5 menit**
4. Status build bisa dilihat di tab **"Logs"**

### Step 7: Aplikasi Live! 🎉

Setelah build selesai:

- URL: `https://YOUR_USERNAME-face-recognition-presensi.hf.space`
- Aplikasi bisa diakses publik (atau private jika di-set private)

---

## 🧪 Testing & Troubleshooting

### Testing Aplikasi

1. Buka Space URL
2. Upload foto wajah mahasiswa
3. Adjust confidence threshold (default 55%)
4. Klik "Recognize"
5. Cek apakah nama terdeteksi dengan benar

### Common Issues & Solutions

#### ❌ Issue: Build Failed - Module Not Found

**Penyebab:** Dependencies tidak lengkap di `requirements.txt`

**Solusi:**

```powershell
# Update requirements.txt, tambahkan package yang kurang
git add requirements.txt
git commit -m "Update dependencies"
git push
```

#### ❌ Issue: Model File Upload Failed

**Penyebab:** Git LFS belum aktif atau file terlalu besar

**Solusi:**

```powershell
# Force push dengan LFS
git lfs push --all origin main

# Atau track file lagi
git lfs track "models/**/*.pth"
git add .gitattributes
git commit -m "Fix LFS tracking"
git push
```

#### ❌ Issue: Application Error - File Not Found

**Penyebab:** Path file salah (Windows vs Linux path)

**Solusi:** Cek di `app.py` dan pastikan pakai `Path` dari `pathlib`:

```python
from pathlib import Path
# Good: Path("config.yaml")
# Bad: "D:\\config.yaml"
```

#### ❌ Issue: Slow Inference (CPU)

**Penyebab:** Free tier pakai CPU, bukan GPU

**Solusi:**

1. Buka Space Settings
2. Hardware → Upgrade to GPU
3. Pilih: **CPU Upgrade (free)** atau **T4 small GPU ($0.60/jam)**
4. GPU bisa di-pause otomatis saat tidak dipakai

#### ❌ Issue: Privacy - Model Mengenali Wajah yang Salah

**Penyebab:** Model perlu retrain atau threshold terlalu rendah

**Solusi:**

- Adjust confidence threshold via slider (naikkan ke 70-80%)
- Retrain model dengan data lebih banyak
- Update model file di Space repo

---

## 📊 Monitoring & Analytics

### View Usage

1. Buka Space → Tab **"Analytics"**
2. Lihat:
   - Total visits
   - Active users
   - API calls

### View Logs

1. Tab **"Logs"**
2. Real-time logs aplikasi
3. Bisa debug error di sini

### Update Model

Jika ada model baru yang lebih baik:

```powershell
# Copy model baru ke Space repo
Copy-Item "models/cnn_new/best_model.pth" "models/cnn_arcface_*/best_model.pth" -Force

# Commit & push
git add models/
git commit -m "Update model to new version"
git push

# Space akan auto-rebuild & reload model
```

---

## 🎓 Tips & Best Practices

### 1. **Privacy & Security**

✅ **JANGAN upload foto mahasiswa** ke public repo  
✅ Set Space ke **Private** jika untuk internal kampus  
✅ Gunakan authentication jika perlu (HF Pro feature)

### 2. **Performance**

✅ Pakai GPU upgrade jika perlu inference cepat  
✅ Set model ke eval mode (sudah di-handle otomatis)  
✅ Batasi max concurrent users (via Gradio config)

### 3. **Cost Management**

✅ Free tier: CPU unlimited, tapi lambat  
✅ GPU: $0.60/jam, tapi bisa di-pause otomatis  
✅ Set auto-pause after 15 minutes idle

### 4. **Sharing**

✅ Share URL Space ke teman/dosen  
✅ Embed Space di website dengan iframe:

```html
<iframe
  src="https://YOUR_USERNAME-face-recognition-presensi.hf.space"
  width="100%"
  height="800px"
>
</iframe>
```

---

## 🔄 Alternatif: Temporary Share Link

Jika hanya mau **demo sementara** tanpa deploy permanent:

```powershell
# Di local machine, jalankan dengan share=True
python main.py --app --share

# Gradio akan generate link: https://xxxxx.gradio.live
# Link aktif 72 jam, gratis
```

**Kelebihan:**

- Instant (tidak perlu setup)
- Gratis
- Bisa langsung share

**Kekurangan:**

- Link expire setelah 72 jam
- Tergantung local machine jalan terus
- Koneksi lewat tunnel (lebih lambat)

---

## 📞 Support & Resources

- **HF Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://gradio.app/docs/
- **HF Forums:** https://discuss.huggingface.co/

---

## ✅ Checklist Deployment

- [ ] Jalankan `prepare_hf_deployment.py` ✅ (DONE)
- [ ] Install HF CLI: `pip install huggingface_hub`
- [ ] Login: `huggingface-cli login`
- [ ] Buat Space di https://huggingface.co/spaces
- [ ] Clone Space repo
- [ ] Setup Git LFS
- [ ] Copy files dari `hf_deployment/`
- [ ] Commit & push
- [ ] Tunggu build selesai
- [ ] Test aplikasi
- [ ] Share URL! 🎉

---

**Selamat! Aplikasi face recognition Anda sekarang live di cloud! 🚀**
