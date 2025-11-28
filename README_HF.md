---
title: Face Recognition Presensi Mahasiswa
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Face Recognition Presensi Mahasiswa

Sistem presensi mahasiswa menggunakan Deep Learning dengan 2 model:
- **CNN**: FaceNet (InceptionResNetV1) + ArcFace
- **Transformer**: DeiT (Data-efficient Image Transformer)

## Features

✅ Real-time face detection dengan MTCNN  
✅ Face recognition dengan confidence threshold  
✅ Auto attendance marking  
✅ Attendance history tracking  
✅ Support 70 mahasiswa

## Tech Stack

- **Deep Learning**: PyTorch, FaceNet, DeiT
- **Face Detection**: MTCNN
- **Framework**: Gradio
- **Training**: ArcFace loss, Mixed precision

## Usage

1. Upload foto atau gunakan webcam
2. Adjust confidence threshold (default: 55%)
3. Klik "Recognize" untuk identifikasi wajah
4. Enable "Auto Mark Attendance" untuk catat kehadiran otomatis

## Model Performance

- **CNN (224x224)**: Val Acc ~99.4%, F1 ~99.4%
- **Input Size**: 224x224 RGB
- **Classes**: 70 mahasiswa

## Privacy Note

Model ini di-train dengan dataset pribadi mahasiswa. Tidak ada foto mahasiswa yang di-upload ke repository ini (hanya model weights).

## Author

Deep Learning Project - Face Recognition System
