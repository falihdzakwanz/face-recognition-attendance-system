# Quick Start Guide

## 🚀 Getting Started

### 1. Setup Environment

⚠️ **IMPORTANT**: For GPU support, see detailed guide: `INSTALLATION_CUDA.md`

#### Option A: With CUDA GPU (Recommended)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 11.8 (FIRST!)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Install remaining dependencies
pip install -r requirements.txt
```

#### Option B: CPU Only (No GPU)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install PyTorch CPU version
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note**: Jika ada error dengan `dlib`, install via conda:
>
> ```powershell
> conda install -c conda-forge dlib
> ```
>
> **For different CUDA versions**: Read `INSTALLATION_CUDA.md`

### 2. Verify Configuration

```powershell
# Test configuration loader
python -c "from src.utils.config_loader import load_config; config = load_config(); print('Config OK!')"
```

### 3. Run Preprocessing (Step by Step)

#### 3.1 Scan Image Formats

```powershell
python -m src.preprocessing.heic_converter --scan-only --source dataset/Train
```

#### 3.2 Convert HEIC Files (if any)

```powershell
python -m src.preprocessing.heic_converter --source dataset/Train --quality 95
```

#### 3.3 Detect and Align Faces

```powershell
python -m src.preprocessing.face_detector --source dataset/Train --output dataset/Train_Aligned --size 160
```

#### 3.4 Split Dataset

```powershell
python -m src.preprocessing.data_splitter --source dataset/Train_Aligned --output dataset
```

#### 3.5 Verify Split

```powershell
python -m src.preprocessing.data_splitter --verify-only --output dataset
```

### 4. Run Complete Pipeline

```powershell
# Run full preprocessing
python main.py --preprocess

# Run training (after preprocessing)
python main.py --train

# Run evaluation
python main.py --evaluate

# Launch application
python main.py --app
```

### 5. Run Everything at Once

```powershell
python main.py --all
```

## 📊 Expected Results

### Preprocessing

- ✅ HEIC files converted to JPEG
- ✅ Faces detected and aligned (160x160 untuk CNN, 224x224 untuk Transformer)
- ✅ Dataset split: 70% train, 15% val, 15% test
- ✅ ~196 images untuk training (~2.8 per mahasiswa)
- ✅ ~42 images untuk validation (~0.6 per mahasiswa)
- ✅ ~42 images untuk testing (~0.6 per mahasiswa)

### Training

- ✅ CNN (FaceNet) trained dengan triplet loss
- ✅ Transformer (DeiT) trained dengan cross-entropy
- ✅ Model checkpoints saved di `models/`
- ✅ Training logs saved di `outputs/logs/`

### Evaluation

- ✅ Confusion matrix untuk kedua model
- ✅ Metrics: Accuracy, Precision, Recall, F1-Score
- ✅ Per-student accuracy analysis
- ✅ Model comparison visualization

## 🔧 Troubleshooting

### Problem: No module named 'yaml'

```powershell
pip install pyyaml
```

### Problem: MTCNN installation fails

```powershell
pip install mtcnn tensorflow
```

### Problem: Pillow-HEIF not working

```powershell
pip install pillow-heif
```

### Problem: Out of Memory during training

Edit `config.yaml`:

```yaml
cnn_model:
  training:
    batch_size: 16 # Reduce from 32

transformer_model:
  training:
    batch_size: 8 # Reduce from 16
```

### Problem: CUDA not available

Sistem akan otomatis menggunakan CPU. Untuk enable CUDA:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Next Steps

1. **Preprocessing** ✅ (Selesai)

   - HEIC conversion
   - Face detection & alignment
   - Dataset splitting

2. **Model Implementation** (In Progress)

   - [ ] Implement CNN FaceNet model
   - [ ] Implement Transformer DeiT model
   - [ ] Training scripts
   - [ ] Evaluation metrics

3. **Desktop Application** (TODO)

   - [ ] Gradio interface
   - [ ] Real-time face recognition
   - [ ] Attendance logging

4. **Testing & Optimization** (TODO)
   - [ ] Model performance tuning
   - [ ] Cross-validation
   - [ ] Deployment

## 📖 Documentation

- **README.md**: Project overview
- **config.yaml**: Configuration file (edit untuk customize)
- **requirements.txt**: Dependencies
- **main.py**: Main entry point

## 💡 Tips

1. **Dataset Quality**: Pastikan semua foto wajah terlihat jelas
2. **Augmentation**: Heavy augmentation karena dataset kecil (hanya 4 foto/mahasiswa)
3. **Model Selection**: CNN (FaceNet) kemungkinan akan perform lebih baik
4. **Validation**: Monitor validation metrics untuk detect overfitting

## 🎯 Expected Performance

- **CNN (FaceNet)**: 85-95% accuracy (expected)
- **Transformer (DeiT)**: 75-90% accuracy (expected)
- **Training Time**:
  - CNN: ~1-2 jam (dengan GPU)
  - Transformer: ~2-4 jam (dengan GPU)

## 📞 Support

Jika ada issues, check:

1. Error logs di `outputs/logs/`
2. Configuration di `config.yaml`
3. Dataset structure di `dataset/`

---

**Good luck! 🚀**
