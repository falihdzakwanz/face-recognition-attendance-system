# 🎓 Penjelasan Arsitektur Model Face Recognition

## Untuk Presentasi

---

## 📋 Overview Sistem

Sistem ini menggunakan **2 arsitektur Deep Learning**:

1. **CNN (FaceNet + ArcFace)** - Model utama dengan akurasi tinggi
2. **Transformer (DeiT)** - Model alternatif dengan arsitektur modern

---

## 🧠 Arsitektur 1: CNN (FaceNet + ArcFace)

### Diagram Alur

```
Input Image (224×224×3)
         ↓
┌────────────────────────┐
│  Face Detection (MTCNN) │
│  - Detect face location │
│  - Align & crop face    │
└────────────────────────┘
         ↓
┌────────────────────────────────────┐
│   InceptionResNetV1 Backbone       │
│   (Pretrained on VGGFace2)         │
│   - 380 layer groups               │
│   - Transfer Learning              │
│   - Freeze 360 layers              │
│   - Fine-tune 20 last layers       │
└────────────────────────────────────┘
         ↓
┌────────────────────────┐
│  Global Average Pooling │
└────────────────────────┘
         ↓
┌────────────────────────────────┐
│  Fully Connected Layer          │
│  512 → 128 dimensions           │
│  + Dropout (0.5)                │
└────────────────────────────────┘
         ↓
┌────────────────────────┐
│  L2 Normalization       │
│  (Unit hypersphere)     │
└────────────────────────┘
         ↓
┌─────────────────────────────────┐
│      ArcFace Loss Layer         │
│  - Angular Margin: 0.5          │
│  - Scale Factor: 30.0           │
│  - Enhance feature separation   │
└─────────────────────────────────┘
         ↓
    Output: 70 classes
    (Predicted Student)
```

---

### Komponen Detail

#### 1. **Preprocessing (MTCNN)**

- **Multi-task Cascaded CNN** untuk deteksi wajah
- 3 stage cascade: P-Net → R-Net → O-Net
- Output: Face bounding box + 5 facial landmarks
- Alignment: Rotate & crop face berdasarkan eye positions

**Mengapa MTCNN?**

- ✅ Akurat untuk berbagai pose & lighting
- ✅ Detect + align sekaligus
- ✅ Real-time performance

---

#### 2. **Backbone: InceptionResNetV1**

**Struktur:**

```
Inception Modules (Parallel Convolutions)
├── 1×1 Conv (dimensionality reduction)
├── 1×1 → 3×3 Conv (spatial features)
├── 1×1 → 5×5 Conv (larger spatial context)
└── 3×3 MaxPool → 1×1 Conv (pooling branch)
```

**Keunggulan:**

- Multi-scale feature extraction (1×1, 3×3, 5×5)
- Efficient computation dengan 1×1 conv
- Residual connections untuk gradient flow

**Transfer Learning:**

- Pretrained pada VGGFace2 (3.3M images, 9K identities)
- Fine-tuning: freeze 360/380 layers, train 20 layers terakhir
- Hasil: 27.9M total params, hanya 7M trainable (25%)

---

#### 3. **Embedding Layer**

**128-dimensional Face Embeddings**

```python
Features (512-dim) → FC(512→128) + Dropout → L2 Norm → Embeddings
```

**Karakteristik:**

- Compact representation (128-dim)
- Unit hypersphere (||embedding|| = 1)
- Semantic meaning: similar faces → similar vectors
- Cosine similarity untuk matching

---

#### 4. **ArcFace Loss**

**Formula:**

```
L = -log( exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ exp(s·cos(θ_j))) )
```

Dimana:

- `s = 30.0` (scale factor)
- `m = 0.5` (angular margin)
- `θ_yi` = angle between embedding & correct class weight

**Visualisasi Konsep:**

```
Traditional Softmax:          ArcFace:
   Class A                      Class A
      ↑                            ↑
      |                            |  (margin m added)
  ----*---- (decision)         ----*---┐
      |                            |   |margin
      ↓                            ↓   ↓
   Class B                      Class B

   → Easier decision           → Harder training = better features
```

**Mengapa ArcFace?**

- ✅ **Intra-class compactness**: Wajah yang sama jadi lebih dekat
- ✅ **Inter-class separation**: Wajah berbeda lebih terpisah
- ✅ **Angular margin**: Decision boundary lebih ketat
- ✅ Hasil: 99.4% validation accuracy!

---

### Training Strategy

```
1. Load pretrained InceptionResNetV1 (VGGFace2)
2. Freeze backbone layers (360/380)
3. Initialize ArcFace layer randomly
4. Train dengan:
   - Optimizer: Adam (lr=0.0001)
   - Batch size: 32
   - Mixed Precision (AMP) untuk speed
   - Early stopping (patience=10)
   - Learning rate scheduler (patience=5, factor=0.5)
5. Data augmentation on-the-fly (20+ transforms)
```

**Hasil Training:**

- Train Loss: 1.24 (dengan margin → harder)
- Val Accuracy: **99.4%**
- Val F1-Score: **99.4%**
- Training time: ~40 menit (GPU)

---

## 🤖 Arsitektur 2: Transformer (DeiT)

### Diagram Alur

```
Input Image (224×224×3)
         ↓
┌──────────────────────────────┐
│   Patch Embedding             │
│   - Split: 14×14 patches      │
│   - Patch size: 16×16         │
│   - Flatten: 196 patches      │
│   - Linear projection: 384-dim│
└──────────────────────────────┘
         ↓
┌──────────────────────────────┐
│   Position Embedding          │
│   (Learnable 1D positional)   │
└──────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│   Transformer Encoder (12 layers)   │
│                                      │
│   For each layer:                    │
│   ┌───────────────────────────┐     │
│   │ Multi-Head Self-Attention  │     │
│   │ - Heads: 6                 │     │
│   │ - Dim per head: 64         │     │
│   └───────────────────────────┘     │
│            ↓                         │
│   ┌───────────────────────────┐     │
│   │ Layer Normalization        │     │
│   └───────────────────────────┘     │
│            ↓                         │
│   ┌───────────────────────────┐     │
│   │ Feed-Forward Network       │     │
│   │ - MLP: 384 → 1536 → 384   │     │
│   │ - GELU activation          │     │
│   └───────────────────────────┘     │
│            ↓                         │
│   ┌───────────────────────────┐     │
│   │ Layer Normalization        │     │
│   └───────────────────────────┘     │
└─────────────────────────────────────┘
         ↓
┌──────────────────────────────┐
│   Classification Head         │
│   - Take [CLS] token          │
│   - Linear: 384 → 70          │
└──────────────────────────────┘
         ↓
    Output: 70 classes
```

---

### Komponen Detail

#### 1. **Patch Embedding**

**Konsep:**

- Image 224×224 → Grid 14×14 patches (16×16 each)
- Total: 196 patches
- Setiap patch di-flatten & project ke 384-dim

**Analogi:**

> Image = Kalimat, Patches = Kata-kata

```
Original Image:        Patches:
┌───────────┐         ┌─┬─┬─┬─┐
│           │   →     ├─┼─┼─┼─┤  (14×14 grid)
│   Face    │         ├─┼─┼─┼─┤
│           │         └─┴─┴─┴─┘
└───────────┘         Each = 16×16 pixels
```

---

#### 2. **Self-Attention Mechanism**

**Multi-Head Self-Attention (6 heads):**

```
Query, Key, Value dari input
    ↓
Attention(Q,K,V) = Softmax(Q·K^T / √d_k) · V
    ↓
Concat all heads → Linear projection
```

**Intuisi:**

- Setiap patch "attend" ke patches lain
- Model belajar: "mata", "hidung", "mulut" → face features
- Global context (tidak seperti CNN yang local)

**Keunggulan:**

- Long-range dependencies
- Position-invariant
- Interpretable attention maps

---

#### 3. **DeiT Distillation**

**Data-Efficient Image Transformer:**

```
Teacher Model (CNN)  →  Knowledge Distillation  →  Student (Transformer)
   ↓                                                      ↓
Hard labels                                         Soft labels
+ Distillation token                                Learn faster
```

**Hasil:**

- Converge lebih cepat dengan dataset kecil
- Pretrained on ImageNet → fine-tune untuk faces

---

## 📊 Perbandingan Kedua Arsitektur

| Aspek             | CNN (FaceNet+ArcFace)                | Transformer (DeiT)                  |
| ----------------- | ------------------------------------ | ----------------------------------- |
| **Paradigma**     | Convolution (local)                  | Attention (global)                  |
| **Parameters**    | 28M (7M trainable)                   | 22M                                 |
| **Input Size**    | 224×224                              | 224×224                             |
| **Accuracy**      | **99.4%** ⭐                         | ~75-85%                             |
| **Training Time** | 40 min (GPU)                         | 60 min (GPU)                        |
| **Inference**     | Fast (~50ms)                         | Slower (~100ms)                     |
| **Pretrain**      | VGGFace2 (faces)                     | ImageNet (objects)                  |
| **Strengths**     | ✅ High accuracy<br>✅ Face-specific | ✅ Global context<br>✅ Modern arch |
| **Weakness**      | ❌ Local receptive field             | ❌ Need more data<br>❌ Slower      |

**Kesimpulan:**

- **CNN** lebih cocok untuk face recognition (99.4% acc)
- **Transformer** sebagai baseline pembanding & research

---

## 🔄 Pipeline Inference (Real-time)

```
Webcam/Upload Image
        ↓
┌──────────────────┐
│ MTCNN Detection  │ ← Detect face(s)
└──────────────────┘
        ↓
   Face aligned
        ↓
┌──────────────────┐
│ Preprocessing    │ ← Resize 224×224, normalize
└──────────────────┘
        ↓
┌──────────────────┐
│ Model Inference  │ ← CNN forward pass
│ (FaceNet+ArcFace)│
└──────────────────┘
        ↓
   128-dim embedding
        ↓
┌──────────────────┐
│ Cosine Similarity│ ← Compare with class weights
│ → Softmax        │
└──────────────────┘
        ↓
   Confidence score (0-1)
        ↓
┌──────────────────┐
│ Threshold Check  │ ← Default: 55%
└──────────────────┘
        ↓
  If > threshold:  Student Name
  Else:            "Unknown"
        ↓
┌──────────────────┐
│ Attendance Log   │ ← Save to CSV (cooldown: 5 min)
└──────────────────┘
```

**Performance:**

- Detection: ~20-30ms (MTCNN)
- Inference: ~30-50ms (CNN)
- **Total: ~50-80ms per frame** (12-20 FPS)

---

## 🎯 Data Augmentation Pipeline

**On-the-fly transformations (20+ types):**

```python
Augmentation Pipeline:
├── Geometric
│   ├── Horizontal Flip (50%)
│   ├── Rotation (±15°)
│   ├── Shift/Scale/Rotate
│   └── Elastic Transform
├── Color
│   ├── Brightness (0.8-1.2)
│   ├── Contrast (0.8-1.2)
│   ├── Hue Shift (±20)
│   ├── RGB Shift
│   └── Channel Shuffle
├── Noise
│   ├── Gaussian Noise
│   ├── Gaussian Blur
│   └── Motion Blur
├── Quality
│   ├── JPEG Compression
│   ├── Image Compression
│   └── Downscale
└── Advanced
    ├── CLAHE (histogram equalization)
    ├── Grayscale (10%)
    ├── Coarse Dropout
    └── Grid Distortion
```

**Mengapa Heavy Augmentation?**

- Dataset kecil: ~4-5 foto/mahasiswa
- Generalization: robust ke lighting, pose, quality
- Simulate real-world conditions

---

## 📈 Training Results Visualization

### Loss Curves

```
Train Loss (CNN):
4.0 ┐
    │ ╲
3.0 ┤  ╲___
    │      ╲___
2.0 ┤          ╲___
    │              ╲___
1.0 ┤                  ╲___________
    │
0.0 └────────────────────────────────
    0    10    20    30    40    50 epochs

Val Accuracy:
100%┐              ___________________
    │         ____/
80% ┤     ___/
    │   _/
60% ┤  /
    │ /
40% ┤/
    │
0%  └────────────────────────────────
    0    10    20    30    40    50 epochs
```

**Observasi:**

- Train loss tinggi (1.2-1.4) karena angular margin
- Val accuracy cepat converge (epoch 10-15)
- Best model: Epoch 43, Val Acc: 99.4%

---

## 🔬 Ablation Study

**Percobaan yang dilakukan:**

| Config | Input Size | Loss         | Val Acc   | Note                      |
| ------ | ---------- | ------------ | --------- | ------------------------- |
| 1      | 160×160    | ArcFace      | **99.9%** | Best match (trained @160) |
| 2      | 224×224    | ArcFace      | **99.4%** | Final model (retrained)   |
| 3      | 224×224    | CrossEntropy | ~85%      | Tanpa ArcFace (baseline)  |
| 4      | 160×160    | Triplet      | ~90%      | Original FaceNet          |

**Kesimpulan:**

- ArcFace > CrossEntropy (+14% accuracy)
- Input size match training = critical
- Transfer learning essential (VGGFace2 pretrain)

---

## 💡 Key Innovations & Contributions

### 1. **Hybrid Architecture**

- CNN untuk accuracy
- Transformer untuk comparison
- Best of both worlds

### 2. **ArcFace Integration**

- State-of-the-art loss function
- Superior feature separation
- 99.4% accuracy pada dataset kecil

### 3. **Robust Preprocessing**

- MTCNN untuk detection
- Heavy augmentation (20+ transforms)
- Handle real-world variations

### 4. **Production-Ready**

- Real-time inference (50-80ms)
- User-friendly Gradio interface
- Automatic attendance logging
- Privacy-focused (cooldown, threshold)

### 5. **Deployment Flexibility**

- Local deployment
- Cloud (Hugging Face Spaces)
- Easy configuration (YAML)

---

## 📚 References & Theory

### Papers Implemented:

1. **FaceNet** (Schroff et al., 2015)

   - Triplet loss for face verification
   - 128-dim embeddings

2. **ArcFace** (Deng et al., 2019)

   - Additive Angular Margin Loss
   - State-of-the-art face recognition

3. **DeiT** (Touvron et al., 2021)

   - Data-efficient Image Transformers
   - Knowledge distillation

4. **MTCNN** (Zhang et al., 2016)
   - Joint face detection & alignment
   - Cascade architecture

---

## 🎯 Conclusion

**Sistem ini berhasil:**

- ✅ 99.4% accuracy (CNN model)
- ✅ Real-time inference (<100ms)
- ✅ 70 mahasiswa supported
- ✅ Production-ready application
- ✅ Privacy-focused design

**Technical Highlights:**

- Transfer learning from VGGFace2
- ArcFace loss untuk enhanced separability
- Heavy augmentation untuk small dataset
- Dual architecture comparison (CNN vs Transformer)

**Impact:**

- Automated attendance system
- Scalable architecture
- Open-source & reproducible

---