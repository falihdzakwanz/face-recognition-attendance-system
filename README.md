# Face Recognition Attendance System

> **Real-time face recognition system for automated student attendance using Deep Learning with CNN (FaceNet + ArcFace) architecture.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

An end-to-end face recognition attendance system designed for educational institutions, featuring:

- **🎓 Multi-Student Support**: Handles 70+ registered students
- **🤖 CNN Architecture**: InceptionResNetV1 (FaceNet) with ArcFace loss (DeiT Transformer for comparison)
- **📸 Real-Time Detection**: MTCNN for accurate face detection and alignment
- **💻 User-Friendly Interface**: Gradio-based web application
- **📊 High Accuracy**: 99.4% validation accuracy with ArcFace loss
- **🔒 Privacy-Focused**: Configurable confidence thresholds, local deployment

## 📋 Features

✅ **Face Detection & Alignment** using MTCNN  
✅ **Advanced Data Augmentation** (20+ transformations)  
✅ **CNN Model Architecture**: InceptionResNetV1 (FaceNet) with ArcFace loss  
✅ **Model Comparison**: Transformer (DeiT) available for benchmarking  
✅ **Automatic Attendance Logging** with cooldown prevention  
✅ **Real-time Webcam Support**  
✅ **Adjustable Confidence Threshold**  
✅ **Attendance History & Analytics**  
✅ **Easy Deployment** (Local or Cloud)

## 🏗️ Project Structure

```
face recognition mahasiswa/
├── dataset/                    # Dataset mahasiswa
│   ├── Train/                  # Data training (70 mahasiswa)
│   ├── Val/                    # Data validasi (akan dibuat)
│   └── Test/                   # Data testing (akan dibuat)
├── src/                        # Source code
│   ├── preprocessing/          # Data preprocessing & augmentation
│   │   ├── __init__.py
│   │   ├── heic_converter.py  # Convert HEIC ke JPEG
│   │   ├── face_detector.py   # Face detection & alignment (MTCNN)
│   │   ├── data_splitter.py   # Train/val/test split
│   │   └── augmentation.py    # Data augmentation pipeline
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   ├── cnn_facenet.py     # CNN FaceNet implementation
│   │   └── transformer_deit.py # Transformer DeiT implementation
│   ├── training/               # Training scripts
│   │   ├── __init__.py
│   │   ├── train_cnn.py       # CNN training
│   │   └── train_transformer.py # Transformer training
│   ├── evaluation/             # Evaluation & metrics
│   │   ├── __init__.py
│   │   └── evaluate.py        # Model evaluation & comparison
│   ├── app/                    # Desktop application
│   │   ├── __init__.py
│   │   ├── gradio_app.py      # Gradio interface
│   │   └── pyqt_app.py        # PyQt5 interface (alternative)
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── config_loader.py   # Config YAML loader
│       ├── logger.py          # Logging utilities
│       └── visualization.py   # Plotting & visualization
├── models/                     # Saved trained models
├── outputs/                    # Training outputs
│   ├── logs/                   # Training logs
│   ├── checkpoints/            # Model checkpoints
│   └── visualizations/         # Plots & confusion matrices
├── notebooks/                  # Jupyter notebooks untuk explorasi
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
└── main.py                     # Main entry point
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (optional, but recommended)
- Webcam for real-time recognition
- ~2GB disk space

### Installation

```powershell
# Clone repository
git clone https://github.com/YOUR_USERNAME/face-recognition-attendance-system.git
cd face-recognition-attendance-system

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install PyTorch with CUDA support (if GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_setup.py
```

### Usage

**All-in-One Pipeline:**

```powershell
# Run complete pipeline: preprocess → train → evaluate
python main.py --preprocess --train --evaluate
```

**Individual Steps:**

```powershell
# 1. Preprocess dataset (face detection & alignment)
python main.py --preprocess

# 2. Train models
python main.py --train                    # Train both CNN & Transformer
python main.py --train --skip-transformer # Train CNN only

# 3. Evaluate models
python main.py --evaluate

# 4. Launch web application
python main.py --app
```

**Accessing the Application:**

- Open browser: `http://localhost:7860`
- Upload image or use webcam
- Adjust confidence threshold (default: 55%)
- Enable auto-attendance marking

## 📊 Model Performance

### Primary Model: CNN (FaceNet + ArcFace)

- **Architecture**: InceptionResNetV1
- **Input Size**: 224×224
- **Validation Accuracy**: 99.4%
- **F1 Score**: 99.4%
- **Loss Function**: ArcFace (angular margin)
- **Training Time**: ~40 minutes (GPU)
- **Status**: ✅ Production model

### Comparison Model: Transformer (DeiT)

- **Architecture**: DeiT Small Patch16
- **Input Size**: 224×224
- **Parameters**: 22M
- **Pre-training**: ImageNet-1K
- **Training Time**: ~60 minutes (GPU)
- **Status**: 📊 Benchmarking only

## 📊 Model Architectures

### CNN Model (FaceNet + ArcFace)

- **Backbone**: InceptionResNetV1 (VGGFace2 pretrained)
- **Input Size**: 224×224
- **Embedding Size**: 128-dimensional
- **Loss Function**: ArcFace (Angular Margin Loss)
- **Accuracy**: 99.4% (validation)
- **Parameters**: ~28M total, 7M trainable

### Transformer Model (DeiT)

- **Architecture**: DeiT-Small Patch16
- **Pre-trained**: ImageNet-1K
- **Input Size**: 224×224
- **Loss Function**: Cross-Entropy
- **Parameters**: 22M
- **Attention Heads**: 6

## 🎯 Key Technologies

| Component             | Technology                      |
| --------------------- | ------------------------------- |
| **Face Detection**    | MTCNN (Multi-task Cascaded CNN) |
| **Data Augmentation** | Albumentations (20+ transforms) |
| **Deep Learning**     | PyTorch 2.7.1 + CUDA 11.8       |
| **Loss Function**     | ArcFace (margin=0.5, scale=30)  |
| **Web Interface**     | Gradio 4.44.0                   |
| **Training**          | Mixed Precision (AMP)           |

## 📈 Performance Metrics

Evaluation includes:

- ✅ **Accuracy**: Overall classification accuracy
- ✅ **Precision, Recall, F1-Score**: Per-class metrics
- ✅ **Confusion Matrix**: Visualization of predictions
- ✅ **Top-5 Accuracy**: Correct in top-5 predictions
- ✅ **Inference Time**: Real-time performance

## 🖥️ Web Application Features

- 📸 **Real-time Recognition**: Webcam or image upload support
- 🎯 **Confidence Threshold**: Adjustable (default: 55%)
- ✅ **Auto Attendance**: Automatic marking with cooldown (5 min)
- 👥 **Student Database**: View all registered students
- 📊 **Attendance Analytics**: History and statistics
- 📁 **Export Data**: CSV export for records
- 🎨 **User-Friendly UI**: Clean Gradio interface

## 📦 Dataset Information

- **Total Students**: 70 registered
- **Images per Student**: 4-8 photos (avg: 4-5)
- **Total Images**: ~680 training, 186 test
- **Formats**: JPG, PNG, WEBP
- **Split**: 75% train, 25% test
- **Augmentation**: On-the-fly (20+ transforms)

## 🚀 Deployment Options

### Local Deployment

```powershell
python main.py --app
# Access: http://localhost:7860
```

### Temporary Public Link

```powershell
python main.py --app --share
# Get temporary gradio.live link (72 hours)
```

## 🔧 Configuration

Customize settings in `config.yaml`:

- Model hyperparameters (learning rate, batch size, epochs)
- Data augmentation pipelines
- Face detection thresholds
- Training parameters (early stopping, learning rate schedule)
- Application settings (confidence threshold, cooldown)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FaceNet**: [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)
- **ArcFace**: [Deng et al., 2019](https://arxiv.org/abs/1801.07698)
- **DeiT**: [Touvron et al., 2021](https://arxiv.org/abs/2012.12877)
- **MTCNN**: [Zhang et al., 2016](https://arxiv.org/abs/1604.02878)
- **PyTorch** and **Hugging Face** communities

## 📞 Contact & Support

For questions, issues, or suggestions:

- Open an [Issue](https://github.com/YOUR_USERNAME/face-recognition-attendance-system/issues)
- Pull requests are welcome!


---

# Model Testing & Validation

To validate your trained model and get evaluation metrics (confusion matrix, accuracy, precision, recall, F1):

- Use the provided `test.py` script for robust evaluation.
- The script automatically finds the latest model and test folder.
- Results include per-image predictions, confusion matrix, metrics, per-class accuracy, and CSV export.

## Quickstart: Model Testing

See [QUICKSTART_TEST.md](docs/QUICKSTART_TEST.md) for step-by-step instructions.

## 📚 Documentation

- **[Architecture Details](docs/ARSITEKTUR_MODEL.md)** - Complete model architecture documentation
- **[Model Specifications](docs/MODELS_FINAL.md)** - Model comparison and specifications
- **[Quick Start Guide](docs/QUICKSTART.md)** - Getting started with the system
- **[Testing Guide](docs/QUICKSTART_TEST.md)** - Model validation and evaluation

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐
