# 💫 Facial Attractiveness Prediction API

A production-ready **Deep Learning API** that predicts human facial attractiveness scores (0–10) using state-of-the-art CNN architectures. Built with **FastAPI**, **PyTorch**, and **ONNX Runtime** for fast, scalable inference.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.18-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Architecture Support** | MobileNetV2, EfficientNet-B3, ResNet50 |
| 🧑‍🤝‍🧑 **Multi-Face Detection** | Detects and scores all faces in a single image using YuNet |
| 🚻 **Gender Classification** | Real-time gender detection via ONNX Runtime |
| 📊 **Calibrated Scoring** | Linear calibration for accurate 0–10 predictions |
| 🔁 **Resume Training** | Checkpoint-based training with automatic resume |
| 🧹 **Auto Dataset Cleaning** | Validates and removes corrupted/missing images |
| 🌐 **Web Interface** | Built-in glassmorphism UI for image upload & prediction |
| 🐳 **Docker Ready** | Production Dockerfile included |
| ☁️ **Cloud Compatible** | Deploy on AWS, Render, Railway, or Heroku |

---

## 🏗️ Architecture

```
attractiveness-api/
│
├── app/
│   ├── main.py                  # FastAPI entry point + config loader
│   ├── model_loader.py          # Model creation & weight loading
│   ├── trainer.py               # Training pipeline (OOP, AMP, checkpoints)
│   ├── routes/
│   │   └── predict.py           # /predict endpoint (batch + multi-face)
│   ├── datasets/
│   │   └── face_dataset.py      # Custom PyTorch Dataset
│   ├── static/
│   │   ├── index.html           # Web UI
│   │   ├── style.css            # Glassmorphism dark theme
│   │   └── script.js            # Frontend logic
│   └── utils/
│       ├── logger.py            # Centralized logging
│       ├── metrics.py           # MAE, RMSE, Accuracy
│       └── preprocessing.py     # Image transforms
│
├── scripts/
│   ├── train_model.py           # Training launcher
│   └── evaluate_model.py        # Evaluation script
│
├── config/
│   └── config.json              # All hyperparams & paths
│
├── models/                      # Trained weights (not in repo)
├── data/                        # Training data (not in repo)
├── Dockerfile                   # Production container
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/attractiveness-api.git
cd attractiveness-api

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Model Weights

Place your trained model files in the `models/` directory:
- `mobilenet_v2_best.pth` — Attractiveness regression model
- `face_detection_yunet_2023mar.onnx` — YuNet face detector
- `model_quantized.onnx` — Gender classification model

### 3. Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser: **http://127.0.0.1:8000**

---

## 🧠 Training

Train a model from scratch using your own dataset:

```bash
python -m scripts.train_model
```

**Training automatically:**
- Loads configuration from `config/config.json`
- Validates dataset and removes corrupted images
- Resumes from checkpoints if available
- Uses AMP (Mixed Precision) when GPU is available
- Saves best model weights based on lowest MAE

### Dataset Format

Place images in `data/images/` and create `data/labels.csv`:

```csv
filename,score
img_001.jpg,7.5
img_002.jpg,4.2
img_003.jpg,8.9
```

---

## 📈 Evaluation

```bash
python -m scripts.evaluate_model
```

Outputs:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **Accuracy within ±0.5** tolerance
- Results saved to `eval_results.csv`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface (upload & predict) |
| `POST` | `/predict` | Batch prediction (multipart form) |
| `GET` | `/docs` | Swagger UI (interactive API docs) |
| `GET` | `/redoc` | ReDoc documentation |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg"
```

### Example Response

```json
{
  "results": [
    {
      "filename": "photo1.jpg",
      "num_faces": 1,
      "face_detected": true,
      "faces": [
        {
          "face_index": 0,
          "bbox": [45, 30, 180, 200],
          "observed": 7.23,
          "fitted": 4.32,
          "fitted_constrained": 4.32,
          "gender": "Female",
          "gender_confidence": 0.94
        }
      ]
    }
  ],
  "batch_elapsed_seconds": 0.85
}
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t attractiveness-api .

# Run
docker run -p 8000:8000 attractiveness-api
```

---

## ⚙️ Configuration

All settings are managed via `config/config.json`:

```json
{
  "server": { "host": "0.0.0.0", "port": 8000 },
  "training": {
    "batch_size": 32,
    "epochs": 60,
    "lr": 0.0002,
    "image_size": 224
  },
  "models": {
    "model_name": "mobilenet_v2"
  }
}
```

Environment variables (`HOST`, `PORT`) override config values for cloud deployment.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Deep Learning** | PyTorch, TorchVision, ONNX Runtime |
| **Computer Vision** | OpenCV, YuNet Face Detector, Albumentations |
| **Data Science** | NumPy, Pandas, Scikit-learn, Matplotlib |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| **DevOps** | Docker, CI/CD Ready |

---

## 📝 License

This project is released under the **MIT License**.  
Free to use for personal and commercial purposes.

---

## 👨‍💻 Author

**Nisar Ahmad** — Full-Stack AI/ML Developer  
Built with ❤️ using PyTorch & FastAPI
