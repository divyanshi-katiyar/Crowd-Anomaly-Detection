# Crowd Anomaly Detection using CNN Autoencoder, Optical Flow & ConvLSTM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research Notice:** *Crowd Anomaly Detection using CNN Autoencoder and ConvLSTM* was developed as an unsupervised deep learning surveillance framework for high-density public spaces.

An end-to-end Computer Vision and Deep Learning framework for automated anomaly detection in high-density crowd surveillance videos. The system learns normal crowd appearance and motion dynamics in an **unsupervised manner**, identifying anomalous events (e.g., panics, sudden stampedes, irregular crowd dispersals) via spatial reconstruction error, optical flow motion deviation, and spatio-temporal sequence modeling.

---

## 📌 Overview

High-density crowd management in public spaces—such as transit hubs, stadiums, airports, and religious gatherings—demands rapid and automated surveillance analytics. Continuous human monitoring of multi-camera CCTV feeds is error-prone, labor-intensive, and susceptible to cognitive fatigue during critical incidents.

Supervised anomaly detection is impractical for real-world surveillance because abnormal crowd events are rare, unpredictable, and difficult to exhaustively label. This project resolves this by adopting an **unsupervised anomaly detection paradigm**:
1. Train deep representation models exclusively on normal, non-anomalous crowd sequences.
2. Measure test-time deviations in spatial appearance and temporal motion.
3. Compute calibrated anomaly scores using adaptive thresholding and temporal windowing to trigger automated alerts.

---

## 🌟 Key Features

- **Spatial Representation Learning:** Deep Convolutional Autoencoder (CNN AE) for appearance-level reconstruction and spatial anomaly localization.
- **Dense Motion Estimation:** Farneback Optical Flow pipeline measuring instantaneous frame-to-frame velocity and direction.
- **Spatio-Temporal Modeling:** ConvLSTM Autoencoder preserving 2D spatial features while capturing long-term temporal dependencies across frame sequences.
- **Hybrid Multi-Modal Fusion:** Weighted score fusion ($0.4 \times \text{Appearance} + 0.6 \times \text{Motion}$) combining spatial and dynamic signals.
- **Temporal Segment Detection:** Sliding-window aggregation with adaptive percentile-based anomaly thresholding.
- **Modular & Config-Driven:** Clean Python package with YAML-based hyperparameters, standalone CLI scripts, and reproducible Jupyter notebook workflow.

---

## 🏗️ System Architecture

```
                       Input Surveillance Video / CCTV
                                      │
                                      ▼
                        Frame Extraction & Preprocessing
                        (Resize to 256x256, [0, 1] Norm)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
      Spatial Feature Learning                   Dense Motion Modeling
     (CNN Autoencoder: MSE Error)             (Farneback Optical Flow: Mag)
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                            Hybrid Feature Fusion
                        (0.4 x AE + 0.6 x Optical Flow)
                                      │
                                      ▼
                          ConvLSTM Sequence Modeling
                         (Spatio-Temporal Recurrence)
                                      │
                                      ▼
                        Score Normalization & Smoothing
                        (Z-Score + 1D Gaussian Filter)
                                      │
                                      ▼
                          Adaptive Thresholding (95%)
                         & Sliding Window (Window = 30)
                                      │
                                      ▼
                         Quantitative & Visual Output
                      (Precision, Recall, F1, IoU, Plots)
```

---

## 📂 Repository Structure

```
Crowd-Anomaly-Detection/
├── .gitignore                      # Git ignore rules for ML checkpoints & data
├── LICENSE                         # MIT License
├── README.md                       # Repository documentation & benchmarks
├── requirements.txt                # Python package dependencies
│
├── configs/
│   └── config.yaml                 # Centralized configuration & hyperparameters
│
├── notebooks/
│   └── crowd_anomaly_detection.ipynb # Research & exploratory notebook
│
├── results/
│   └── .gitkeep                    # Target directory for evaluation plots & logs
│
├── scripts/
│   ├── train.py                    # Standalone CLI for model training
│   └── evaluate.py                 # Standalone CLI for evaluation & benchmarking
│
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py              # Frame & Sequence PyTorch Datasets
    │   └── preprocessing.py        # Optical flow, Z-score, Gaussian smoothing
    ├── models/
    │   ├── __init__.py
    │   ├── cnn_autoencoder.py      # Convolutional Autoencoder architecture
    │   └── convlstm.py             # ConvLSTM Cell & Spatio-Temporal Autoencoder
    ├── training/
    │   ├── __init__.py
    │   └── train.py                # PyTorch training loops & device handling
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluate.py             # Anomaly scoring, metrics & detection logic
    └── utils/
        ├── __init__.py
        ├── config.py               # YAML configuration loader
        └── visualization.py        # Time-series, GT overlay & heatmap plotting
```

---

## 🧠 Model Architectures

### 1. CNN Autoencoder (`CrowdAutoencoder`)
* **Encoder:** 3 convolutional layers ($3 \to 16 \to 32 \to 64$ filters, $3\times 3$ kernel, stride 2, padding 1, ReLU activations) downsampling a $256\times 256$ input frame into a compact $32\times 32\times 64$ latent representation.
* **Decoder:** 3 transposed convolutional layers ($64 \to 32 \to 16 \to 3$ channels, $4\times 4$ kernel, stride 2, padding 1, ReLU + Sigmoid) reconstructing the input frame.
* **Objective:** Mean Squared Error (MSE). Normal crowd appearances yield low reconstruction loss; abnormal crowd objects or configurations produce elevated residuals.

### 2. Dense Optical Flow (Farneback Algorithm)
* Calculates motion vectors $(\Delta x, \Delta y)$ across consecutive grayscale frames:
  $$\text{mag}(x, y) = \sqrt{(\Delta x)^2 + (\Delta y)^2}$$
* Frame-level motion intensity is quantified via mean magnitude across all spatial pixels, capturing sudden directional changes and rush dynamics.

### 3. Spatio-Temporal ConvLSTM (`ConvLSTMAutoencoderV2`)
* Combines spatial convolutional feature extraction with recurrent gating (Input, Forget, Output, and Cell state gates) in a 2D topology:
  $$i_t = \sigma(W_{xi} * X_t + W_{hi} * \mathcal{H}_{t-1} + b_i)$$
  $$f_t = \sigma(W_{xf} * X_t + W_{hf} * \mathcal{H}_{t-1} + b_f)$$
  $$\mathcal{C}_t = f_t \odot \mathcal{C}_{t-1} + i_t \odot \tanh(W_{xc} * X_t + W_{hc} * \mathcal{H}_{t-1} + b_c)$$
  $$o_t = \sigma(W_{xo} * X_t + W_{ho} * \mathcal{H}_{t-1} + b_o)$$
  $$\mathcal{H}_t = o_t \odot \tanh(\mathcal{C}_t)$$
* Operates on sequences of $T=10$ frames, decoding representations back to RGB space at each timestep.

---

## 📊 Experimental Results & Benchmarks

All metrics below represent verified empirical results evaluated on the standard Times Square test sequence ($1026$ frames) with ground truth anomaly interval $[780, 830]$:

### 1. Hybrid Fusion Model Performance ($0.4 \times \text{AE} + 0.6 \times \text{Flow}$)

| Metric | Measured Value | Description |
|---|:---:|---|
| **Precision** | **$0.6076$** | Ratio of correctly identified anomaly frames to total predicted anomaly frames |
| **Recall** | **$0.9600$** | Detection coverage across true anomaly ground-truth interval |
| **F1-Score** | **$0.7442$** | Harmonic mean of precision and recall |
| **Frame IoU** | **$0.5926$** | Intersection over Union at the frame classification level |
| **Temporal IoU** | **$0.6000$** | Maximum 1D temporal segment overlap with ground-truth window |
| **Detection Delay** | **$2\text{ frames}$** | Detection triggered at frame $782$ (Ground Truth onset at frame $780$) |

### 2. Multi-Model Signal Comparison

| Model Pipeline | Detection Delay | Peak Anomaly Strength ($\max(S) - \mu(S)$) |
|---|:---:|:---:|
| **CNN Autoencoder (Appearance)** | $56\text{ frames}$ | $1.6048$ |
| **Optical Flow (Motion)** | *N/A (Adaptive)* | $0.9857$ |
| **Hybrid Fusion (AE + Flow)** | **$2\text{ frames}$** | $0.9034$ |
| **ConvLSTM (Spatio-Temporal)** | *N/A (Adaptive)* | **$1.7925$** |

### 3. Model Training Convergence
* **CNN Autoencoder:** Loss decreased monotonically from $0.041313$ (Epoch 1) to $0.012396$ (Epoch 10).
* **ConvLSTM Autoencoder:** Sequence MSE loss converged from $0.032975$ (Epoch 1) to $0.009195$ (Epoch 8).

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA (recommended) or Apple Silicon / CPU

### 1. Clone Repository
```bash
git clone https://github.com/avijain1412/Crowd-Anomaly-Detection.git
cd Crowd-Anomaly-Detection
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

The project is designed and evaluated on benchmark high-density crowd surveillance datasets (e.g., **Abnormal High-density Crowds** dataset):

Organize your dataset directory structure as follows:
```
data/
└── 1_Times_Square/
    └── View_1/
        ├── Train/    # Normal crowd frames (e.g., 379 frames: 001.png, 002.png, ...)
        └── Test/     # Test frames containing anomalous events (e.g., 1026 frames)
```

Configure the dataset paths in `configs/config.yaml` or provide them directly via CLI flags `--data_path`.

---

## 🚀 Usage Guide

### 1. Train CNN Autoencoder (Appearance Model)
```bash
python scripts/train.py --model autoencoder --epochs 10 --batch_size 8 --lr 0.001
```

### 2. Train ConvLSTM Autoencoder (Spatio-Temporal Model)
```bash
python scripts/train.py --model convlstm --epochs 8 --batch_size 2 --lr 0.0001
```

### 3. Run Full Pipeline Evaluation
```bash
python scripts/evaluate.py --config configs/config.yaml --results_dir results
```
This executes:
1. Reconstruction error computation for CNN AE
2. Dense Farneback optical flow calculation
3. Hybrid multi-modal score fusion ($0.4 \times \text{AE} + 0.6 \times \text{Flow}$)
4. Spatio-temporal ConvLSTM sequence evaluation
5. Generation of benchmark plots and `evaluation_metrics.json` saved in `results/`.

### 4. Interactive Research Notebook
To explore the original interactive workflow and visual explanations:
```bash
jupyter notebook notebooks/crowd_anomaly_detection.ipynb
```

---

## ⚙️ Configuration Reference (`configs/config.yaml`)

```yaml
data:
  train_path: "data/1_Times_Square/View_1/Train"
  test_path: "data/1_Times_Square/View_1/Test"
  image_size: [256, 256]
  sequence_length: 10

model_ae:
  in_channels: 3
  epochs: 10
  batch_size: 8
  learning_rate: 0.001
  checkpoint_path: "results/checkpoints/crowd_autoencoder.pt"

model_convlstm:
  in_channels: 3
  hidden_dim: 64
  kernel_size: 3
  epochs: 8
  batch_size: 2
  learning_rate: 0.0001
  checkpoint_path: "results/checkpoints/convlstm_autoencoder.pt"

detection:
  gaussian_sigma: 10.0
  window_size: 30
  threshold_percentile: 95.0
  fusion_weight_app: 0.4
  fusion_weight_mot: 0.6

ground_truth:
  gt_start: 780
  gt_end: 830
```

---

## 🔮 Future Improvements

- [ ] **Transformer-based Temporal Modeling:** Explore Video Vision Transformers (ViViT / TimeSformer) for self-attention across long temporal spans.
- [ ] **Spatial Bounding-Box Localization:** Integrate YOLO object detection to localize anomalies per-individual/per-cluster within frames.
- [ ] **Edge AI & TensorRT Deployment:** Optimize model inference latency using ONNX and TensorRT for real-time edge processing on Jetson devices.
- [ ] **Multi-Camera Temporal Alignment:** Extend the fusion engine to aggregate multi-perspective camera angles across complex facilities.

