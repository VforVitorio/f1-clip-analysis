# 🏎️ F1 CLIP Analysis

Comparing Pre-CLIP and CLIP multimodal embeddings on a Formula 1 dataset.

## 📋 Overview

This project analyzes image-text similarity using two approaches:

- **Pre-CLIP** : Separate ResNet50 (images) + sentence-transformers (text)
- **CLIP** : Unified multimodal model

Dataset: 20 F1 images across 4 categories with manually written captions.

---

## 🛠️ Technologies

- **PyTorch 2.1.0** (CUDA 11.8)
- **Transformers** (HuggingFace CLIP)
- **sentence-transformers** (distiluse-base-multilingual)
- **torchvision** (ResNet50)
- **Docker** (containerized environment)
- **Python 3.10**

---

## 📁 Project Structure

```
f1-clip-analysis/
├── .gitignore
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── dataset/                      # ⚠️ NOT INCLUDED (see below)
│   ├── 1_drivers_emotions/
│   ├── 2_pit_stops/
│   ├── 3_cars_tracks_moments/
│   ├── 4_strategy_data/
│   └── captions.json
├── notebooks/
│   └── analysis.ipynb           # Analysis notebook
├── results/
│   ├── clip/
│   │   ├── image_embeddings.npy
│   │   ├── image_metadata.csv
│   │   ├── similarity_matrix.csv
│   │   ├── text_embeddings.npy
│   │   └── text_metadata.csv
│   └── preclip/
│       ├── image_embeddings.npy
│       ├── image_metadata.csv
│       ├── similarity_matrix.csv
│       ├── text_embeddings.npy
│       └── text_metadata.csv
└── src/
    ├── utils.py                 # Common utilities (dataset, similarity, saving)
    ├── clip/
    │   ├── clip_models.py       # CLIP model
    │   └── run_clip.py          # Task 3: CLIP analysis
    └── preclip/
        ├── preclip_models.py    # ResNet + sentence-transformer
        ├── preclip_utils.py     # Pre-CLIP specific (PCA projection)
        └── run_preclip.py       # Task 2: Pre-CLIP analysis
```

---

## ⚠️ Dataset Warning

**The dataset is NOT included in this repository** due to size/licensing constraints.
If needed, contact me through my links of my bio.

The dataset structure required:

```
dataset/
├── 1_drivers_emotions/driver_01.jpg → driver_05.jpg
├── 2_pit_stops/pitstop_01.jpg → pitstop_05.jpg
├── 3_cars_tracks_moments/car_01.jpg → car_05.jpg
├── 4_strategy_data/strategy_01.jpg → strategy_05.jpg
└── captions.json
```

---

## 🚀 Quick Start

**Prerequisites:**

- Docker with GPU support ([nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU with CUDA support

**Build & Run:**

```bash
# Build Docker image
make build

# Run Pre-CLIP analysis (Task 2)
make run-preclip

# Run CLIP analysis (Task 3)
make run-clip

# Compare results
make compare

# Interactive shell
make shell
```

---

## 📊 Results

Results are saved in `results/` with:

- Similarity matrices (CSV)
- Visualizations (heatmaps, plots)
- Embedding files (.npz)

## 📝 License

Apache License 2.0

## Author

Víctor Vega Sobral - UIE 2024/2025
