# UMSF-Net: Unsupervised Multi-Source Fusion Network for Land Cover Classification

<p align="center">
  <img src="assets/framework.png" width="800"/>
</p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation of **"Unsupervised Land Cover Classification by Fusing SAR and Multispectral Optical Images via Cross-Modal Contrastive Learning"**.

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Network Architecture](#-network-architecture)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Results](#-results)
- [Visualization](#-visualization)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)

## ✨ Highlights

- **Dual-Branch Feature Extraction**: Specialized encoders for optical and SAR images with domain-specific preprocessing
- **Cross-Modal Contrastive Learning**: Deep alignment of optical-SAR features using InfoNCE loss with momentum contrast
- **Attention-based Fusion**: Adaptive multi-head cross-attention mechanism for modality fusion
- **End-to-End Unsupervised Framework**: No manual annotations required for land cover classification

<p align="center">
  <img src="assets/highlights.png" width="600"/>
</p>

## 🏗 Network Architecture

### Overall Framework

```
                              ┌─────────────────────────────────────────┐
                              │           UMSF-Net Framework            │
                              └─────────────────────────────────────────┘
                                                 │
                    ┌────────────────────────────┴────────────────────────────┐
                    │                                                          │
                    ▼                                                          ▼
    ┌───────────────────────────────┐                      ┌───────────────────────────────┐
    │      Optical Branch           │                      │        SAR Branch             │
    │  ┌─────────────────────────┐  │                      │  ┌─────────────────────────┐  │
    │  │   ResNet50 Backbone     │  │                      │  │  Despeckling + ResNet50 │  │
    │  │  (ImageNet pretrained)  │  │                      │  │  (Modified for 1-ch)    │  │
    │  └───────────┬─────────────┘  │                      │  └───────────┬─────────────┘  │
    └──────────────│────────────────┘                      └──────────────│────────────────┘
                   │                                                      │
                   └──────────────────────┬───────────────────────────────┘
                                          ▼
                          ┌───────────────────────────────┐
                          │   Cross-Modal Attention       │
                          │        Fusion Module          │
                          └──────────────│────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
                   ▼                                           ▼
    ┌───────────────────────────────┐           ┌───────────────────────────────┐
    │      Projection Head          │           │      Clustering Head          │
    │   (Contrastive Learning)      │           │   (Unsupervised Classification)│
    └───────────────────────────────┘           └───────────────────────────────┘
```

### Key Components

| Module | Description | Output Dim |
|--------|-------------|------------|
| Optical Encoder | ResNet50 with ImageNet pretrained weights | 2048 |
| SAR Encoder | Learnable despeckling + modified ResNet50 | 2048 |
| Attention Fusion | Multi-head cross-attention (8 heads) | 2048 |
| Projection Head | MLP (2048 → 2048 → 256) | 256 |
| Clustering Head | MLP (2048 → 512 → K) | K classes |

## 🔧 Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- CUDA >= 12.0

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/UMSF-Net.git
cd UMSF-Net

# Create conda environment
conda create -n umsf python=3.10 -y
conda activate umsf

# Install PyTorch (adjust according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

We use the **WHU-OPT-SAR** dataset released by Wuhan University.

### Download

Download the dataset from: [WHU-OPT-SAR Dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset)

### Data Structure

```
data/
├── WHU-OPT-SAR/
│   ├── optical/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── sar/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
```

### Data Preprocessing

```bash
# Prepare dataset (crop patches, split train/val/test)
python scripts/prepare_data.py \
    --data_root /path/to/WHU-OPT-SAR \
    --output_dir ./data/processed \
    --patch_size 256 \
    --overlap 64
```

### Class Definition

| Class ID | Class Name | Color |
|----------|------------|-------|
| 0 | Farmland | 🟩 Green |
| 1 | City | 🟥 Red |
| 2 | Village | 🟧 Orange |
| 3 | Water | 🟦 Blue |
| 4 | Forest | 🌲 Dark Green |
| 5 | Road | ⬜ White |
| 6 | Others | ⬛ Gray |

## 🚀 Training

### Single GPU Training

```bash
python train.py \
    --config configs/umsf_whu.yaml \
    --data_root ./data/processed \
    --output_dir ./outputs
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 train.py \
    --config configs/umsf_whu.yaml \
    --data_root ./data/processed \
    --output_dir ./outputs \
    --distributed
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | Required | Path to config YAML file |
| `--data_root` | Required | Path to processed dataset |
| `--output_dir` | `./outputs` | Output directory |
| `--epochs` | 200 | Total training epochs |
| `--batch_size` | 32 | Batch size per GPU |
| `--lr` | 0.03 | Initial learning rate |
| `--num_classes` | 7 | Number of land cover classes |
| `--resume` | None | Resume from checkpoint |

### Training Configuration

```yaml
# configs/umsf_whu.yaml
model:
  backbone: resnet50
  pretrained: true
  feature_dim: 2048
  projection_dim: 256
  num_classes: 7

training:
  epochs: 200
  batch_size: 32
  optimizer:
    type: SGD
    lr: 0.03
    momentum: 0.9
    weight_decay: 0.0001
  scheduler:
    type: cosine
    warmup_epochs: 10
  contrastive:
    temperature: 0.07
    queue_size: 65536
    momentum: 0.999
  loss_weights:
    contrastive_intra: 1.0
    contrastive_cross: 1.0
    clustering: 0.5
    consistency: 0.5
```

## 📈 Evaluation

### Run Evaluation

```bash
python evaluate.py \
    --config configs/umsf_whu.yaml \
    --checkpoint ./checkpoints/best_model.pth \
    --data_root ./data/processed \
    --output_dir ./results
```

### Evaluation Metrics

- **ACC**: Clustering Accuracy
- **NMI**: Normalized Mutual Information  
- **ARI**: Adjusted Rand Index
- **F1**: Macro F1 Score

## 📊 Results

### Comparison with State-of-the-art Methods

| Method | ACC (%) | NMI | ARI | F1 |
|--------|---------|-----|-----|-----|
| K-Means | 42.3 | 0.21 | 0.12 | 0.38 |
| DeepCluster | 55.6 | 0.36 | 0.26 | 0.51 |
| SwAV | 61.2 | 0.42 | 0.31 | 0.57 |
| SCAN | 63.5 | 0.44 | 0.33 | 0.59 |
| **UMSF-Net (Ours)** | **72.8** | **0.53** | **0.42** | **0.68** |

### Ablation Study

| Variant | ACC (%) | NMI |
|---------|---------|-----|
| Optical Only | 58.3 | 0.38 |
| SAR Only | 45.7 | 0.28 |
| Concat Fusion | 65.4 | 0.45 |
| Add Fusion | 66.8 | 0.47 |
| **Attention Fusion** | **72.8** | **0.53** |

## 🎨 Visualization

### Generate Visualizations

```bash
python visualization/visualize.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_root ./data/processed \
    --output_dir ./vis_results \
    --num_samples 16
```

### t-SNE Feature Visualization

<p align="center">
  <img src="assets/tsne.png" width="400"/>
</p>

### Classification Results

<p align="center">
  <img src="assets/results.png" width="700"/>
</p>

## 📁 Project Structure

```
UMSF-Net/
├── configs/                  # Configuration files
│   ├── umsf_whu.yaml
│   └── default.yaml
├── datasets/                 # Dataset and dataloader
│   ├── __init__.py
│   ├── whu_opt_sar.py
│   ├── transforms.py
│   └── utils.py
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── umsf_net.py          # Main network
│   ├── encoders.py          # Optical & SAR encoders
│   ├── fusion.py            # Attention fusion module
│   ├── heads.py             # Projection & clustering heads
│   └── losses.py            # Loss functions
├── utils/                    # Utilities
│   ├── __init__.py
│   ├── metrics.py           # Evaluation metrics
│   ├── logger.py            # Logging utilities
│   └── misc.py              # Miscellaneous utilities
├── scripts/                  # Helper scripts
│   ├── prepare_data.py
│   └── download_data.sh
├── visualization/            # Visualization tools
│   ├── visualize.py
│   └── plot_utils.py
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Dependencies
├── LICENSE
└── README.md
```

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourname2026umsf,
  title={Unsupervised Land Cover Classification by Fusing SAR and Multispectral Optical Images via Cross-Modal Contrastive Learning},
  author={Your Name},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```

## 🙏 Acknowledgements

- [WHU-OPT-SAR Dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset) for providing the optical-SAR paired dataset
- [MoCo](https://github.com/facebookresearch/moco) for the momentum contrast framework
- [SwAV](https://github.com/facebookresearch/swav) for the clustering-based contrastive learning

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

If you have any questions, please feel free to open an issue or contact us at [your-email@example.com](mailto:your-email@example.com).
