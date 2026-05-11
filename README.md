# 🫁 Lung Tumor Segmentation Using Attention U-Net

A modular, production-ready deep learning pipeline for automated lung tumor segmentation from CT scans using an **Attention U-Net** architecture trained on the **NSCLC-Radiomics** dataset.

---

## Architecture Overview

```
Input CT Slice (128×128×1)
        │
  ┌─────▼─────┐
  │  Encoder   │  (4 levels: 64 → 128 → 256 → 512 filters)
  │  Conv Blocks│  Each: Conv→BN→ReLU → Conv→BN→ReLU → MaxPool
  └─────┬─────┘
        │ skip connections (with attention gates)
  ┌─────▼─────┐
  │ Bottleneck │  (1024 filters + Dropout)
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │  Decoder   │  (4 levels: 512 → 256 → 128 → 64 filters)
  │  Attention │  ConvTranspose → AttGate(skip) → Concat → Conv
  │  Gates     │
  └─────┬─────┘
        │
  ┌─────▼─────┐
  │  Output    │  Conv2D(1, sigmoid) → Binary Mask (128×128×1)
  └───────────┘
```

**Loss:** Combined BCE + Dice Loss (50/50 weighting)  
**Optimizer:** Adam (lr=0.0001, with ReduceLROnPlateau)  
**Metrics:** Dice, IoU, Precision, Recall

---

## Dataset Setup (NSCLC-Radiomics)

1. Register at [TCIA](https://www.cancerimagingarchive.net/)
2. Download the **NSCLC-Radiomics** collection using the NBIA Data Retriever
3. Place the manifest directory at:
   ```
   c:\Lung_tumor_segmentation\manifest-1603198545583\NSCLC-Radiomics\
   ```
   The folder should contain `LUNG1-001/`, `LUNG1-002/`, … up to `LUNG1-422/`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/devamsolanki009/Lung_tumor_segmentation_using_unetplus_attention.git
cd Lung_tumor_segmentation_using_unetplus_attention
git checkout dev

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline (preprocess → train → evaluate):
```bash
python main.py --all
```

### Run individual stages:
```bash
python main.py --preprocess    # DICOM → .npy slices + train/val/test splits
python main.py --train         # Train Attention U-Net, saves best_model.keras
python main.py --evaluate      # Full evaluation report, plots, ROC curve
```

### With a custom config:
```bash
python main.py --all --config path/to/my_config.yaml
```

---

## Project Structure

```
Lung_tumor_segmentation/
├── main.py                          ← Master entry point
├── requirements.txt
├── README.md
├── lung_tumor_segmentation/
│   ├── config/
│   │   ├── config.yaml              ← All hyperparameters & paths
│   │   └── loader.py                ← YAML → Python dict
│   ├── preprocessing/
│   │   ├── dicom_loader.py          ← DICOM reading & series detection
│   │   ├── preprocessor.py          ← Resize, normalize, filter empty masks
│   │   └── dataset_builder.py       ← Train/val/test splits, Keras generators
│   ├── models/
│   │   ├── attention_unet.py        ← Attention U-Net++ (Functional API)
│   │   └── losses.py                ← Dice, BCE+Dice, IoU, Precision, Recall
│   ├── training/
│   │   └── trainer.py               ← Training loop + callbacks
│   ├── evaluation/
│   │   ├── metrics.py               ← Pure NumPy metric functions
│   │   └── evaluator.py             ← Full evaluation & reporting pipeline
│   ├── visualization/
│   │   └── visualizer.py            ← Grid plots, overlays, history curves
│   ├── scripts/
│   │   ├── run_preprocessing.py     ← CLI: preprocessing
│   │   ├── run_training.py          ← CLI: training
│   │   └── run_evaluation.py        ← CLI: evaluation
│   └── outputs/
│       ├── models/                  ← best_model.keras
│       ├── logs/                    ← TensorBoard + CSV training logs
│       ├── metrics/                 ← evaluation_report.json, summary.csv
│       └── figures/                 ← all saved plots
└── notebooks/
    └── legacy_notebook.ipynb        ← Original monolithic notebook (preserved)
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Dice Coefficient** | Primary overlap metric; 2×\|X∩Y\| / (\|X\|+\|Y\|) |
| **IoU (Jaccard)** | \|X∩Y\| / \|X∪Y\| |
| **Precision** | TP / (TP + FP) — how many predicted pixels are correct |
| **Recall** | TP / (TP + FN) — how many true tumor pixels are found |
| **F1 Score** | Harmonic mean of precision and recall |
| **Specificity** | TN / (TN + FP) — true negative rate |
| **Volumetric Similarity** | 1 - \|V_pred - V_true\| / (V_pred + V_true) |
| **Hausdorff Distance** | 95th-percentile boundary distance (lower = better) |

---

## Sample Results

> After training, evaluation outputs are saved to `lung_tumor_segmentation/outputs/`:

- `figures/training_curves.png` — Loss & Dice vs epoch
- `figures/roc_curve.png` — ROC curve with AUC
- `figures/threshold_analysis.png` — Dice & IoU vs threshold
- `figures/dice_distribution.png` — Violin + box plot of Dice scores
- `figures/best_prediction_01.png` — Best predicted segmentation
- `figures/worst_prediction_01.png` — Worst predicted segmentation
- `metrics/evaluation_summary.csv` — Aggregated stats table
- `metrics/evaluation_report.json` — Full per-slice metrics

---

## Configuration

Edit `lung_tumor_segmentation/config/config.yaml` to tune:

- `preprocessing.target_size` — Input resolution (default: 128×128)
- `model.filters` — Encoder depth (default: [64, 128, 256, 512])
- `training.epochs` — Max epochs (default: 100)
- `training.learning_rate` — Initial LR (default: 0.0001)
- `augmentation.enabled` — Toggle data augmentation

---

## Known Issues & Future Work

### Known Issues
- LUNG1-128 is missing from the NSCLC-Radiomics dataset
- Some patients have very few tumor slices (e.g., LUNG1-050 = 4 slices)
- Large dataset (~422 patients) requires significant disk space and RAM

### Future Work
- [ ] 3D U-Net for volumetric context across slices
- [ ] Higher resolution (256×256 or 512×512)
- [ ] Elastic deformation augmentation
- [ ] Transfer learning with pretrained ImageNet encoders
- [ ] CRF post-processing for boundary refinement
- [ ] MLflow / W&B experiment tracking integration
- [ ] Unit tests for preprocessing and metrics modules
- [ ] Docker containerisation for reproducibility

---

## Citation

If you use this code, please cite the original Attention U-Net paper:

```bibtex
@article{oktay2018attention,
  title={Attention u-net: Learning where to look for the pancreas},
  author={Oktay, Ozan et al.},
  journal={MIDL 2018},
  year={2018}
}
```

---

## License

MIT License — see `LICENSE` for details.
