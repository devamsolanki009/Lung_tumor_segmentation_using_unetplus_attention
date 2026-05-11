"""One-time script to build and save train/val/test splits from existing .npy files.

Run this from the project root:
    python build_splits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lung_tumor_segmentation.config.loader import load_config
from lung_tumor_segmentation.preprocessing.dataset_builder import DatasetBuilder

cfg = load_config()

# Point to the existing preprocessed directories
ct_dir  = Path(r"c:\Lung_tumor_segmentation\preprocessed_cts_128x128")
mask_dir = Path(r"c:\Lung_tumor_segmentation\preprocessed_masks_128x128")

print(f"CT  directory : {ct_dir}  ({len(list(ct_dir.glob('*.npy')))} files)")
print(f"Mask directory: {mask_dir}  ({len(list(mask_dir.glob('*.npy')))} files)")

builder = DatasetBuilder(cfg)
splits  = builder.build_dataset(ct_dir, mask_dir)
X_train, y_train, X_val, y_val, X_test, y_test = splits

print("\n=== Split Summary ===")
print(f"Train : {len(X_train)} slices  {X_train.shape}")
print(f"Val   : {len(X_val)}   slices  {X_val.shape}")
print(f"Test  : {len(X_test)}  slices  {X_test.shape}")

splits_out = Path(r"c:\Lung_tumor_segmentation\lung_tumor_segmentation\outputs\splits")
builder.save_splits(splits, splits_out)
print(f"\nSplits saved → {splits_out}")
