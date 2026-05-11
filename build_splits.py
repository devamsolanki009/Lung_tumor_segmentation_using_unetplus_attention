"""build_splits.py — Fast index-based split builder.

PURPOSE
-------
Instead of loading all 28K slices into RAM (which takes minutes and uses ~2 GB),
this script:
  1. Scans both preprocessed directories for .npy files
  2. Matches CT ↔ mask pairs by filename
  3. Groups pairs by patient ID
  4. Splits patients into train / val / test (70 / 15 / 15)
  5. Saves THREE CSV index files — one per split
     (outputs/splits/train_index.csv, val_index.csv, test_index.csv)

Each CSV has two columns:  ct_path, mask_path
Training then streams files on-demand via tf.data — zero RAM overhead.

Usage
-----
    python build_splits.py
"""

import sys
import csv
import random
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lung_tumor_segmentation.config.loader import load_config
# ---------------------------------------------------------------------------

cfg  = load_config()
prep = cfg["preprocessing"]

CT_DIR   = Path(r"c:\Lung_tumor_segmentation\preprocessed_cts_128x128")
MASK_DIR = Path(r"c:\Lung_tumor_segmentation\preprocessed_masks_128x128")
OUT_DIR  = Path(r"c:\Lung_tumor_segmentation\lung_tumor_segmentation\outputs\splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = prep["train_ratio"]   # 0.70
VAL_RATIO   = prep["val_ratio"]     # 0.15
SEED        = prep["random_seed"]   # 42

# ---------------------------------------------------------------------------
# Step 1 — Build a map:  patient_id → [(ct_path, mask_path), ...]
# ---------------------------------------------------------------------------
print("Scanning CT directory …", end=" ", flush=True)
ct_files = {f.stem: f for f in CT_DIR.glob("*_ct_slice_*.npy")}
print(f"{len(ct_files)} files found.")

print("Scanning mask directory …", end=" ", flush=True)
mask_files = {f.stem: f for f in MASK_DIR.glob("*_mask_slice_*.npy")}
print(f"{len(mask_files)} files found.")

# Match by swapping "_ct_slice_" for "_mask_slice_"
patient_pairs: dict[str, list] = defaultdict(list)
unmatched = 0

for ct_stem, ct_path in sorted(ct_files.items()):
    mask_stem = ct_stem.replace("_ct_slice_", "_mask_slice_")
    if mask_stem in mask_files:
        pid = ct_stem.split("_ct_slice_")[0]   # e.g. "LUNG1-001"
        patient_pairs[pid].append((str(ct_path), str(mask_files[mask_stem])))
    else:
        unmatched += 1

print(f"\nPaired slices  : {sum(len(v) for v in patient_pairs.values())}")
print(f"Unmatched CTs  : {unmatched}")
print(f"Unique patients: {len(patient_pairs)}")

# ---------------------------------------------------------------------------
# Step 2 — Patient-level shuffle + split (avoids data leakage)
# ---------------------------------------------------------------------------
patient_ids = sorted(patient_pairs.keys())
random.seed(SEED)
random.shuffle(patient_ids)

n          = len(patient_ids)
n_train    = int(n * TRAIN_RATIO)
n_val      = int(n * VAL_RATIO)

train_pids = patient_ids[:n_train]
val_pids   = patient_ids[n_train : n_train + n_val]
test_pids  = patient_ids[n_train + n_val :]

def count_slices(pids):
    return sum(len(patient_pairs[p]) for p in pids)

print(f"\n=== Patient Split ===")
print(f"Train : {len(train_pids)} patients  |  {count_slices(train_pids)} slices")
print(f"Val   : {len(val_pids)} patients  |  {count_slices(val_pids)} slices")
print(f"Test  : {len(test_pids)} patients  |  {count_slices(test_pids)} slices")

# ---------------------------------------------------------------------------
# Step 3 — Write CSV index files
# ---------------------------------------------------------------------------
def write_csv(pids, filename):
    out_path = OUT_DIR / filename
    rows = []
    for pid in pids:
        rows.extend(patient_pairs[pid])
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ct_path", "mask_path"])
        writer.writerows(rows)
    print(f"  Saved {len(rows):>6} rows → {out_path.name}")
    return len(rows)

print(f"\n=== Saving index CSVs to {OUT_DIR} ===")
write_csv(train_pids, "train_index.csv")
write_csv(val_pids,   "val_index.csv")
write_csv(test_pids,  "test_index.csv")

print("\n✔  Done. No large .npy files were created — training streams files on-demand.")
print("   Next step: python main.py --train")
