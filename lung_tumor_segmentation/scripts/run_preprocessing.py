"""run_preprocessing.py — CLI entry point for the preprocessing pipeline.

Usage
-----
    python scripts/run_preprocessing.py
    python scripts/run_preprocessing.py --config path/to/config.yaml

What it does
------------
1. Reads the project config.
2. Iterates all patient directories in the raw dataset.
3. For each patient, runs DICOMLoader + Preprocessor → saves .npy slice files.
4. Calls DatasetBuilder to build and save train / val / test splits.
5. Prints a final summary and writes a log to outputs/logs/preprocessing.log.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on the path when run directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lung_tumor_segmentation.config.loader import load_config
from lung_tumor_segmentation.preprocessing.dicom_loader import DICOMLoader
from lung_tumor_segmentation.preprocessing.preprocessor import Preprocessor
from lung_tumor_segmentation.preprocessing.dataset_builder import DatasetBuilder


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "preprocessing.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    _setup_logging(cfg["paths"]["output_logs"])
    logger = logging.getLogger("run_preprocessing")

    paths = cfg["paths"]
    raw_dir = Path(paths["raw_dataset"])
    # For NSCLC-Radiomics the DICOM data lives inside the manifest folder
    manifest_dir = _REPO_ROOT / "manifest-1603198545583" / "NSCLC-Radiomics"
    dataset_dir = manifest_dir if manifest_dir.exists() else raw_dir

    ct_out_dir = Path(paths["preprocessed_cts"])
    mask_out_dir = Path(paths["preprocessed_masks"])

    logger.info("=== Lung Tumor Segmentation — Preprocessing Pipeline ===")
    logger.info("Dataset directory  : %s", dataset_dir)
    logger.info("CT output dir      : %s", ct_out_dir)
    logger.info("Mask output dir    : %s", mask_out_dir)

    loader = DICOMLoader()
    preprocessor = Preprocessor(cfg)

    patient_dirs = loader.get_patient_list(dataset_dir)
    total = len(patient_dirs)
    processed = 0
    skipped = 0
    total_slices = 0
    skipped_patients: list[str] = []

    t0 = time.time()

    for idx, patient_dir in enumerate(patient_dirs, start=1):
        pid = patient_dir.name
        logger.info("[%d/%d] Processing %s …", idx, total, pid)

        result = preprocessor.process_patient(patient_dir)
        if result is None:
            skipped += 1
            skipped_patients.append(pid)
            continue

        cts, masks = result
        preprocessor.save_patient_arrays(pid, cts, masks, ct_out_dir, mask_out_dir)
        n_slices = cts.shape[0]
        total_slices += n_slices
        processed += 1
        logger.info("  → saved %d slices for %s", n_slices, pid)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=== Preprocessing Summary ===")
    logger.info("Total patients   : %d", total)
    logger.info("Processed        : %d", processed)
    logger.info("Skipped          : %d", skipped)
    logger.info("Total slices     : %d", total_slices)
    logger.info("Elapsed time     : %.1f s", elapsed)
    if skipped_patients:
        logger.info("Skipped patients : %s", ", ".join(skipped_patients))

    # ------------------------------------------------------------------
    # Build and save train / val / test splits
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=== Building dataset splits ===")
    builder = DatasetBuilder(cfg)
    splits = builder.build_dataset(ct_out_dir, mask_out_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = splits

    splits_dir = _REPO_ROOT / "lung_tumor_segmentation" / "outputs" / "splits"
    builder.save_splits(splits, splits_dir)

    logger.info("")
    logger.info("=== Split Summary ===")
    logger.info("Train slices : %d", len(X_train))
    logger.info("Val slices   : %d", len(X_val))
    logger.info("Test slices  : %d", len(X_test))
    logger.info("Splits saved to: %s", splits_dir)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml).",
    )
    args = parser.parse_args()
    main(config_path=args.config)
