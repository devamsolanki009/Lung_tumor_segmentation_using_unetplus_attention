"""run_training.py — CLI entry point for model training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lung_tumor_segmentation.config.loader import load_config
from lung_tumor_segmentation.models.attention_unet import get_model
from lung_tumor_segmentation.preprocessing.dataset_builder import DatasetBuilder
from lung_tumor_segmentation.training.trainer import Trainer
from lung_tumor_segmentation.training.gpu_utils import setup_gpu, log_gpu_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_training")


def main(config_path: str | None = None, splits_dir: str | None = None) -> None:
    cfg = load_config(config_path)

    # ------------------------------------------------------------------
    # GPU setup — MUST be first, before any TF graph is built
    # ------------------------------------------------------------------
    gpu_info = setup_gpu(cfg)

    # Resolve splits directory
    if splits_dir is None:
        splits_dir = _REPO_ROOT / "lung_tumor_segmentation" / "outputs" / "splits"
    else:
        splits_dir = Path(splits_dir)

    logger.info("=== Lung Tumor Segmentation — Training Pipeline ===")
    logger.info("Splits directory: %s", splits_dir)
    logger.info("GPU: %s | Mixed precision: %s", gpu_info["gpu_name"], gpu_info["mixed_precision"])

    # ------------------------------------------------------------------
    # Load CSV index files (fast — just file paths, no RAM loading)
    # ------------------------------------------------------------------
    builder = DatasetBuilder(cfg)
    try:
        train_ct, train_mask, val_ct, val_mask, test_ct, test_mask = builder.load_splits(splits_dir)
    except FileNotFoundError as exc:
        logger.error("Splits not found. Run  python build_splits.py  first.\n  %s", exc)
        sys.exit(1)

    logger.info(
        "Index loaded — train: %d, val: %d, test: %d slices",
        len(train_ct), len(val_ct), len(test_ct),
    )

    # ------------------------------------------------------------------
    # Build streaming tf.data datasets (no RAM allocation)
    # ------------------------------------------------------------------
    batch_size = cfg["training"]["batch_size"]
    train_ds = builder.get_tf_dataset(train_ct, train_mask, batch_size=batch_size, augment=True,  shuffle=True)
    val_ds   = builder.get_tf_dataset(val_ct,   val_mask,   batch_size=batch_size, augment=False, shuffle=False)

    steps_per_epoch = len(train_ct) // batch_size
    val_steps       = len(val_ct)   // batch_size

    logger.info(
        "Batch size: %d | Steps/epoch: %d | Val steps: %d",
        batch_size, steps_per_epoch, val_steps,
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    logger.info("Building Attention U-Net model …")
    model = get_model(cfg)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(model, cfg)
    history = trainer.train(train_ds, val_ds, steps_per_epoch, val_steps)
    trainer.save_training_curve(history)
    log_gpu_memory()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    dice_hist = history.history.get("val_dice_coefficient", [])
    if dice_hist:
        best_epoch = int(dice_hist.index(max(dice_hist))) + 1
        best_dice  = max(dice_hist)
        logger.info("=== Training Summary ===")
        logger.info("Best val Dice : %.4f  (epoch %d)", best_dice, best_epoch)
    logger.info(
        "Best model saved → %s",
        Path(cfg["paths"]["output_models"]) / "best_model.keras",
    )
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--splits-dir", type=str, default=None)
    args = parser.parse_args()
    main(config_path=args.config, splits_dir=args.splits_dir)
