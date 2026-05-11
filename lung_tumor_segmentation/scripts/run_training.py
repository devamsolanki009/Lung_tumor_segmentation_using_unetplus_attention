"""run_training.py — CLI entry point for model training.

Usage
-----
    python scripts/run_training.py
    python scripts/run_training.py --config path/to/config.yaml
    python scripts/run_training.py --splits-dir path/to/splits/

What it does
------------
1. Loads config.
2. Loads pre-saved train / val / test splits from outputs/splits/.
3. Builds and compiles the Attention U-Net via get_model(config).
4. Instantiates Trainer and runs train().
5. Saves best model to outputs/models/best_model.keras.
6. Saves training history plot to outputs/figures/training_curves.png.
"""

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_training")


def main(config_path: str | None = None, splits_dir: str | None = None) -> None:
    cfg = load_config(config_path)

    # Resolve splits directory
    if splits_dir is None:
        splits_dir = _REPO_ROOT / "lung_tumor_segmentation" / "outputs" / "splits"
    else:
        splits_dir = Path(splits_dir)

    logger.info("=== Lung Tumor Segmentation — Training Pipeline ===")
    logger.info("Splits directory: %s", splits_dir)

    # ------------------------------------------------------------------
    # Load data splits
    # ------------------------------------------------------------------
    builder = DatasetBuilder(cfg)
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = builder.load_splits(splits_dir)
    except FileNotFoundError as exc:
        logger.error(
            "Splits not found. Run run_preprocessing.py first.\n  %s", exc
        )
        sys.exit(1)

    logger.info(
        "Data loaded — train: %d, val: %d, test: %d slices",
        len(X_train), len(X_val), len(X_test),
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    logger.info("Building Attention U-Net model …")
    model = get_model(cfg)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = Trainer(model, cfg, (X_train, y_train, X_val, y_val))
    history = trainer.train()
    trainer.save_training_curve(history)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    best_epoch = int(history.history["val_dice_coefficient"].index(
        max(history.history["val_dice_coefficient"])
    )) + 1
    best_dice = max(history.history["val_dice_coefficient"])
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
