"""run_evaluation.py — CLI entry point for model evaluation.

Usage
-----
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --config path/to/config.yaml
    python scripts/run_evaluation.py --model path/to/model.keras

What it does
------------
1. Loads config and the best trained model.
2. Loads the pre-saved test split.
3. Runs Evaluator.evaluate_dataset().
4. Calls generate_report(), save_metric_distributions(),
   save_best_worst_predictions(), plot_roc_curve(),
   plot_threshold_sensitivity().
5. Prints final summary to console.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import tensorflow as tf

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lung_tumor_segmentation.config.loader import load_config
from lung_tumor_segmentation.evaluation.evaluator import Evaluator
from lung_tumor_segmentation.models.losses import (
    bce_dice_loss,
    dice_coefficient,
    iou_metric,
    precision_metric,
    recall_metric,
)
from lung_tumor_segmentation.preprocessing.dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_evaluation")


def main(
    config_path: str | None = None,
    model_path: str | None = None,
    splits_dir: str | None = None,
) -> None:
    cfg = load_config(config_path)

    # Resolve paths
    splits_dir = (
        Path(splits_dir)
        if splits_dir
        else _REPO_ROOT / "lung_tumor_segmentation" / "outputs" / "splits"
    )
    model_path = (
        Path(model_path)
        if model_path
        else Path(cfg["paths"]["output_models"]) / "best_model.keras"
    )

    logger.info("=== Lung Tumor Segmentation — Evaluation Pipeline ===")
    logger.info("Model path  : %s", model_path)
    logger.info("Splits dir  : %s", splits_dir)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    builder = DatasetBuilder(cfg)
    try:
        _, _, _, _, X_test, y_test = builder.load_splits(splits_dir)
    except FileNotFoundError as exc:
        logger.error("Splits not found. Run run_preprocessing.py first.\n  %s", exc)
        sys.exit(1)
    logger.info("Test slices: %d", len(X_test))

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not model_path.exists():
        logger.error(
            "Model not found at %s. Run run_training.py first.", model_path
        )
        sys.exit(1)

    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "dice_coefficient": dice_coefficient,
        "iou_metric": iou_metric,
        "precision_metric": precision_metric,
        "recall_metric": recall_metric,
    }
    model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    evaluator = Evaluator(model, cfg)
    results_df = evaluator.evaluate_dataset(X_test, y_test)

    evaluator.generate_report(results_df)
    evaluator.save_metric_distributions(results_df)
    evaluator.save_best_worst_predictions(X_test, y_test, results_df, n=cfg["evaluation"]["save_best_n"])
    evaluator.plot_roc_curve(X_test, y_test)
    evaluator.plot_threshold_sensitivity(X_test, y_test)

    logger.info("=== Evaluation Complete ===")
    logger.info("All outputs saved to:")
    logger.info("  Metrics  → %s", cfg["paths"]["output_metrics"])
    logger.info("  Figures  → %s", cfg["paths"]["output_figures"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, dest="model_path")
    parser.add_argument("--splits-dir", type=str, default=None)
    args = parser.parse_args()
    main(
        config_path=args.config,
        model_path=args.model_path,
        splits_dir=args.splits_dir,
    )
