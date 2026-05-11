"""Evaluator — runs the full post-training evaluation pipeline.

Produces:
- Per-slice metrics DataFrame
- Aggregated JSON report + CSV summary
- Violin / box plots of metric distributions
- Best / worst prediction visualisations with TP/FP/FN overlay
- ROC curve with AUC
- Threshold sensitivity analysis (Dice & IoU vs threshold)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

from lung_tumor_segmentation.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_METRIC_DISPLAY = {
    "dice": "Dice Coefficient",
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "specificity": "Specificity",
    "volumetric_similarity": "Volumetric Similarity",
    "hausdorff_distance": "Hausdorff Distance (95th pct)",
}


class Evaluator:
    """Full evaluation pipeline for the Attention U-Net segmentation model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    config : dict
        Full project config dict.
    """

    def __init__(self, model: tf.keras.Model, config: dict) -> None:
        self.model = model
        self.cfg = config
        ecfg = config["evaluation"]
        self.threshold: float = ecfg["threshold"]
        self.save_best_n: int = ecfg["save_best_n"]
        self.save_worst_n: int = ecfg["save_worst_n"]

        paths = config["paths"]
        self.metrics_dir = Path(paths["output_metrics"])
        self.figures_dir = Path(paths["output_figures"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Predict on all test slices and compute per-slice metrics.

        Parameters
        ----------
        X_test : np.ndarray  shape (N, H, W, 1)
        y_test : np.ndarray  shape (N, H, W, 1)

        Returns
        -------
        pd.DataFrame
            One row per slice; columns = metric names + "slice_idx".
        """
        logger.info("Running inference on %d test slices …", len(X_test))
        y_pred = self.model.predict(X_test, batch_size=16, verbose=1)  # (N, H, W, 1)

        records = []
        for i in range(len(X_test)):
            gt = y_test[i, :, :, 0]
            pred = y_pred[i, :, :, 0]
            m = compute_all_metrics(gt, pred, threshold=self.threshold)
            m["slice_idx"] = i
            records.append(m)

        df = pd.DataFrame(records)
        logger.info("Evaluation complete for %d slices.", len(df))
        return df

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, results_df: pd.DataFrame) -> None:
        """Save JSON full report + CSV summary, and print table to console."""
        metric_cols = [c for c in results_df.columns if c != "slice_idx"]

        # Aggregated stats
        agg = results_df[metric_cols].agg(["mean", "std", "min", "max", "median"])

        # Save full per-slice report as JSON
        json_path = self.metrics_dir / "evaluation_report.json"
        report = {
            "per_slice": results_df.to_dict(orient="records"),
            "aggregate": agg.to_dict(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Full report saved → %s", json_path)

        # Save aggregated CSV
        csv_path = self.metrics_dir / "evaluation_summary.csv"
        agg.to_csv(csv_path)
        logger.info("Summary CSV saved → %s", csv_path)

        # Pretty-print table
        self._print_table(agg)

    def _print_table(self, agg: pd.DataFrame) -> None:
        """Print a formatted metrics table to stdout."""
        width_name, width_val = 26, 8
        sep = "─" * (width_name + 2 * (width_val + 3) + 5)
        print(f"\n┌{'─' * (width_name + 2)}┬{'─' * (width_val + 2)}┬{'─' * (width_val + 2)}┐")
        print(f"│  {'Metric':<{width_name}}│  {'Mean':<{width_val}}│  {'Std':<{width_val}}│")
        print(f"├{'─' * (width_name + 2)}┼{'─' * (width_val + 2)}┼{'─' * (width_val + 2)}┤")
        for key, display in _METRIC_DISPLAY.items():
            if key not in agg.columns:
                continue
            mean_val = agg.at["mean", key]
            std_val = agg.at["std", key]
            if key == "hausdorff_distance":
                mean_str = f"{mean_val:>{width_val}.2f}"
                std_str = f"{std_val:>{width_val}.2f}"
            else:
                mean_str = f"{mean_val:>{width_val}.4f}"
                std_str = f"{std_val:>{width_val}.4f}"
            print(f"│  {display:<{width_name}}│  {mean_str}│  {std_str}│")
        print(f"└{'─' * (width_name + 2)}┴{'─' * (width_val + 2)}┴{'─' * (width_val + 2)}┘\n")

    # ------------------------------------------------------------------
    # Distribution plots
    # ------------------------------------------------------------------

    def save_metric_distributions(self, results_df: pd.DataFrame) -> None:
        """Save violin + box plots for Dice and IoU distributions."""
        for metric in ("dice", "iou"):
            data = results_df[metric].dropna().values

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"{_METRIC_DISPLAY[metric]} Distribution (n={len(data)} slices)",
                fontsize=13,
            )

            # Violin
            axes[0].violinplot(data, showmedians=True)
            axes[0].set_title("Violin Plot")
            axes[0].set_ylabel(_METRIC_DISPLAY[metric])
            axes[0].set_xticks([1])
            axes[0].set_xticklabels([metric.upper()])

            # Box
            axes[1].boxplot(data, patch_artist=True)
            axes[1].set_title("Box Plot")
            axes[1].set_ylabel(_METRIC_DISPLAY[metric])
            axes[1].set_xticks([1])
            axes[1].set_xticklabels([metric.upper()])

            plt.tight_layout()
            out = self.figures_dir / f"{metric}_distribution.png"
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Distribution plot saved → %s", out)

    # ------------------------------------------------------------------
    # Best / worst predictions
    # ------------------------------------------------------------------

    def save_best_worst_predictions(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        results_df: pd.DataFrame,
        n: int = 5,
    ) -> None:
        """Save side-by-side visualisations of the N best and N worst slices."""
        y_pred = self.model.predict(X_test, batch_size=16, verbose=0)

        sorted_df = results_df.sort_values("dice")
        worst_idx = sorted_df["slice_idx"].values[:n].astype(int)
        best_idx = sorted_df["slice_idx"].values[-n:][::-1].astype(int)

        for label, indices in [("best", best_idx), ("worst", worst_idx)]:
            for rank, idx in enumerate(indices, start=1):
                ct = X_test[idx, :, :, 0]
                gt = y_test[idx, :, :, 0]
                pred = y_pred[idx, :, :, 0]
                pred_bin = (pred >= self.threshold).astype(np.uint8)

                self._save_overlay_figure(
                    ct, gt, pred_bin,
                    title=f"{label.upper()} #{rank}  |  Dice={results_df.loc[results_df['slice_idx']==idx, 'dice'].values[0]:.4f}",
                    out_path=self.figures_dir / f"{label}_prediction_{rank:02d}.png",
                )
        logger.info("Best/worst prediction images saved to %s", self.figures_dir)

    def _save_overlay_figure(
        self,
        ct: np.ndarray,
        gt: np.ndarray,
        pred_bin: np.ndarray,
        title: str,
        out_path: Path,
    ) -> None:
        """CT | Ground Truth | Prediction | TP/FP/FN Overlay."""
        # Build RGB overlay
        overlay = np.stack([ct, ct, ct], axis=-1)
        overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-6)

        tp = (gt == 1) & (pred_bin == 1)
        fp = (gt == 0) & (pred_bin == 1)
        fn = (gt == 1) & (pred_bin == 0)

        overlay_rgb = overlay.copy()
        overlay_rgb[tp] = [0.0, 1.0, 0.0]   # green  → TP
        overlay_rgb[fp] = [1.0, 0.0, 0.0]   # red    → FP
        overlay_rgb[fn] = [0.0, 0.0, 1.0]   # blue   → FN

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle(title, fontsize=11)

        axes[0].imshow(ct, cmap="gray"); axes[0].set_title("CT Slice"); axes[0].axis("off")
        axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
        axes[2].imshow(pred_bin, cmap="gray"); axes[2].set_title("Prediction"); axes[2].axis("off")
        axes[3].imshow(overlay_rgb); axes[3].set_title("Overlay\n(TP=green, FP=red, FN=blue)"); axes[3].axis("off")

        plt.tight_layout()
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Plot ROC curve and annotate AUC."""
        logger.info("Computing ROC curve …")
        y_pred = self.model.predict(X_test, batch_size=16, verbose=0)

        y_true_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
        auc = roc_auc_score(y_true_flat, y_pred_flat)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve — Lung Tumor Segmentation", fontsize=13)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        out = self.figures_dir / "roc_curve.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curve saved → %s  (AUC=%.4f)", out, auc)

    # ------------------------------------------------------------------
    # Threshold sensitivity
    # ------------------------------------------------------------------

    def plot_threshold_sensitivity(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        steps: int = 50,
    ) -> None:
        """Plot Dice and IoU as a function of threshold."""
        logger.info("Computing threshold sensitivity …")
        y_pred = self.model.predict(X_test, batch_size=16, verbose=0)

        thresholds = np.linspace(0.0, 1.0, steps)
        dices, ious = [], []

        for thr in thresholds:
            pred_bin = (y_pred >= thr).astype(np.uint8)
            gt_bin = (y_test >= 0.5).astype(np.uint8)
            intersection = np.sum(gt_bin * pred_bin)
            denom = np.sum(gt_bin) + np.sum(pred_bin)
            dices.append(float(2 * intersection / denom) if denom > 0 else 1.0)
            union = denom - intersection
            ious.append(float(intersection / union) if union > 0 else 1.0)

        best_thr = thresholds[np.argmax(dices)]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, dices, label="Dice", linewidth=2)
        ax.plot(thresholds, ious, label="IoU", linewidth=2)
        ax.axvline(x=best_thr, color="red", linestyle="--",
                   label=f"Optimal threshold = {best_thr:.2f}")
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Dice & IoU vs Prediction Threshold", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        out = self.figures_dir / "threshold_analysis.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(
            "Threshold analysis saved → %s  (best threshold=%.2f)", out, best_thr
        )
