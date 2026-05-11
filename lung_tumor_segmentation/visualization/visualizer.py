"""Visualizer — plotting utilities for CT/mask/prediction grids and overlays."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Reusable plotting helpers for the lung tumour segmentation project.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory where figures are saved. Defaults to current directory.
    """

    def __init__(self, output_dir: str | Path = ".") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sample grid
    # ------------------------------------------------------------------

    def plot_sample_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n: int = 8,
        save_name: str = "sample_batch.png",
    ) -> None:
        """Plot a grid of CT slices with their corresponding masks.

        Parameters
        ----------
        X : np.ndarray  shape (N, H, W, 1)  — CT slices.
        y : np.ndarray  shape (N, H, W, 1)  — binary masks.
        n : int         Number of samples to display (≤ len(X)).
        save_name : str Filename for the saved figure.
        """
        n = min(n, len(X))
        fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5))
        fig.suptitle(f"Sample CT Slices (top) and Ground-Truth Masks (bottom)", fontsize=12)

        for i in range(n):
            axes[0, i].imshow(X[i, :, :, 0], cmap="gray")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"CT {i}", fontsize=8)

            axes[1, i].imshow(y[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Mask {i}", fontsize=8)

        plt.tight_layout()
        out = self.output_dir / save_name
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Sample batch saved → %s", out)

    # ------------------------------------------------------------------
    # Prediction overlay
    # ------------------------------------------------------------------

    def plot_prediction_overlay(
        self,
        ct: np.ndarray,
        mask_true: np.ndarray,
        mask_pred: np.ndarray,
        title: str = "Prediction Overlay",
        save_name: str = "prediction_overlay.png",
    ) -> None:
        """Side-by-side: CT | Ground Truth | Prediction | Coloured Overlay.

        Overlay legend
        --------------
        Green = True Positive, Red = False Positive, Blue = False Negative.

        Parameters
        ----------
        ct        : np.ndarray  shape (H, W)   — CT slice (grayscale).
        mask_true : np.ndarray  shape (H, W)   — binary ground-truth.
        mask_pred : np.ndarray  shape (H, W)   — binary prediction.
        """
        # Normalise CT to [0, 1] for display
        ct_norm = (ct - ct.min()) / (ct.max() - ct.min() + 1e-6)
        overlay = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

        tp = (mask_true == 1) & (mask_pred == 1)
        fp = (mask_true == 0) & (mask_pred == 1)
        fn = (mask_true == 1) & (mask_pred == 0)

        overlay[tp] = [0.0, 1.0, 0.0]
        overlay[fp] = [1.0, 0.0, 0.0]
        overlay[fn] = [0.0, 0.0, 1.0]

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle(title, fontsize=11)

        for ax, img, lbl, cmap in zip(
            axes,
            [ct_norm, mask_true, mask_pred, overlay],
            ["CT Slice", "Ground Truth", "Prediction", "TP/FP/FN Overlay"],
            ["gray", "gray", "gray", None],
        ):
            ax.imshow(img, cmap=cmap)
            ax.set_title(lbl, fontsize=9)
            ax.axis("off")

        plt.tight_layout()
        out = self.output_dir / save_name
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Overlay figure saved → %s", out)

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        history_csv: str | Path,
        save_name: str = "training_history.png",
    ) -> None:
        """Plot loss and dice coefficient from a CSVLogger output file.

        Parameters
        ----------
        history_csv : str or Path
            Path to the CSV written by Keras CSVLogger callback.
        """
        history_csv = Path(history_csv)
        if not history_csv.exists():
            logger.warning("History CSV not found: %s", history_csv)
            return

        df = pd.read_csv(history_csv)
        epochs = df["epoch"] + 1  # 0-indexed → 1-indexed

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training History", fontsize=13)

        # Loss
        axes[0].plot(epochs, df["loss"], label="Train Loss", linewidth=2)
        if "val_loss" in df.columns:
            axes[0].plot(epochs, df["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title("Loss (BCE + Dice)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Dice
        dice_col = "dice_coefficient"
        if dice_col in df.columns:
            axes[1].plot(epochs, df[dice_col], label="Train Dice", linewidth=2)
            if f"val_{dice_col}" in df.columns:
                axes[1].plot(epochs, df[f"val_{dice_col}"], label="Val Dice", linewidth=2)
            axes[1].set_title("Dice Coefficient")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Dice")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / save_name
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training history plot saved → %s", out)
