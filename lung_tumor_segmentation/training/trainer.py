"""Trainer — orchestrates the training loop with callbacks and history saving.

Usage
-----
    from lung_tumor_segmentation.training.trainer import Trainer
    trainer = Trainer(model, config, data)
    history = trainer.train()
    trainer.save_training_curve(history)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks as cb

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the full training loop for the Attention U-Net.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled model returned by ``get_model(config)``.
    config : dict
        Full project config dict.
    data : tuple
        ``(X_train, y_train, X_val, y_val)`` — NumPy arrays, each of shape
        ``(N, H, W, 1)``.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: dict,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.model = model
        self.cfg = config
        self.X_train, self.y_train, self.X_val, self.y_val = data

        paths = config["paths"]
        self.model_dir = Path(paths["output_models"])
        self.log_dir = Path(paths["output_logs"])
        self.figures_dir = Path(paths["output_figures"])
        for d in (self.model_dir, self.log_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

        tcfg = config["training"]
        self.epochs: int = tcfg["epochs"]
        self.batch_size: int = tcfg["batch_size"]
        self.patience: int = tcfg["patience"]
        self.reduce_lr_factor: float = tcfg["reduce_lr_factor"]
        self.reduce_lr_patience: int = tcfg["reduce_lr_patience"]

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def get_callbacks(self) -> list:
        """Return the full list of Keras callbacks for training.

        Callbacks
        ---------
        - ModelCheckpoint — saves best weights (by val_dice_coefficient).
        - EarlyStopping   — stops if val_dice_coefficient stops improving.
        - ReduceLROnPlateau — halves LR when val loss stagnates.
        - TensorBoard     — writes logs to outputs/logs/tensorboard/.
        - CSVLogger       — appends per-epoch metrics to a CSV file.
        """
        best_model_path = self.model_dir / "best_model.keras"

        checkpoint = cb.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        )

        early_stop = cb.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = cb.ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        )

        tb_log_dir = self.log_dir / "tensorboard"
        tensorboard = cb.TensorBoard(
            log_dir=str(tb_log_dir),
            histogram_freq=0,
            write_graph=False,
        )

        csv_logger = cb.CSVLogger(
            filename=str(self.log_dir / "training_history.csv"),
            append=False,
        )

        return [checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> tf.keras.callbacks.History:
        """Run model.fit() and return the History object.

        Returns
        -------
        tf.keras.callbacks.History
        """
        logger.info(
            "Starting training — epochs=%d, batch_size=%d, "
            "train_samples=%d, val_samples=%d",
            self.epochs,
            self.batch_size,
            len(self.X_train),
            len(self.X_val),
        )

        history = self.model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=self.get_callbacks(),
            verbose=1,
        )

        logger.info("Training complete.")
        return history

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def save_training_curve(
        self,
        history: tf.keras.callbacks.History,
    ) -> None:
        """Save loss and Dice coefficient curves to outputs/figures/.

        Parameters
        ----------
        history : tf.keras.callbacks.History
            Object returned by model.fit().
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = range(1, len(history.history["loss"]) + 1)

        # Loss
        axes[0].plot(epochs, history.history["loss"], label="Train Loss", linewidth=2)
        axes[0].plot(epochs, history.history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title("Training vs Validation Loss", fontsize=14)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE + Dice Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Dice coefficient
        dice_key = "dice_coefficient"
        if dice_key in history.history:
            axes[1].plot(
                epochs, history.history[dice_key], label="Train Dice", linewidth=2
            )
            axes[1].plot(
                epochs,
                history.history[f"val_{dice_key}"],
                label="Val Dice",
                linewidth=2,
            )
            axes[1].set_title("Training vs Validation Dice Coefficient", fontsize=14)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Dice Coefficient")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = self.figures_dir / "training_curves.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training curves saved → %s", out_path)
