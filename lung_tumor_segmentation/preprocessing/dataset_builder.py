"""DatasetBuilder — CSV-index-based streaming dataset for Keras training.

Design
------
Instead of loading all slices into RAM (which would need ~2 GB for 28K files),
this class works with **index CSV files** produced by ``build_splits.py``:

    outputs/splits/train_index.csv   (ct_path, mask_path)
    outputs/splits/val_index.csv
    outputs/splits/test_index.csv

Training uses ``tf.data.Dataset`` to stream .npy files from disk on-demand,
one batch at a time.  RAM usage stays constant regardless of dataset size.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Streaming dataset builder backed by CSV index files.

    Parameters
    ----------
    config : dict
        Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.input_shape: Tuple[int, int] = tuple(config["model"]["input_shape"][:2])
        self.batch_size: int = config["training"]["batch_size"]
        self.prefetch: int = config.get("gpu", {}).get("prefetch_buffer", 2)

    # ------------------------------------------------------------------
    # Index loading helpers
    # ------------------------------------------------------------------

    def load_index(self, csv_path: str | Path) -> Tuple[list, list]:
        """Read a split CSV and return (ct_paths, mask_paths) lists.

        Parameters
        ----------
        csv_path : str or Path
            Path to a CSV with columns ``ct_path, mask_path``.

        Returns
        -------
        Tuple[list, list]
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Index CSV not found: {csv_path}\n"
                "Run  python build_splits.py  first."
            )
        ct_paths, mask_paths = [], []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ct_paths.append(row["ct_path"])
                mask_paths.append(row["mask_path"])
        logger.info("Loaded index: %s  (%d slices)", csv_path.name, len(ct_paths))
        return ct_paths, mask_paths

    def load_splits(self, splits_dir: str | Path) -> Tuple[list, list, list, list, list, list]:
        """Load all three index CSVs.

        Parameters
        ----------
        splits_dir : str or Path
            Directory containing train_index.csv, val_index.csv, test_index.csv.

        Returns
        -------
        Tuple
            ``(train_ct, train_mask, val_ct, val_mask, test_ct, test_mask)``
            where each element is a list of file path strings.
        """
        splits_dir = Path(splits_dir)
        train_ct, train_mask = self.load_index(splits_dir / "train_index.csv")
        val_ct,   val_mask   = self.load_index(splits_dir / "val_index.csv")
        test_ct,  test_mask  = self.load_index(splits_dir / "test_index.csv")
        logger.info(
            "Splits loaded — train: %d, val: %d, test: %d slices",
            len(train_ct), len(val_ct), len(test_ct),
        )
        return train_ct, train_mask, val_ct, val_mask, test_ct, test_mask

    # ------------------------------------------------------------------
    # tf.data pipeline
    # ------------------------------------------------------------------

    def get_tf_dataset(
        self,
        ct_paths: list,
        mask_paths: list,
        batch_size: int | None = None,
        augment: bool = False,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """Build a streaming tf.data.Dataset from file path lists.

        Files are loaded on-the-fly by a tf.numpy_function wrapper.
        Memory usage is constant (one batch in RAM at a time).

        Parameters
        ----------
        ct_paths : list[str]   List of absolute paths to CT .npy files.
        mask_paths : list[str] List of absolute paths to mask .npy files.
        batch_size : int, optional   Defaults to config value.
        augment : bool   Apply random horizontal flip augmentation.
        shuffle : bool   Shuffle file order each epoch.

        Returns
        -------
        tf.data.Dataset
            Yields ``(ct_batch, mask_batch)`` tensors of shape
            ``(B, H, W, 1)`` dtype ``float32``.
        """
        batch_size = batch_size or self.batch_size
        h, w = self.input_shape

        # Convert to tensors so tf.data can handle them
        ct_tensor   = tf.constant(ct_paths,   dtype=tf.string)
        mask_tensor = tf.constant(mask_paths, dtype=tf.string)

        ds = tf.data.Dataset.from_tensor_slices((ct_tensor, mask_tensor))

        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(ct_paths), 2000), reshuffle_each_iteration=True)

        # Load .npy files via numpy function
        def _load_pair(ct_path: tf.Tensor, mask_path: tf.Tensor):
            ct, mask = tf.numpy_function(
                func=self._load_npy_pair,
                inp=[ct_path, mask_path],
                Tout=[tf.float32, tf.float32],
            )
            ct.set_shape([h, w, 1])
            mask.set_shape([h, w, 1])
            return ct, mask

        ds = ds.map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            ds = ds.map(self._augment_tf, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(self.prefetch)
        return ds

    # ------------------------------------------------------------------
    # In-memory helpers kept for small datasets / evaluation
    # ------------------------------------------------------------------

    def load_test_arrays(
        self,
        test_ct: list,
        test_mask: list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load the test split fully into RAM (only the test set is small enough).

        Parameters
        ----------
        test_ct : list   CT file paths.
        test_mask : list Mask file paths.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            ``X_test`` shape ``(N, H, W, 1)``, ``y_test`` shape ``(N, H, W, 1)``.
        """
        h, w = self.input_shape
        logger.info("Loading %d test slices into RAM …", len(test_ct))
        cts, masks = [], []
        for cp, mp in zip(test_ct, test_mask):
            ct, mask = self._load_npy_pair(cp, mp)
            cts.append(ct)
            masks.append(mask)
        X = np.stack(cts)   # (N, H, W, 1)
        y = np.stack(masks) # (N, H, W, 1)
        logger.info("Test arrays loaded: X=%s  y=%s", X.shape, y.shape)
        return X, y

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_npy_pair(ct_path, mask_path) -> Tuple[np.ndarray, np.ndarray]:
        """Load one CT + mask .npy pair and add channel dim."""
        # Accept both bytes (from tf) and str
        if isinstance(ct_path, bytes):
            ct_path   = ct_path.decode("utf-8")
            mask_path = mask_path.decode("utf-8")
        ct   = np.load(ct_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        # Ensure (H, W) → (H, W, 1)
        if ct.ndim == 2:
            ct   = ct[..., np.newaxis]
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]
        return ct, mask

    @staticmethod
    def _augment_tf(
        ct: tf.Tensor,
        mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Random horizontal flip augmentation (applied per-sample)."""
        if tf.random.uniform(()) > 0.5:
            ct   = tf.image.flip_left_right(ct)
            mask = tf.image.flip_left_right(mask)
        return ct, mask
