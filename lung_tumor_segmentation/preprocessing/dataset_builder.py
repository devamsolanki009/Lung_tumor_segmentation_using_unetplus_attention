"""DatasetBuilder — loads preprocessed .npy slices, builds train/val/test splits,
and provides Keras data generators with optional augmentation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Type alias
SplitTuple = Tuple[
    np.ndarray, np.ndarray,  # X_train, y_train
    np.ndarray, np.ndarray,  # X_val,   y_val
    np.ndarray, np.ndarray,  # X_test,  y_test
]


class DatasetBuilder:
    """Builds train / val / test splits from preprocessed .npy slice files.

    Parameters
    ----------
    config : dict
        Full project config dict.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        prep = config["preprocessing"]
        self.train_ratio: float = prep["train_ratio"]
        self.val_ratio: float = prep["val_ratio"]
        self.test_ratio: float = prep["test_ratio"]
        self.random_seed: int = prep["random_seed"]
        self.input_shape: Tuple[int, int] = tuple(config["model"]["input_shape"][:2])

    # ------------------------------------------------------------------
    # Building splits
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        cts_dir: str | Path,
        masks_dir: str | Path,
    ) -> SplitTuple:
        """Load all .npy files, shuffle, and split into train/val/test.

        The split is done **patient-aware**: slices from the same patient
        are kept together in a single split to avoid data leakage.

        Parameters
        ----------
        cts_dir : str or Path
            Directory containing ``*_ct_slice_*.npy`` files.
        masks_dir : str or Path
            Directory containing ``*_mask_slice_*.npy`` files.

        Returns
        -------
        SplitTuple
            ``(X_train, y_train, X_val, y_val, X_test, y_test)``
            where X arrays are shape ``(N, H, W, 1)`` and y arrays are
            shape ``(N, H, W, 1)``, all ``float32``.
        """
        cts_dir = Path(cts_dir)
        masks_dir = Path(masks_dir)

        # Collect all patient IDs present in both dirs
        patient_ids = self._get_patient_ids(cts_dir, masks_dir)
        logger.info("Found %d unique patients with preprocessed data.", len(patient_ids))

        # Shuffle patients
        rng = np.random.default_rng(self.random_seed)
        patient_ids = list(patient_ids)
        rng.shuffle(patient_ids)

        # Split patient list
        n = len(patient_ids)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        train_pids = patient_ids[:n_train]
        val_pids = patient_ids[n_train: n_train + n_val]
        test_pids = patient_ids[n_train + n_val:]

        logger.info(
            "Split: %d train / %d val / %d test patients.",
            len(train_pids), len(val_pids), len(test_pids),
        )

        X_train, y_train = self._load_patient_slices(train_pids, cts_dir, masks_dir)
        X_val, y_val = self._load_patient_slices(val_pids, cts_dir, masks_dir)
        X_test, y_test = self._load_patient_slices(test_pids, cts_dir, masks_dir)

        logger.info(
            "Slice counts — train: %d, val: %d, test: %d",
            len(X_train), len(X_val), len(X_test),
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_splits(self, splits: SplitTuple, output_dir: str | Path) -> None:
        """Save all six split arrays as .npy files.

        Parameters
        ----------
        splits : SplitTuple
            Result of build_dataset().
        output_dir : str or Path
            Directory to save into (will be created if missing).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
        for name, arr in zip(names, splits):
            path = output_dir / f"{name}.npy"
            np.save(path, arr)
            logger.info("Saved %s → %s  (shape %s)", name, path, arr.shape)

    def load_splits(self, splits_dir: str | Path) -> SplitTuple:
        """Load pre-saved split .npy files.

        Parameters
        ----------
        splits_dir : str or Path
            Directory previously written by save_splits().

        Returns
        -------
        SplitTuple
        """
        splits_dir = Path(splits_dir)
        names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
        arrays = []
        for name in names:
            path = splits_dir / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}")
            arrays.append(np.load(path))
            logger.info("Loaded %s from %s (shape %s)", name, path, arrays[-1].shape)
        return tuple(arrays)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Keras data generator
    # ------------------------------------------------------------------

    def get_data_generator(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        augment: bool = False,
    ) -> Generator:
        """Yield (batch_X, batch_y) pairs indefinitely.

        Parameters
        ----------
        X : np.ndarray
            Input CT slices, shape ``(N, H, W, 1)``.
        y : np.ndarray
            Mask slices, shape ``(N, H, W, 1)``.
        batch_size : int
        augment : bool
            If True, applies random horizontal flip to each batch.

        Yields
        ------
        Tuple[np.ndarray, np.ndarray]
        """
        n = len(X)
        indices = np.arange(n)
        rng = np.random.default_rng(self.random_seed)

        while True:
            rng.shuffle(indices)
            for start in range(0, n, batch_size):
                batch_idx = indices[start: start + batch_size]
                bx = X[batch_idx].copy()
                by = y[batch_idx].copy()

                if augment:
                    bx, by = self._augment_batch(bx, by, rng)

                yield bx, by

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_patient_ids(cts_dir: Path, masks_dir: Path) -> List[str]:
        """Return patient IDs present in both CT and mask directories."""
        ct_files = {f.stem for f in cts_dir.glob("*_ct_slice_*.npy")}
        mask_files = {f.stem for f in masks_dir.glob("*_mask_slice_*.npy")}

        # Extract patient IDs: e.g. "LUNG1-001_ct_slice_000" → "LUNG1-001"
        def _pid(stem: str, tag: str) -> str:
            return stem.split(tag)[0].rstrip("_")

        ct_pids = {_pid(s, "_ct_slice_") for s in ct_files}
        mask_pids = {_pid(s, "_mask_slice_") for s in mask_files}
        return sorted(ct_pids & mask_pids)

    def _load_patient_slices(
        self,
        patient_ids: List[str],
        cts_dir: Path,
        masks_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and concatenate slices for a list of patient IDs."""
        all_cts, all_masks = [], []
        for pid in patient_ids:
            ct_files = sorted(cts_dir.glob(f"{pid}_ct_slice_*.npy"))
            mask_files = sorted(masks_dir.glob(f"{pid}_mask_slice_*.npy"))
            for cf, mf in zip(ct_files, mask_files):
                ct = np.load(cf).astype(np.float32)
                mask = np.load(mf).astype(np.float32)
                # Add channel dimension: (H, W) → (H, W, 1)
                all_cts.append(ct[..., np.newaxis])
                all_masks.append(mask[..., np.newaxis])

        if not all_cts:
            h, w = self.input_shape
            return np.empty((0, h, w, 1), np.float32), np.empty((0, h, w, 1), np.float32)

        return np.stack(all_cts), np.stack(all_masks)

    @staticmethod
    def _augment_batch(
        bx: np.ndarray,
        by: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random horizontal flip augmentation in-place."""
        for i in range(len(bx)):
            if rng.random() > 0.5:
                bx[i] = np.fliplr(bx[i])
                by[i] = np.fliplr(by[i])
        return bx, by
