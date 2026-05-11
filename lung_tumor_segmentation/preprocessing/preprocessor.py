"""Preprocessing utilities — resizing, normalization, and patient-level pipeline.

Provides the Preprocessor class which operates on raw NumPy volumes produced
by DICOMLoader and returns clean, model-ready slice arrays.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from skimage.transform import resize

from lung_tumor_segmentation.preprocessing.dicom_loader import DICOMLoader

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses CT volumes and segmentation masks for model training.

    Parameters
    ----------
    config : dict
        The full project config dict (loaded via config.loader.load_config).
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        prep = config["preprocessing"]
        self.target_size: Tuple[int, int] = tuple(prep["target_size"])
        self.normalize: bool = prep["min_max_normalize"]
        self.skip_empty: bool = prep["skip_empty_masks"]
        self._loader = DICOMLoader()

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------

    def resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize each slice of a (H, W, num_slices) volume to target_size.

        Parameters
        ----------
        volume : np.ndarray
            Shape ``(H, W, num_slices)`` or ``(num_slices, H, W)``.

        Returns
        -------
        np.ndarray
            Shape ``(target_H, target_W, num_slices)``, dtype ``float32``.
        """
        # Normalise to (H, W, num_slices)
        if volume.ndim == 3 and volume.shape[0] < volume.shape[2]:
            volume = np.transpose(volume, (1, 2, 0))

        num_slices = volume.shape[2]
        resized = np.zeros((*self.target_size, num_slices), dtype=np.float32)
        for i in range(num_slices):
            resized[:, :, i] = resize(
                volume[:, :, i],
                self.target_size,
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
        return resized

    def resize_masks(self, masks: np.ndarray) -> np.ndarray:
        """Resize each mask slice to target_size (nearest-neighbour, no AA).

        Parameters
        ----------
        masks : np.ndarray
            Shape ``(num_slices, H, W)`` (output of DICOMLoader.load_seg_masks).

        Returns
        -------
        np.ndarray
            Shape ``(num_slices, target_H, target_W)``, dtype ``uint8``.
        """
        num_slices = masks.shape[0]
        resized = np.zeros((num_slices, *self.target_size), dtype=np.uint8)
        for i in range(num_slices):
            resized[i] = resize(
                masks[i],
                self.target_size,
                order=0,  # nearest-neighbour keeps binary values
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.uint8)
        return resized

    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Apply min-max normalization to map values to [0, 1].

        Parameters
        ----------
        volume : np.ndarray
            Any shape, dtype convertible to float32.

        Returns
        -------
        np.ndarray
            Same shape, dtype ``float32``, values in ``[0.0, 1.0]``.
        """
        vol = volume.astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin < 1e-6:
            return np.zeros_like(vol)
        return (vol - vmin) / (vmax - vmin)

    def filter_empty_masks(
        self,
        cts: np.ndarray,
        masks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove slice pairs where the mask is entirely zero.

        Parameters
        ----------
        cts : np.ndarray
            Shape ``(num_slices, H, W)``.
        masks : np.ndarray
            Shape ``(num_slices, H, W)``.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered (cts, masks) arrays.
        """
        non_empty = [i for i in range(masks.shape[0]) if masks[i].max() > 0]
        if not non_empty:
            return cts[:0], masks[:0]  # empty arrays
        return cts[non_empty], masks[non_empty]

    # ------------------------------------------------------------------
    # Patient-level pipeline
    # ------------------------------------------------------------------

    def process_patient(
        self,
        patient_dir: str | Path,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Run the full preprocessing pipeline for one patient.

        Steps
        -----
        1. Detect CT and segmentation series.
        2. Load volumes.
        3. Resize to target_size.
        4. Normalize CT (if configured).
        5. Filter empty masks (if configured).

        Parameters
        ----------
        patient_dir : str or Path
            Path to the patient's root DICOM folder.

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            ``(cts, masks)`` each of shape ``(num_slices, H, W)``,
            or ``None`` if CT/mask series could not be found.
        """
        patient_dir = Path(patient_dir)
        patient_id = patient_dir.name

        ct_path, seg_path = self._loader.get_ct_and_seg_paths(patient_dir)
        if ct_path is None:
            logger.warning("[%s] No CT series found — skipping.", patient_id)
            return None
        if seg_path is None:
            logger.warning("[%s] No segmentation series found — skipping.", patient_id)
            return None

        try:
            ct_volume = self._loader.load_ct_volume(ct_path)  # (H, W, num_slices)
            seg_masks = self._loader.load_seg_masks(seg_path)  # (num_slices, H, W)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] Failed to load volume: %s", patient_id, exc)
            return None

        # Resize
        ct_resized = self.resize_volume(ct_volume)  # (H, W, num_slices)
        mask_resized = self.resize_masks(seg_masks)  # (num_slices, H, W)

        # Convert CT to (num_slices, H, W)
        ct_slices = np.transpose(ct_resized, (2, 0, 1))

        # Align slice counts (may differ due to segmentation spanning fewer slices)
        n_ct = ct_slices.shape[0]
        n_mask = mask_resized.shape[0]
        n = min(n_ct, n_mask)
        ct_slices = ct_slices[:n]
        mask_resized = mask_resized[:n]

        # Normalize CT
        if self.normalize:
            ct_slices = self.normalize_volume(ct_slices)

        # Filter empty masks
        if self.skip_empty:
            ct_slices, mask_resized = self.filter_empty_masks(ct_slices, mask_resized)

        if ct_slices.shape[0] == 0:
            logger.warning("[%s] No non-empty mask slices — skipping.", patient_id)
            return None

        logger.info(
            "[%s] Processed %d slices (after filtering).",
            patient_id,
            ct_slices.shape[0],
        )
        return ct_slices, mask_resized

    def save_patient_arrays(
        self,
        patient_id: str,
        cts: np.ndarray,
        masks: np.ndarray,
        ct_output_dir: str | Path,
        mask_output_dir: str | Path,
    ) -> Dict[str, list]:
        """Save per-slice .npy files for a patient.

        Parameters
        ----------
        patient_id : str
            Patient identifier string (e.g. ``"LUNG1-001"``).
        cts : np.ndarray
            Shape ``(num_slices, H, W)``.
        masks : np.ndarray
            Shape ``(num_slices, H, W)``.
        ct_output_dir : str or Path
        mask_output_dir : str or Path

        Returns
        -------
        dict
            ``{"ct_files": [...], "mask_files": [...]}``
        """
        ct_output_dir = Path(ct_output_dir)
        mask_output_dir = Path(mask_output_dir)
        ct_output_dir.mkdir(parents=True, exist_ok=True)
        mask_output_dir.mkdir(parents=True, exist_ok=True)

        ct_files, mask_files = [], []
        for i in range(cts.shape[0]):
            ct_fname = ct_output_dir / f"{patient_id}_ct_slice_{i:03d}.npy"
            mask_fname = mask_output_dir / f"{patient_id}_mask_slice_{i:03d}.npy"
            np.save(ct_fname, cts[i])
            np.save(mask_fname, masks[i])
            ct_files.append(str(ct_fname))
            mask_files.append(str(mask_fname))

        return {"ct_files": ct_files, "mask_files": mask_files}
