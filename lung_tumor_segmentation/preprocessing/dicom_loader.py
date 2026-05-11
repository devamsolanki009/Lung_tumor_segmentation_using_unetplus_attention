"""DICOM loading utilities for the Lung Tumor Segmentation project.

Provides DICOMLoader — a class that handles reading DICOM series from the
NSCLC-Radiomics dataset, detecting CT vs. segmentation series, and
producing raw NumPy volumes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class DICOMLoader:
    """Loads and identifies DICOM series for each patient directory.

    Parameters
    ----------
    segmentation_dir_keyword : str
        Substring present in segmentation series folder names.
        Default: ``"300.000000-Segmentation-"`` (NSCLC-Radiomics convention).
    """

    def __init__(
        self,
        segmentation_dir_keyword: str = "300.000000-Segmentation-",
    ) -> None:
        self.seg_keyword = segmentation_dir_keyword

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_patient_list(self, dataset_dir: str | Path) -> List[Path]:
        """Return a sorted list of valid patient sub-directories.

        Parameters
        ----------
        dataset_dir : str or Path
            Root directory that contains one sub-folder per patient
            (e.g. ``LUNG1-001``, ``LUNG1-002``, …).

        Returns
        -------
        List[Path]
            Sorted list of patient directory paths.
        """
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        patients = sorted(
            [p for p in dataset_dir.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )
        logger.info("Found %d patient directories in %s", len(patients), dataset_dir)
        return patients

    def load_patient_series(self, patient_dir: str | Path) -> Dict[str, dict]:
        """Walk a patient directory and collect metadata per sub-folder.

        Returns
        -------
        dict
            Mapping ``folder_path`` → ``{"modality": str, "n_files": int}``
        """
        patient_dir = Path(patient_dir)
        series_info: Dict[str, dict] = {}

        for root, _dirs, files in os.walk(patient_dir):
            dcm_files = [f for f in files if f.lower().endswith(".dcm")]
            if not dcm_files:
                continue
            try:
                sample = pydicom.dcmread(
                    os.path.join(root, dcm_files[0]), force=True
                )
                modality = getattr(sample, "Modality", "UNKNOWN")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read DICOM header in %s: %s", root, exc)
                modality = "UNKNOWN"

            series_info[root] = {
                "modality": modality,
                "n_files": len(dcm_files),
                "files": sorted(dcm_files),
            }

        return series_info

    def detect_ct_series(self, series_info: Dict[str, dict]) -> Optional[str]:
        """Return the path of the CT series (Modality == 'CT').

        Picks the folder with Modality='CT' and the most DICOM files.
        """
        ct_candidates = {
            path: meta
            for path, meta in series_info.items()
            if meta["modality"] == "CT"
            and self.seg_keyword not in path
        }
        if not ct_candidates:
            return None
        # Prefer folder with most files
        return max(ct_candidates, key=lambda p: ct_candidates[p]["n_files"])

    def detect_seg_series(self, series_info: Dict[str, dict]) -> Optional[str]:
        """Return the path of the segmentation series.

        Detects by Modality in ['SEG', 'RTSTRUCT'] **or** by the
        segmentation keyword appearing in the folder path.
        """
        for path, meta in series_info.items():
            if meta["modality"] in ("SEG", "RTSTRUCT") or self.seg_keyword in path:
                return path
        return None

    def load_ct_volume(self, series_path: str | Path) -> np.ndarray:
        """Load a CT DICOM series and return a 3-D NumPy array.

        Returns
        -------
        np.ndarray
            Shape ``(H, W, num_slices)``, dtype ``float32``.
        """
        series_path = Path(series_path)
        dcm_files = sorted(
            [f for f in series_path.iterdir() if f.suffix.lower() == ".dcm"]
        )
        if not dcm_files:
            raise ValueError(f"No DICOM files found in {series_path}")

        slices = []
        for f in dcm_files:
            try:
                ds = sitk.ReadImage(str(f))
                arr = sitk.GetArrayFromImage(ds)
                # SimpleITK gives (slices, H, W) — take first plane
                slices.append(arr[0].astype(np.float32))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping file %s: %s", f.name, exc)

        if not slices:
            raise ValueError(f"Could not load any slices from {series_path}")

        volume = np.stack(slices, axis=-1)  # (H, W, num_slices)
        return volume

    def load_seg_masks(self, series_path: str | Path) -> np.ndarray:
        """Load a segmentation DICOM file and return a 3-D binary mask.

        Returns
        -------
        np.ndarray
            Shape ``(num_slices, H, W)``, dtype ``uint8`` (0/1).
        """
        series_path = Path(series_path)
        dcm_files = sorted(
            [f for f in series_path.iterdir() if f.suffix.lower() == ".dcm"]
        )
        if not dcm_files:
            raise ValueError(f"No DICOM files found in {series_path}")

        # Segmentation is typically a single multi-frame DICOM
        seg_file = dcm_files[0]
        try:
            seg_img = sitk.ReadImage(str(seg_file))
            seg_array = sitk.GetArrayFromImage(seg_img)  # (slices, H, W)
            return (seg_array > 0).astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load segmentation mask from {seg_file}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_ct_and_seg_paths(
        self, patient_dir: str | Path
    ) -> Tuple[Optional[str], Optional[str]]:
        """Detect and return (ct_series_path, seg_series_path) for a patient."""
        series_info = self.load_patient_series(patient_dir)
        ct_path = self.detect_ct_series(series_info)
        seg_path = self.detect_seg_series(series_info)
        return ct_path, seg_path
