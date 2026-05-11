"""Pure NumPy evaluation metrics for semantic segmentation.

All functions accept NumPy arrays and operate **independently** of TensorFlow —
suitable for post-training analysis on the held-out test set.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Dice coefficient (F1 score) between two binary arrays.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary mask (0/1), any shape.
    y_pred : np.ndarray
        Predicted probability map or binary mask, same shape as y_true.
    threshold : float
        Binarisation threshold applied to y_pred.

    Returns
    -------
    float
        Dice score in [0, 1].  Returns 1.0 when both masks are empty.
    """
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    intersection = np.sum(y_true_bin * y_pred_bin)
    denom = np.sum(y_true_bin) + np.sum(y_pred_bin)
    if denom == 0:
        return 1.0  # both masks are empty → perfect agreement
    return float(2.0 * intersection / denom)


def compute_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Intersection over Union (Jaccard index).

    Returns 1.0 when both masks are empty.
    """
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Precision = TP / (TP + FP).  Returns 1.0 when no positives predicted."""
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    tp = np.sum(y_true_bin * y_pred_bin)
    fp = np.sum((1 - y_true_bin) * y_pred_bin)
    if tp + fp == 0:
        return 1.0
    return float(tp / (tp + fp))


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Recall (sensitivity) = TP / (TP + FN).  Returns 1.0 when no positives in GT."""
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    tp = np.sum(y_true_bin * y_pred_bin)
    fn = np.sum(y_true_bin * (1 - y_pred_bin))
    if tp + fn == 0:
        return 1.0
    return float(tp / (tp + fn))


def compute_f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Specificity (true negative rate) = TN / (TN + FP)."""
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    tn = np.sum((1 - y_true_bin) * (1 - y_pred_bin))
    fp = np.sum((1 - y_true_bin) * y_pred_bin)
    if tn + fp == 0:
        return 1.0
    return float(tn / (tn + fp))


def compute_volumetric_similarity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Volumetric similarity = 1 - |V_pred - V_true| / (V_pred + V_true).

    Returns 1.0 when both volumes are zero.
    """
    y_pred_bin = (y_pred >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    v_pred = float(np.sum(y_pred_bin))
    v_true = float(np.sum(y_true_bin))
    if v_pred + v_true == 0:
        return 1.0
    return float(1.0 - abs(v_pred - v_true) / (v_pred + v_true))


def compute_hausdorff_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """95th-percentile Hausdorff distance using distance transforms.

    Returns 0.0 when both masks are empty, and a large sentinel (999.0)
    when only one mask is empty (undefined boundary).
    """
    y_pred_bin = (y_pred >= threshold).astype(bool)
    y_true_bin = (y_true >= 0.5).astype(bool)

    if not y_true_bin.any() and not y_pred_bin.any():
        return 0.0
    if not y_true_bin.any() or not y_pred_bin.any():
        return 999.0  # sentinel — one boundary is empty

    # Distance from each GT boundary voxel to nearest predicted boundary
    dist_gt_to_pred = distance_transform_edt(~y_pred_bin)[y_true_bin]
    # Distance from each predicted boundary voxel to nearest GT boundary
    dist_pred_to_gt = distance_transform_edt(~y_true_bin)[y_pred_bin]

    hd95 = float(
        max(
            np.percentile(dist_gt_to_pred, 95),
            np.percentile(dist_pred_to_gt, 95),
        )
    )
    return hd95


# ---------------------------------------------------------------------------
# Aggregated compute
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the full suite of metrics for a single slice.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth mask.
    y_pred : np.ndarray
        Predicted probability map.
    threshold : float
        Binarisation threshold.

    Returns
    -------
    dict
        Keys: dice, iou, precision, recall, f1, specificity,
              volumetric_similarity, hausdorff_distance.
    """
    dice = compute_dice(y_true, y_pred, threshold)
    iou = compute_iou(y_true, y_pred, threshold)
    precision = compute_precision(y_true, y_pred, threshold)
    recall = compute_recall(y_true, y_pred, threshold)
    f1 = compute_f1(precision, recall)
    specificity = compute_specificity(y_true, y_pred, threshold)
    vol_sim = compute_volumetric_similarity(y_true, y_pred, threshold)
    hausdorff = compute_hausdorff_distance(y_true, y_pred, threshold)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "volumetric_similarity": vol_sim,
        "hausdorff_distance": hausdorff,
    }
