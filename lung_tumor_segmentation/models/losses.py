"""Custom loss functions and metrics for lung tumor segmentation.

All functions are compatible with TensorFlow/Keras (operate on tf.Tensor)
and can be passed directly to model.compile().
"""

from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def dice_coefficient(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """Sørensen–Dice coefficient — primary segmentation metric.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth binary mask, values in {0, 1}.
    y_pred : tf.Tensor
        Predicted probability map, values in [0, 1].
    smooth : float
        Small constant to avoid division by zero.

    Returns
    -------
    tf.Tensor
        Scalar Dice coefficient in [0, 1].
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Differentiable Dice loss = 1 - Dice coefficient."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> tf.Tensor:
    """Weighted sum of Binary Cross-Entropy and Dice loss.

    This combined loss handles class imbalance better than BCE alone and
    provides smoother gradients than Dice alone at the start of training.

    Parameters
    ----------
    y_true : tf.Tensor
    y_pred : tf.Tensor
    bce_weight : float
        Weight for the BCE component (default 0.5).
    dice_weight : float
        Weight for the Dice component (default 0.5).

    Returns
    -------
    tf.Tensor
        Scalar combined loss.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dl = dice_loss(y_true, y_pred)
    return bce_weight * bce + dice_weight * dl


# ---------------------------------------------------------------------------
# Additional metrics
# ---------------------------------------------------------------------------

def iou_metric(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """Intersection over Union (Jaccard index)."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_metric(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    threshold: float = 0.5,
) -> tf.Tensor:
    """Precision = TP / (TP + FP)."""
    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    return tp / (tp + fp + 1e-6)


def recall_metric(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    threshold: float = 0.5,
) -> tf.Tensor:
    """Recall (Sensitivity) = TP / (TP + FN)."""
    y_pred_bin = tf.cast(y_pred >= threshold, tf.float32)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    return tp / (tp + fn + 1e-6)
