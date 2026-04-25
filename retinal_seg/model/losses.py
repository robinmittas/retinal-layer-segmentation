"""Loss functions and evaluation metrics for multi-class retinal segmentation.

All callables follow the Keras loss/metric signature:
    fn(y_true, y_pred) -> scalar tensor

y_true: one-hot encoded ground truth  (batch, H, W, num_classes), float32
y_pred: softmax model output           (batch, H, W, num_classes), float32
"""
from __future__ import annotations

import tensorflow as tf
import keras
from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy

from retinal_seg.config import CLASS_WEIGHTS, DICE_SMOOTH, NUM_CLASSES

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Global Dice coefficient across all classes (flattened).

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar Dice coefficient of dtype float32 in [0, 1].
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + DICE_SMOOTH) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + DICE_SMOOTH
    )


def dice_coef_per_class(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Macro-averaged Dice coefficient computed per class.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax predictions of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar mean Dice of dtype float32 across NUM_CLASSES classes.
    """
    scores = []
    for idx in range(NUM_CLASSES):
        y_true_f = K.flatten(y_true[:, :, :, idx])
        y_pred_f = K.flatten(y_pred[:, :, :, idx])
        intersection = K.sum(y_true_f * y_pred_f)
        denom = K.sum(y_true_f + y_pred_f)
        scores.append(2.0 * (intersection + DICE_SMOOTH) / (denom + DICE_SMOOTH))
    return tf.math.reduce_mean(scores)


# ------------------------------------------------------------------
# Core Dice losses
# ------------------------------------------------------------------


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Soft Dice loss: 1 − dice_coef (all classes, flattened).

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar loss of dtype float32 in [0, 1].
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f + y_pred_f)
    return K.cast(
        1.0 - 2.0 * (intersection + DICE_SMOOTH) / (denom + DICE_SMOOTH),
        tf.float32,
    )


def dice_loss_inner_layers(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Soft Dice loss computed only on non-background classes (index 1+).

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax predictions of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar Dice loss of dtype float32 for foreground classes only.
    """
    y_true_f = K.flatten(y_true[:, :, :, 1:])
    y_pred_f = K.flatten(y_pred[:, :, :, 1:])
    intersection = K.sum(y_true_f * y_pred_f)
    denom = K.sum(y_true_f + y_pred_f)
    return 1.0 - 2.0 * (intersection + DICE_SMOOTH) / (denom + DICE_SMOOTH)


# ------------------------------------------------------------------
# Cross-entropy losses
# ------------------------------------------------------------------


def weighted_ce(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """Weighted categorical cross-entropy with per-class scaling.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax predictions of dtype float32, shape (batch, H, W, C).
        weights: Tensor of dtype float32, shape (1, C) with per-class multipliers.
            Defaults to CLASS_WEIGHTS from config.

    Returns:
        Scalar weighted cross-entropy loss of dtype float32.
    """
    if weights is None:
        weights = tf.constant([list(CLASS_WEIGHTS)], dtype=tf.float32)
    return categorical_crossentropy(y_true * weights, y_pred)


# ------------------------------------------------------------------
# Combined losses (primary training objectives)
# ------------------------------------------------------------------


def weighted_ce_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weighted CE + soft Dice loss — primary training loss.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar combined loss of dtype float32.
    """
    return dice_loss(y_true, y_pred) + weighted_ce(y_true, y_pred)


def ce_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Unweighted categorical CE + soft Dice loss.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar combined loss of dtype float32.
    """
    return dice_loss(y_true, y_pred) + categorical_crossentropy(y_true, y_pred)


def weighted_dice_with_categorical_ce(
    y_true: tf.Tensor, y_pred: tf.Tensor
) -> tf.Tensor:
    """0.2 × Dice loss + categorical cross-entropy.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar combined loss of dtype float32.
    """
    return 0.2 * dice_loss(y_true, y_pred) + categorical_crossentropy(y_true, y_pred)


# ------------------------------------------------------------------
# Tversky-based losses
# ------------------------------------------------------------------


def tversky(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.7,
) -> tf.Tensor:
    """Tversky similarity index (asymmetrically weighted Dice).

    Args:
        y_true: Ground-truth tensor of dtype float32, shape (batch, H, W, C).
        y_pred: Prediction tensor of dtype float32, shape (batch, H, W, C).
        alpha: Weight for false negatives; (1−alpha) weights false positives.
            alpha > 0.5 penalises false negatives more heavily.

    Returns:
        Scalar Tversky index of dtype float32 in [0, 1].
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1.0 - y_pred_f))
    false_pos = K.sum((1.0 - y_true_f) * y_pred_f)
    return (true_pos + DICE_SMOOTH) / (
        true_pos + alpha * false_neg + (1.0 - alpha) * false_pos + DICE_SMOOTH
    )


def tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Tversky loss: 1 − tversky index.

    Args:
        y_true: Ground-truth tensor of dtype float32, shape (batch, H, W, C).
        y_pred: Prediction tensor of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar Tversky loss of dtype float32.
    """
    return 1.0 - tversky(y_true, y_pred)


def focal_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float = 0.75,
) -> tf.Tensor:
    """Focal Tversky loss — focuses training on hard, misclassified examples.

    Args:
        y_true: Ground-truth tensor of dtype float32, shape (batch, H, W, C).
        y_pred: Prediction tensor of dtype float32, shape (batch, H, W, C).
        gamma: Focusing exponent; higher values concentrate on hard examples.

    Returns:
        Scalar focal Tversky loss of dtype float32.
    """
    return 1.0 - keras.ops.power(tversky(y_true, y_pred), gamma)


# ------------------------------------------------------------------
# Focal loss
# ------------------------------------------------------------------


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Create a multi-class focal loss function.

    FL(p_t) = −alpha · (1 − p_t)^gamma · log(p_t)

    Args:
        gamma: Focusing parameter; reduces weight for well-classified
            examples. gamma=0 recovers standard cross-entropy.
        alpha: Scalar class-balancing factor.

    Returns:
        Callable focal_loss_fn(y_true, y_pred) → scalar loss tensor.
    """
    gamma = float(gamma)
    alpha = float(alpha)

    def _focal_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        epsilon = 1e-9
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        model_out = y_pred + epsilon
        ce = y_true * (-tf.math.log(model_out))
        weight = y_true * tf.pow(1.0 - model_out, gamma)
        fl = alpha * weight * ce
        return tf.reduce_mean(tf.reduce_max(fl, axis=1))

    return _focal_loss_fn
