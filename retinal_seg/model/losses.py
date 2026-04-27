"""Loss functions and evaluation metrics for multi-class retinal segmentation.

All callables follow the Keras loss/metric signature:
    fn(y_true, y_pred) -> scalar tensor

y_true: one-hot encoded ground truth  (batch, H, W, num_classes), float32
y_pred: softmax model output           (batch, H, W, num_classes), float32
"""
from __future__ import annotations

import keras
import keras.ops as ops
from keras.losses import binary_crossentropy, categorical_crossentropy

from retinal_seg.config import CLASS_WEIGHTS, DICE_SMOOTH, NUM_CLASSES

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


def dice_coef(y_true, y_pred):
    """Global Dice coefficient across all classes (flattened).

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar Dice coefficient of dtype float32 in [0, 1].
    """
    y_true_f = ops.reshape(y_true, [-1])
    y_pred_f = ops.reshape(y_pred, [-1])
    intersection = ops.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + DICE_SMOOTH) / (
        ops.sum(y_true_f) + ops.sum(y_pred_f) + DICE_SMOOTH
    )


def dice_coef_per_class(y_true, y_pred):
    """Macro-averaged Dice coefficient computed per class.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax predictions of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar mean Dice of dtype float32 across NUM_CLASSES classes.
    """
    scores = []
    for idx in range(NUM_CLASSES):
        y_true_f = ops.reshape(y_true[:, :, :, idx], [-1])
        y_pred_f = ops.reshape(y_pred[:, :, :, idx], [-1])
        intersection = ops.sum(y_true_f * y_pred_f)
        denom = ops.sum(y_true_f + y_pred_f)
        scores.append(2.0 * (intersection + DICE_SMOOTH) / (denom + DICE_SMOOTH))
    return ops.mean(ops.stack(scores))


# ------------------------------------------------------------------
# Core Dice losses
# ------------------------------------------------------------------


def dice_loss(y_true, y_pred):
    """Soft Dice loss: 1 − dice_coef (all classes, flattened).

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar loss of dtype float32 in [0, 1].
    """
    y_true_f = ops.reshape(y_true, [-1])
    y_pred_f = ops.reshape(y_pred, [-1])
    intersection = ops.sum(y_true_f * y_pred_f)
    denom = ops.sum(y_true_f + y_pred_f)
    return ops.cast(
        1.0 - 2.0 * (intersection + DICE_SMOOTH) / (denom + DICE_SMOOTH),
        "float32",
    )


def dice_loss_inner_layers(y_true, y_pred):
    """Soft Dice loss computed only on non-background classes (index 1+).

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax predictions of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar Dice loss of dtype float32 for foreground classes only.
    """
    y_true_f = ops.reshape(y_true[:, :, :, 1:], [-1])
    y_pred_f = ops.reshape(y_pred[:, :, :, 1:], [-1])
    intersection = ops.sum(y_true_f * y_pred_f)
    denom = ops.sum(y_true_f + y_pred_f)
    return 1.0 - 2.0 * (intersection + DICE_SMOOTH) / (denom + DICE_SMOOTH)


# ------------------------------------------------------------------
# Cross-entropy losses
# ------------------------------------------------------------------


def weighted_ce(y_true, y_pred, weights=None):
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
        weights = ops.convert_to_tensor([list(CLASS_WEIGHTS)], dtype="float32")
    return categorical_crossentropy(y_true * weights, y_pred)


# ------------------------------------------------------------------
# Combined losses (primary training objectives)
# ------------------------------------------------------------------


def weighted_ce_dice(y_true, y_pred):
    """Weighted CE + soft Dice loss — primary training loss.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar combined loss of dtype float32.
    """
    return dice_loss(y_true, y_pred) + weighted_ce(y_true, y_pred)


def ce_dice(y_true, y_pred):
    """Unweighted categorical CE + soft Dice loss.

    Args:
        y_true: One-hot ground-truth of dtype float32, shape (batch, H, W, C).
        y_pred: Softmax prediction of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar combined loss of dtype float32.
    """
    return dice_loss(y_true, y_pred) + categorical_crossentropy(y_true, y_pred)


def weighted_dice_with_categorical_ce(y_true, y_pred):
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


def tversky(y_true, y_pred, alpha: float = 0.7):
    """Tversky similarity index (asymmetrically weighted Dice).

    Args:
        y_true: Ground-truth tensor of dtype float32, shape (batch, H, W, C).
        y_pred: Prediction tensor of dtype float32, shape (batch, H, W, C).
        alpha: Weight for false negatives; (1−alpha) weights false positives.
            alpha > 0.5 penalises false negatives more heavily.

    Returns:
        Scalar Tversky index of dtype float32 in [0, 1].
    """
    y_true_f = ops.reshape(y_true, [-1])
    y_pred_f = ops.reshape(y_pred, [-1])
    true_pos = ops.sum(y_true_f * y_pred_f)
    false_neg = ops.sum(y_true_f * (1.0 - y_pred_f))
    false_pos = ops.sum((1.0 - y_true_f) * y_pred_f)
    return (true_pos + DICE_SMOOTH) / (
        true_pos + alpha * false_neg + (1.0 - alpha) * false_pos + DICE_SMOOTH
    )


def tversky_loss(y_true, y_pred):
    """Tversky loss: 1 − tversky index.

    Args:
        y_true: Ground-truth tensor of dtype float32, shape (batch, H, W, C).
        y_pred: Prediction tensor of dtype float32, shape (batch, H, W, C).

    Returns:
        Scalar Tversky loss of dtype float32.
    """
    return 1.0 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma: float = 0.75):
    """Focal Tversky loss — focuses training on hard, misclassified examples.

    Args:
        y_true: Ground-truth tensor of dtype float32, shape (batch, H, W, C).
        y_pred: Prediction tensor of dtype float32, shape (batch, H, W, C).
        gamma: Focusing exponent; higher values concentrate on hard examples.

    Returns:
        Scalar focal Tversky loss of dtype float32.
    """
    return 1.0 - ops.power(tversky(y_true, y_pred), gamma)


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

    def _focal_loss_fn(y_true, y_pred):
        epsilon = 1e-9
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        model_out = y_pred + epsilon
        ce = y_true * (-ops.log(model_out))
        weight = y_true * ops.power(1.0 - model_out, gamma)
        fl = alpha * weight * ce
        return ops.mean(ops.max(fl, axis=1))

    return _focal_loss_fn
