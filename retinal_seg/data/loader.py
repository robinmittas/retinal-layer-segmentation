"""Dataset loading, label remapping, augmentation, and train/val/test splitting.

Typical usage:
    train_ds, val_ds, test_ds = build_datasets(x_y_map, batch_size=8)
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from keras.layers import GaussianNoise

from retinal_seg.config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    LABEL_REMAP,
    NUM_CLASSES,
    SHUFFLE_BUFFER_SIZE,
)


# ------------------------------------------------------------------
# Label utilities
# ------------------------------------------------------------------


def remap_labels(y_data: np.ndarray) -> np.ndarray:
    """Remap 11-class OCTExplorer labels to 4 segmentation classes.

    Applies LABEL_REMAP in-place using np.where to avoid a for-loop copy
    on large arrays.

    Args:
        y_data: Integer label array with values 0–11.

    Returns:
        Label array with values 0–3 (same shape as input).
    """
    for src, dst in LABEL_REMAP.items():
        y_data = np.where(y_data == src, dst, y_data)
    return y_data


# ------------------------------------------------------------------
# Single-sample I/O (called via tf.numpy_function)
# ------------------------------------------------------------------


def read_npy_file(
    x_path: bytes,
    y_path: bytes,
    train: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load, resize, and optionally augment one OCT / segmentation pair.

    Reads .npy files, resizes to (IMG_HEIGHT, IMG_WIDTH), remaps labels
    from 11 classes to 4, one-hot encodes the target, and optionally
    applies Gaussian noise augmentation.

    Args:
        x_path: Byte-string path to the OCT image .npy file.
        y_path: Byte-string path to the segmentation label .npy file.
        train: If True, applies Gaussian noise augmentation to the image.

    Returns:
        Tuple of (image, label) tensors, both dtype float32.
            image shape: (IMG_HEIGHT, IMG_WIDTH, 1)
            label shape: (IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
    """
    x_data = np.load(x_path.decode())
    y_data = np.load(y_path.decode())

    x_data = cv2.resize(x_data, (IMG_WIDTH, IMG_HEIGHT))
    y_data = cv2.resize(y_data, (IMG_WIDTH, IMG_HEIGHT)).astype(np.intc)

    y_data = remap_labels(y_data)
    target = tf.one_hot(y_data, NUM_CLASSES)

    x_data = tf.cast(tf.expand_dims(x_data, axis=-1), tf.float32)
    target = tf.cast(target, tf.float32)

    if train:
        x_data = GaussianNoise(stddev=5)(x_data, training=True)

    return x_data, target


# ------------------------------------------------------------------
# Dataset construction
# ------------------------------------------------------------------


def _make_tf_dataset(
    x_paths: list[str],
    y_paths: list[str],
    train: bool,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Wrap path lists in a batched, optionally shuffled tf.data.Dataset.

    Args:
        x_paths: Ordered list of OCT image .npy file paths.
        y_paths: Corresponding segmentation .npy file paths.
        train: Passed to read_npy_file to control augmentation.
        batch_size: Number of samples per batch.
        shuffle: If True, shuffle before batching.

    Returns:
        Batched tf.data.Dataset yielding (image, label) pairs.
    """
    dataset = tf.data.Dataset.from_tensor_slices((x_paths, y_paths))
    dataset = dataset.map(
        lambda x, y: tuple(
            tf.numpy_function(
                read_npy_file, [x, y, train], [tf.float32, tf.float32]
            )
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_val_test_split(
    x_y_map: Dict[str, str],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    path_sep: str = "/",
    patient_name_parts: int = 2,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Split an image→label dict into train/val/test sets by patient.

    Splits at the patient level so no patient's scans appear in more than
    one split, preventing data leakage.

    Args:
        x_y_map: Dict mapping OCT image paths to segmentation label paths.
        val_ratio: Fraction of unique patients reserved for validation.
        test_ratio: Fraction of unique patients reserved for testing.
        path_sep: Path separator used to extract the filename component.
        patient_name_parts: Number of underscore-delimited filename parts
            that together identify a patient.

    Returns:
        Tuple of (train_map, val_map, test_map) dicts with the same
        key/value structure as x_y_map.
    """
    patients = sorted({
        "_".join(p.split(path_sep)[-1].split("_")[:patient_name_parts])
        for p in x_y_map
    })

    n = len(patients)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_set = set(patients[:n_test])
    val_set = set(patients[n_test: n_test + n_val])

    train_map: Dict[str, str] = {}
    val_map: Dict[str, str] = {}
    test_map: Dict[str, str] = {}

    for x_path, y_path in x_y_map.items():
        patient = "_".join(
            x_path.split(path_sep)[-1].split("_")[:patient_name_parts]
        )
        if patient in test_set:
            test_map[x_path] = y_path
        elif patient in val_set:
            val_map[x_path] = y_path
        else:
            train_map[x_path] = y_path

    return train_map, val_map, test_map


def build_datasets(
    x_y_map: Dict[str, str],
    batch_size: int = 8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create train/val/test tf.data.Datasets from an image→label path dict.

    Args:
        x_y_map: Dict mapping OCT image .npy paths to segmentation .npy paths.
        batch_size: Number of samples per batch.
        val_ratio: Fraction of patients reserved for validation.
        test_ratio: Fraction of patients reserved for testing.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_map, val_map, test_map = train_val_test_split(
        x_y_map, val_ratio=val_ratio, test_ratio=test_ratio
    )
    train_ds = _make_tf_dataset(
        list(train_map.keys()), list(train_map.values()),
        train=True, batch_size=batch_size, shuffle=True,
    )
    val_ds = _make_tf_dataset(
        list(val_map.keys()), list(val_map.values()),
        train=False, batch_size=batch_size, shuffle=False,
    )
    test_ds = _make_tf_dataset(
        list(test_map.keys()), list(test_map.values()),
        train=False, batch_size=batch_size, shuffle=False,
    )
    return train_ds, val_ds, test_ds


def datasets_from_maps(
    train_map: Dict[str, str],
    val_map: Dict[str, str],
    test_map: Dict[str, str],
    batch_size: int = 8,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create train/val/test tf.data.Datasets from pre-split path dicts.

    Use this when you have already partitioned paths into train/val/test
    (e.g. after custom subset selection in a notebook) and just need the
    tf.data pipeline.

    Args:
        train_map: Dict mapping training OCT image paths to label paths.
        val_map: Dict mapping validation OCT image paths to label paths.
        test_map: Dict mapping test OCT image paths to label paths.
        batch_size: Number of samples per batch.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_ds = _make_tf_dataset(
        list(train_map.keys()), list(train_map.values()),
        train=True, batch_size=batch_size, shuffle=True,
    )
    val_ds = _make_tf_dataset(
        list(val_map.keys()), list(val_map.values()),
        train=False, batch_size=batch_size, shuffle=False,
    )
    test_ds = _make_tf_dataset(
        list(test_map.keys()), list(test_map.values()),
        train=False, batch_size=batch_size, shuffle=False,
    )
    return train_ds, val_ds, test_ds


# ------------------------------------------------------------------
# Class balance analysis
# ------------------------------------------------------------------


def get_class_distribution(
    x_y_map: Dict[str, str],
) -> Tuple[float, float, float, float]:
    """Calculate the mean pixel fraction per segmentation class across a dataset.

    Loads each label file, remaps to 4 classes, and averages the pixel
    fractions across all files.  Useful for determining class weights.

    Args:
        x_y_map: Dict mapping OCT image paths to segmentation label paths.

    Returns:
        Tuple of four floats (class_0, class_1, class_2, class_3) where
        each value is the mean fraction of pixels belonging to that class.
    """
    class_fracs: list[list[float]] = [[], [], [], []]

    for y_path in x_y_map.values():
        y_data = remap_labels(np.load(y_path))
        total = y_data.size
        for cls in range(NUM_CLASSES):
            class_fracs[cls].append(float((y_data == cls).sum() / total))

    return tuple(float(np.mean(fracs)) for fracs in class_fracs)  # type: ignore[return-value]
