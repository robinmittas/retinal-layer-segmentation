"""Central configuration: training defaults, image dimensions, and label constants."""
from typing import Dict, Tuple

# ------------------------------------------------------------------
# Image dimensions (Heidelberg Spectralis default output size)
# ------------------------------------------------------------------
IMG_HEIGHT: int = 496
IMG_WIDTH: int = 512
IMG_CHANNELS: int = 1
NUM_CLASSES: int = 4

# ------------------------------------------------------------------
# Training defaults
# ------------------------------------------------------------------
BATCH_SIZE: int = 8
EPOCHS: int = 30
SHUFFLE_BUFFER_SIZE: int = 100
LEARNING_RATE: float = 1e-3

# ------------------------------------------------------------------
# Label remapping: 11-class OCTExplorer labels → 4 segmentation classes
# 0: background
# 1: ILM–OPL–HFL  (inner retinal layers, classes 2–5 collapsed)
# 2: ONL           (Outer Nuclear Layer, class 6)
# 3: BMEIS–OB–RPE  (outer retinal layers, classes 7–11 collapsed)
# ------------------------------------------------------------------
LABEL_REMAP: Dict[int, int] = {
    2: 1, 3: 1, 4: 1, 5: 1,
    6: 2,
    7: 3, 8: 3, 9: 3, 10: 3, 11: 3,
}

LABEL_NAMES: Dict[int, str] = {
    0: "background",
    1: "ilm_opl_hfl",
    2: "onl",
    3: "bmeis_ob_rpe",
}

# Class weights for weighted cross-entropy; ONL (class 2) is upweighted
# to counteract severe class imbalance in the training set.
CLASS_WEIGHTS: Tuple[float, ...] = (1.0, 1.0, 3.5, 1.0)

# Smoothing constant shared by all Dice-based metrics and losses
DICE_SMOOTH: float = 0.001

# ------------------------------------------------------------------
# DICOM / pixel-space defaults (Heidelberg Spectralis fallback values)
# ------------------------------------------------------------------
B_SCAN_RESIZE_FACTOR: int = 2
DEFAULT_Y_RESOLUTION: float = 0.003872   # mm per pixel (depth axis)
DEFAULT_X_RESOLUTION: float = 0.012034   # mm per pixel (lateral axis)
DEFAULT_SLICE_THICKNESS: float = 0.128197  # mm
