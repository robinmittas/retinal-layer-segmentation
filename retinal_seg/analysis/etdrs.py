"""ETDRS (Early Treatment Diabetic Retinopathy Study) grid analysis utilities.

Provides zone-based thickness and tissue statistics aligned to the standard
9-region ETDRS grid: central foveal disc (C0), inner ring (S1/N1/I1/T1),
and outer ring (S2/N2/I2/T2).

Typical usage:
    utils = ETDRSUtils("path/to/volume.npy")
    stats = utils.get_etdrs_stats()
    df = pd.DataFrame([stats])
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np
from pydicom import dcmread

from retinal_seg.analysis.thickness_map import (
    thickness_profiler,
    thickness_spatial_grid,
)
from retinal_seg.config import B_SCAN_RESIZE_FACTOR, DEFAULT_X_RESOLUTION, DEFAULT_Y_RESOLUTION, DEFAULT_SLICE_THICKNESS

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level constants
# ------------------------------------------------------------------

TISSUE_LABELS: Dict[int, str] = {
    1: "epiretinal_membrane",
    2: "neurosensory_retina",
    3: "intraretinal_fluid",
    4: "subretinal_fluid",
    5: "subretinal_hyperreflective_material",
    6: "rpe",
    7: "fibrovascular_ped",
    8: "drusen",
    9: "posterior_hyaloid_membrane",
    10: "choroid",
    13: "fibrosis",
}

ETDRS_REGIONS = ["C0", "S2", "S1", "N1", "N2", "I1", "I2", "T1", "T2"]


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------


def create_circular_mask(
    h: int,
    w: int,
    center: Optional[list[int]] = None,
    radius: Optional[float] = None,
) -> np.ndarray:
    """Create a boolean disc mask on an (h, w) grid.

    Args:
        h: Grid height in pixels.
        w: Grid width in pixels.
        center: [col, row] pixel coordinates of the disc centre.
            Defaults to the image centre.
        radius: Disc radius in pixels.  Defaults to the largest disc that
            fits within the image boundaries.

    Returns:
        Boolean array of shape (h, w); True inside the disc.
    """
    if center is None:
        center = [w // 2, h // 2]
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    return (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius ** 2


# ------------------------------------------------------------------
# RPE atrophy map
# ------------------------------------------------------------------


def rpe_profiler(volume: np.ndarray) -> np.ndarray:
    """Map geographic RPE atrophy locations across a segmentation volume.

    For each B-scan identifies A-scan columns where the RPE label (class 6)
    is absent within the neurosensory retina span (class 2), then
    interpolates atrophy between sampled B-scan positions.

    Args:
        volume: Integer label volume of shape (scans, height, width).

    Returns:
        Binary 2D map of shape (height, width); 1 = atrophy present.
    """
    n_scans = volume.shape[0]
    map_positions = np.linspace(0, 255, n_scans, dtype=np.int32)
    rpe_map = np.zeros((volume.shape[1], volume.shape[2]), dtype=np.int32)

    for i, pos in enumerate(map_positions):
        retina_cols = np.argwhere(np.sum(volume[i] == 2, axis=0))
        if not retina_cols.size:
            continue
        start, stop = int(retina_cols.min()), int(retina_cols.max())
        rpe_present = np.sum(volume[i] == 6, axis=0)[start:stop]
        rpe_map[pos, start:stop] = (rpe_present == 0).astype(np.int32)

    previous: Optional[int] = None
    for k, pos in enumerate(map_positions):
        if k > 0 and previous is not None:
            both_atrophy = (rpe_map[pos] == 1) & (rpe_map[previous] == 1)
            for col in np.where(both_atrophy)[0]:
                rpe_map[previous:pos, col] = 1
        previous = int(pos)

    return rpe_map


# ------------------------------------------------------------------
# DICOM pixel-space conversion
# ------------------------------------------------------------------


def get_pixel_to_volume_factors(dicom_path: str) -> tuple[float, float]:
    """Compute voxel-volume and pixel-height conversion factors from a DICOM.

    Falls back to default Spectralis calibration values if the DICOM
    header is unreadable.

    Args:
        dicom_path: Path to a .dcm file or a directory containing one.

    Returns:
        Tuple (pixel_to_volume_mm3, pixel_height_to_mm).
    """
    try:
        path = (
            os.path.join(dicom_path, os.listdir(dicom_path)[0])
            if os.path.isdir(dicom_path)
            else dicom_path
        )
        dc = dcmread(path, stop_before_pixels=True)
        shared = dc.SharedFunctionalGroupsSequence[0][("0028", "9110")][0]
        y_res = float(shared.PixelSpacing[0])
        x_res = float(shared.PixelSpacing[1])
        slice_t = float(shared.SliceThickness)
    except Exception as exc:
        logger.warning("DICOM resolution unavailable (%s); using defaults.", exc)
        y_res, x_res, slice_t = DEFAULT_Y_RESOLUTION, DEFAULT_X_RESOLUTION, DEFAULT_SLICE_THICKNESS

    return (
        B_SCAN_RESIZE_FACTOR * y_res * x_res * slice_t,
        B_SCAN_RESIZE_FACTOR * y_res,
    )


# ------------------------------------------------------------------
# Volume spatial grid
# ------------------------------------------------------------------


def volume_spatial_grid(volume: np.ndarray) -> np.ndarray:
    """Distribute B-scans uniformly into a cubic spatial container.

    Places n_scans B-scans at evenly spaced positions along the first
    axis of a (width, height, width) zero-initialised volume.

    Args:
        volume: Array of shape (scans, height, width).

    Returns:
        Array of shape (width, height, width).
    """
    n_scans, height, width = volume.shape
    container = np.zeros((width, height, width))
    b_locs = np.round(np.linspace(0, width - 1, n_scans)).astype(int)
    for k, loc in enumerate(b_locs):
        container[loc] = volume[k]
    return container


# ------------------------------------------------------------------
# Laterality formatting
# ------------------------------------------------------------------


def _format_laterality(laterality: str) -> str:
    """Expand single-character laterality codes to full words.

    Args:
        laterality: 'L', 'R', or an already-expanded string.

    Returns:
        'Left', 'Right', or the original string if not recognised.
    """
    return {"L": "Left", "R": "Right"}.get(laterality, laterality)


# ------------------------------------------------------------------
# ETDRSUtils class
# ------------------------------------------------------------------


class ETDRSUtils:
    """Compute ETDRS-grid zone statistics for a retinal segmentation volume.

    Partitions the macular area into 9 ETDRS regions and returns per-region
    mean thickness, atrophy percentage, and tissue pixel counts.

    Attributes:
        path: Path to the .npy segmentation volume.
        patient: Patient identifier extracted from the filename.
        study_date: Study date extracted from the filename.
        laterality: 'Left' or 'Right'.
        etdrs_bool_grid: Dict mapping region names to boolean 2D masks
            (populated after calling zones()).
    """

    def __init__(self, path: str) -> None:
        """Initialise from a segmentation volume path.

        The filename must follow the convention
        <patient>_<laterality>_<date>.npy.

        Args:
            path: Path to a .npy segmentation volume.
        """
        self.path = path
        self._record = os.path.basename(path)
        self.patient: str = self._record.split("_")[0]
        self.study_date: str = self._record.split("_")[2].replace(".npy", "")
        self.laterality: str = _format_laterality(self._record.split("_")[1])
        self.height: Optional[int] = None
        self.width: Optional[int] = None
        self.etdrs_bool_grid: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def get_etdrs_stats(self) -> Dict[str, object]:
        """Compute per-region thickness, atrophy, and tissue pixel counts.

        Loads the volume, builds the ETDRS boolean mask grid, and returns
        a flat dictionary suitable for appending to a pandas DataFrame.

        Returns:
            Dict with keys like 'C0_thickness_mean', 'C0_atrophy_percentage',
            'C0_neurosensory_retina', etc., plus a 'record' key.
        """
        volume = np.load(self.path)
        _, self.height, self.width = volume.shape

        atrophy_map = rpe_profiler(volume)
        volume_spatial = volume_spatial_grid(volume)
        thickness_map = thickness_profiler(np.copy(volume))
        thickness_spatial = thickness_spatial_grid(thickness_map)

        self.zones()

        record: Dict[str, object] = {"record": self._record.replace(".npy", "")}

        for region, bool_mask in self.etdrs_bool_grid.items():
            region_segment = volume_spatial[bool_mask]
            thickness_seg = thickness_spatial[bool_mask[0]]
            atrophy_seg = atrophy_map[bool_mask[0]]

            nonzero_t = thickness_seg[np.nonzero(thickness_seg)]
            record[f"{region}_thickness_mean"] = (
                float(np.mean(nonzero_t)) if nonzero_t.size else 0.0
            )
            record[f"{region}_atrophy_percentage"] = float(np.mean(atrophy_seg))

            region_total = 0
            for tissue_id, tissue_name in TISSUE_LABELS.items():
                count = int(np.sum(region_segment == tissue_id))
                record[f"{region}_{tissue_name}"] = count
                if tissue_id not in {9, 10}:
                    region_total += count
            record[f"{region}_total"] = region_total

        return record

    def zones(self) -> None:
        """Build the ETDRS boolean mask grid and store in self.etdrs_bool_grid.

        Raises:
            ValueError: If height/width have not been set (call get_etdrs_stats
                first or set them manually).
        """
        if self.height is None or self.width is None:
            raise ValueError("Load the volume first via get_etdrs_stats().")

        masks = {
            "upper_right": self._upper_right(),
            "lower_right": self._lower_right(),
            "upper_left":  self._upper_left(),
            "lower_left":  self._lower_left(),
            "middle":      self._middle_disk(),
            "outer":       self._outer_disk(),
        }

        roi_combos = [
            ("upper_left",  "outer",  "upper_right"),
            ("upper_left",  "middle", "upper_right"),
            ("lower_right", "middle", "upper_right"),
            ("lower_right", "outer",  "upper_right"),
            ("lower_right", "middle", "lower_left"),
            ("lower_right", "outer",  "lower_left"),
            ("upper_left",  "middle", "lower_left"),
            ("upper_left",  "outer",  "lower_left"),
        ]

        self.etdrs_bool_grid["C0"] = self._inner_disk()
        for k, (a, b, c) in enumerate(roi_combos, 1):
            self.etdrs_bool_grid[ETDRS_REGIONS[k]] = masks[a] & masks[b] & masks[c]

    # ------------------------------------------------------------------
    # Private geometry helpers
    # ------------------------------------------------------------------

    def _upper_right(self) -> np.ndarray:
        return np.arange(self.height)[:, None] <= np.arange(self.width)

    def _lower_right(self) -> np.ndarray:
        return self._upper_right()[::-1]

    def _upper_left(self) -> np.ndarray:
        return self._lower_left()[::-1]

    def _lower_left(self) -> np.ndarray:
        return np.arange(self.height)[:, None] > np.arange(self.width)

    def _inner_disk(self) -> np.ndarray:
        return create_circular_mask(self.height, self.width, radius=self.height // 12)

    def _middle_disk(self) -> np.ndarray:
        disk = create_circular_mask(self.height, self.width, radius=self.height // 4)
        disk[self._inner_disk()] = False
        return disk

    def _outer_disk(self) -> np.ndarray:
        disk = create_circular_mask(self.height, self.width, radius=self.height // 2)
        disk[self._inner_disk()] = False
        disk[self._middle_disk()] = False
        return disk
