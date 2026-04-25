"""Retinal thickness map computation and visualisation.

Consolidates the two previous Map implementations (utils/map.py and
utils/thickness_map.py) into a single ThicknessMap class, and exposes
standalone volume-level analysis helpers used by the ETDRS pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# Labels excluded from tissue thickness measurement (artefacts, choroid etc.)
_THICKNESS_EXCLUSION = frozenset([0, 1, 9, 10, 11, 12, 14, 15])


class ThicknessMap:
    """Compute and visualise a 2D retinal thickness map from OCT segmentations.

    Given a DicomTable (providing pixel-space scan coordinates) and a list of
    per-frame segmentation masks, computes a µm-scale 2D thickness map on a
    fundus grid by aggregating per-column 1-D depth vectors.

    Attributes:
        dicom: DicomTable instance with per-frame coordinate metadata.
        segmentations: List of (H, W) integer/boolean arrays, one per B-scan.
        dicom_path: Path to the source .dcm file (used for logging).
        thickness_map: Computed 2D thickness array after calling depth_grid(),
            or None before that call.
    """

    def __init__(
        self,
        dicom,
        segmentations: list[np.ndarray],
        dicom_path: str,
    ) -> None:
        """Initialise with DICOM metadata and per-B-scan segmentation masks.

        Args:
            dicom: DicomTable instance for this recording.
            segmentations: List of (H, W) integer/boolean arrays.
            dicom_path: Path to the corresponding .dcm file.
        """
        self.dicom = dicom
        self.segmentations = segmentations
        self.dicom_path = dicom_path
        self.thickness_map: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def depth_grid(
        self,
        grid_dim: Tuple[int, int] = (768, 768),
        interpolation: str = "linear",
    ) -> None:
        """Compute the 2D fundus thickness map and store it in self.thickness_map.

        Iterates over all B-scans, converts per-column pixel depths to µm,
        places them on a zero-initialised fundus grid at their DICOM-derived
        scan positions, then fills gaps with pandas interpolation.

        Args:
            grid_dim: Output grid dimensions (rows, cols).  Default 768×768
                matches the Heidelberg fundus image.
            interpolation: Interpolation method passed to
                pd.DataFrame.interpolate() for gap-filling between B-scans.
        """
        y_cord, x_cord = self._get_iterable_dimension()
        grid = np.zeros(grid_dim)
        startx_pos, endx_pos, starty_pos, endy_pos = self._get_position_series()

        for i, seg in enumerate(self.segmentations):
            d_v = self._get_depth_vector(seg)
            d_v = self._oct_pixel_to_um(d_v, i, x_cord, y_cord)
            try:
                if y_cord == "iterable":
                    x_start, x_end = int(startx_pos[i]), int(endx_pos[i])
                    y_start = int(starty_pos[i])
                    d_v = _pad_or_trim(d_v, x_end - x_start)
                    grid[y_start, x_start:x_end] = d_v
                else:
                    y_start, y_end = int(starty_pos[i]), int(endy_pos[i])
                    x_start = int(startx_pos[i])
                    d_v = _pad_or_trim(d_v, abs(y_start - y_end))
                    grid[y_end:y_start, x_start] = d_v
            except Exception as exc:
                logger.warning(
                    "Could not place depth vector for frame %d (%s): %s",
                    i, self.dicom_path, exc,
                )

        grid[grid == 0] = np.nan
        axis = 0 if y_cord == "iterable" else 1
        grid_df = pd.DataFrame(grid).interpolate(
            limit_direction="both", axis=axis, method=interpolation
        )

        if y_cord == "iterable":
            min_y = int(min(starty_pos.iloc[1:]))
            max_y = int(max(starty_pos.iloc[1:]))
            grid_df.iloc[:min_y] = 0
            grid_df.iloc[max_y:] = 0
        else:
            min_x = int(min(startx_pos.iloc[1:]))
            max_x = int(max(startx_pos.iloc[1:]))
            grid_df.iloc[:, :min_x] = 0
            grid_df.iloc[:, max_x:] = 0

        self.thickness_map = np.array(grid_df.fillna(0))

    def plot_thickness_map(self, save_path: str) -> None:
        """Render and save the thickness map using the Heidelberg colormap.

        Args:
            save_path: Destination file path for the figure (e.g. 'map.png').

        Raises:
            ValueError: If depth_grid() has not been called yet.
        """
        if self.thickness_map is None:
            raise ValueError("Call depth_grid() before plot_thickness_map().")

        cm = _heidelberg_colormap()
        cm.set_bad(color="black")
        cropped = _crop_image(self.thickness_map)
        masked = np.ma.masked_where(cropped < 100, cropped)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(masked, cmap=cm, vmin=100, vmax=750)
        ax.axis("off")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_iterable_dimension(self) -> Tuple[Optional[str], Optional[str]]:
        y_unique = np.unique(self.dicom.record_lookup.y_starts).shape[0]
        x_unique = np.unique(self.dicom.record_lookup.x_starts).shape[0]
        if y_unique > x_unique:
            return "iterable", None
        return None, "iterable"

    def _oct_pixel_to_um(
        self,
        depth_vector: np.ndarray,
        frame_idx: int,
        x_cord: Optional[str],
        y_cord: Optional[str],
    ) -> np.ndarray:
        """Convert a pixel-unit depth vector to micrometres using DICOM scaling.

        Args:
            depth_vector: 1-D pixel-unit thickness values.
            frame_idx: Index into the DicomTable record_lookup DataFrame.
            x_cord: 'iterable' when the scan axis is x; otherwise None.
            y_cord: 'iterable' when the scan axis is y; otherwise None.

        Returns:
            Depth vector scaled to µm.
        """
        if y_cord == "iterable":
            scale = float(self.dicom.record_lookup.y_scales.iloc[frame_idx])
        else:
            scale = float(self.dicom.record_lookup.x_scales.iloc[frame_idx])
        return depth_vector * scale * 1000.0

    def _get_position_series(self):
        lk = self.dicom.record_lookup
        return (
            lk.x_starts.reset_index(drop=True).fillna(0),
            lk.x_ends.reset_index(drop=True).fillna(0),
            lk.y_starts.reset_index(drop=True).fillna(0),
            lk.y_ends.reset_index(drop=True).fillna(0),
        )

    @staticmethod
    def _get_depth_vector(img: np.ndarray) -> np.ndarray:
        """Compute per-column retinal thickness from a segmentation mask.

        Measures the pixel distance between the topmost and bottommost
        non-zero pixel in each column.  Zero-valued gaps (missing or
        occluded columns) are filled by averaging their nearest non-zero
        neighbours.

        Args:
            img: 2D segmentation array (H, W).

        Returns:
            1-D depth vector of length W in pixels.
        """
        depth = np.zeros(img.shape[1])
        for col in range(img.shape[1]):
            nz = np.argwhere(img[:, col]).ravel()
            if nz.size:
                depth[col] = int(nz.max()) - int(nz.min())

        idx_nonzero = np.argwhere(depth).ravel()
        if idx_nonzero.size == 0:
            return depth

        idx_zero = np.where(depth == 0)[0]
        for patch in _split_contiguous(idx_zero):
            left = idx_nonzero[np.abs(idx_nonzero - patch[0]).argmin()]
            right = idx_nonzero[np.abs(idx_nonzero - patch[-1]).argmin()]
            depth[patch] = (depth[left] + depth[right]) / 2.0

        return depth


# ------------------------------------------------------------------
# Standalone volume-level analysis helpers
# ------------------------------------------------------------------


def thickness_profiler(volume: np.ndarray) -> np.ndarray:
    """Compute per-B-scan, per-A-scan tissue thickness from a label volume.

    Zeroes out excluded label classes (background, structural artefacts,
    choroid) and binarises remaining tissue, then measures the span of
    non-zero pixels per column in each B-scan.

    Args:
        volume: Integer label volume of shape (scans, height, width).

    Returns:
        Thickness array of shape (scans, width, 1) in pixels.
    """
    vol = volume.copy()
    exclusion = list(_THICKNESS_EXCLUSION)
    vol[np.isin(vol, exclusion)] = 0
    vol[vol != 0] = 1

    n_scans, _, width = vol.shape
    thickness = np.zeros((n_scans, width, 1))

    for i in range(n_scans):
        b_scan = vol[i]
        for j in range(b_scan.shape[1]):
            col = np.argwhere(b_scan[:, j]).ravel()
            if col.size:
                thickness[i, j, 0] = int(col.max()) - int(col.min())

    return thickness


def thickness_spatial_grid(
    thickness_map: np.ndarray,
    output_shape: Tuple[int, int] = (496, 512),
    n_bscans: int = 49,
    interpolate: bool = False,
) -> np.ndarray:
    """Project per-B-scan thickness vectors onto a 2D spatial grid.

    Places n_bscans uniformly onto the row axis of a zero-initialised
    (output_shape) grid.  Gaps between sampled rows can be filled via
    column-wise linear interpolation.

    Args:
        thickness_map: Array of shape (n_scans, width, 1) from
            thickness_profiler().
        output_shape: Desired output grid (rows, cols).
        n_bscans: Number of B-scans to distribute across the row axis.
        interpolate: If True, fill empty rows by 1-D linear interpolation
            along each column.

    Returns:
        2D thickness grid of shape output_shape.
    """
    grid = np.zeros(output_shape)
    b_locs = np.round(np.linspace(0, output_shape[0] - 1, n_bscans)).astype(int)

    for k, loc in enumerate(b_locs):
        grid[loc, :] = thickness_map[k, :, 0]

    if interpolate:
        for col in range(output_shape[1]):
            col_vals = grid[:, col].copy()
            col_vals[col_vals == 0] = np.nan
            nan_mask = np.isnan(col_vals)
            if nan_mask.any() and not nan_mask.all():
                x = np.arange(len(col_vals))
                grid[:, col] = np.interp(x, x[~nan_mask], col_vals[~nan_mask])

    return grid


# ------------------------------------------------------------------
# Private module-level utilities
# ------------------------------------------------------------------


def _split_contiguous(indices: np.ndarray) -> list[np.ndarray]:
    """Split a sorted index array into contiguous sub-arrays (patches).

    Args:
        indices: Sorted 1-D integer array.

    Returns:
        List of contiguous sub-arrays; empty list if indices is empty.
    """
    if indices.size == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return np.split(indices, splits)


def _pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resize a 1-D array to exactly target_len by trimming or zero-padding.

    Args:
        arr: Input array.
        target_len: Desired output length.

    Returns:
        Array of length target_len.
    """
    if arr.shape[0] > target_len:
        return arr[:target_len]
    if arr.shape[0] < target_len:
        return np.append(arr, np.zeros(target_len - arr.shape[0]))
    return arr


def _crop_image(img: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Crop zero-valued border rows and columns from an image array.

    Args:
        img: 2D or 3D numpy array.
        tol: All border rows/columns whose max value ≤ tol are removed.

    Returns:
        Cropped array with the same number of dimensions as img.
    """
    mask = img[:, :, 0] > tol if img.ndim > 2 else img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def _heidelberg_colormap() -> LinearSegmentedColormap:
    """Build the Heidelberg Heyex-style matplotlib colormap.

    Reproduces the clinical display colormap used in Heidelberg's Heyex
    software: black → purple → blue → green → yellow → red → white.

    Returns:
        LinearSegmentedColormap instance named 'heidelberg'.
    """
    cdict = {
        "blue": [
            (0.0, 0.0, 0.0), (0.1, 1.0, 1.0), (0.2, 1.0, 1.0),
            (0.3, 0.0, 0.0), (0.4, 0.0, 0.0), (0.55, 0.0, 0.0),
            (0.65, 1.0, 1.0), (1.0, 1.0, 1.0),
        ],
        "green": [
            (0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0),
            (0.3, 1.0, 1.0), (0.4, 1.0, 1.0), (0.55, 0.0, 0.0),
            (0.65, 1.0, 1.0), (1.0, 1.0, 1.0),
        ],
        "red": [
            (0.0, 0.0, 0.0), (0.1, 1.0, 1.0), (0.2, 0.0, 0.0),
            (0.3, 0.0, 0.0), (0.4, 1.0, 1.0), (0.55, 1.0, 1.0),
            (0.65, 1.0, 1.0), (1.0, 1.0, 1.0),
        ],
    }
    return LinearSegmentedColormap("heidelberg", cdict)
