"""DICOM and XML segmentation preprocessing utilities.

Converts raw Heidelberg Spectralis DICOM volumes and OCTExplorer XML
segmentations into per-slice .npy files for downstream model training.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from bs4 import BeautifulSoup
from pydicom import dcmread
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# DICOM → NumPy conversion
# ------------------------------------------------------------------


def transform_dcm_to_npy(paths: List[str]) -> None:
    """Convert DICOM OCT volumes (z, x, y) into individual per-slice .npy files.

    Each 3D DICOM volume is flipped along the z-axis to match OCTExplorer's
    coordinate convention, then split into z individual (x, y) arrays saved
    as <original_stem>_<z_index>.npy.

    Args:
        paths: File paths to .dcm OCT volumes.
    """
    for path in paths:
        pydicom_image = dcmread(path)
        # Flip z-axis to align with OCTExplorer segmentation orientation
        oct_volume = np.flip(np.array(pydicom_image.pixel_array), axis=0)
        stem = path.split(".dcm")[0]
        for i in range(oct_volume.shape[0]):
            np.save(f"{stem}_{i}.npy", oct_volume[i])


# ------------------------------------------------------------------
# XML segmentation → NumPy conversion
# ------------------------------------------------------------------


def transform_segmentation_to_npy(paths_y: List[str]) -> None:
    """Convert OCTExplorer XML segmentations into per-slice .npy label files.

    Parses the OCTExplorer surface format (one <surface> element per layer),
    fills a 3D integer label volume using cumulative y-boundary indices, then
    saves each B-scan as <original_stem>_<z_index>.npy.

    Args:
        paths_y: File paths to .xml segmentation files exported from
            OCTExplorer.
    """
    for path_y in paths_y:
        with open(path_y, "r") as f:
            bs_data = BeautifulSoup(f.read(), "html.parser")

        size_tag = bs_data.find_all("scan_characteristics")[0]
        x_size = int(size_tag.find("x").getText())
        y_size = int(size_tag.find("y").getText())
        z_size = int(size_tag.find("z").getText())

        segmentation = np.zeros([z_size, y_size, x_size])
        idx_y_previous = np.zeros([z_size, x_size])

        for idx_label, layer in enumerate(bs_data.find_all("surface")):
            for idx_z, b_scan in enumerate(layer.find_all("bscan")):
                for idx_x, idx_y_tag in enumerate(b_scan.find_all("y")):
                    idx_y_val = int(idx_y_tag.getText())
                    segmentation[
                        idx_z,
                        int(idx_y_previous[idx_z, idx_x]):idx_y_val,
                        idx_x,
                    ] = idx_label
                    idx_y_previous[idx_z, idx_x] = idx_y_val

        stem = path_y.split(".xml")[0]
        for z in range(segmentation.shape[0]):
            np.save(f"{stem}_{z}.npy", segmentation[z])


# ------------------------------------------------------------------
# DICOM metadata extraction
# ------------------------------------------------------------------


def _try_open_dcm(
    pydicom_image,
    sitk_reader: sitk.ImageFileReader,
    dc_path: str,
) -> Tuple[bool, bool]:
    """Attempt to read a DICOM file via both pydicom and SimpleITK.

    Args:
        pydicom_image: pydicom Dataset already loaded with dcmread.
        sitk_reader: Configured SimpleITK ImageFileReader.
        dc_path: Path to the DICOM file (used by SimpleITK ReadImage).

    Returns:
        Tuple (pydicom_ok, sitk_ok) indicating which library succeeded.
    """
    try:
        _ = np.array(pydicom_image.pixel_array)
        pydicom_ok = True
    except Exception as exc:
        logger.warning("pydicom failed on %s: %s", dc_path, exc)
        pydicom_ok = False

    try:
        sitk_reader.ReadImageInformation()
        _ = sitk.ReadImage(str(dc_path))
        sitk_ok = True
    except Exception as exc:
        logger.warning("SimpleITK failed on %s: %s", dc_path, exc)
        sitk_ok = False

    return pydicom_ok, sitk_ok


def get_all_dcm_meta(dc_paths: List[str]) -> pd.DataFrame:
    """Extract metadata for a collection of DICOM files into a DataFrame.

    For each file reads all non-pixel DICOM tags, tests readability via
    pydicom and SimpleITK, and records volume dimensions when available.

    Args:
        dc_paths: File paths to .dcm files.

    Returns:
        DataFrame with one row per file.  Columns include all non-pixel DICOM
        keywords plus derived fields: patient_id, doctor_id, num_slices,
        img_size, pydicom_readable, sitk_readable, oct_readable, filename.
    """
    rows = []
    for dc_path in tqdm(dc_paths, desc="Reading DICOM metadata"):
        sitk_reader = sitk.ImageFileReader()
        sitk_reader.SetImageIO("GDCMImageIO")
        sitk_reader.SetFileName(str(dc_path))

        pydicom_reader = dcmread(str(dc_path))
        row: Dict = {
            el.keyword: getattr(pydicom_reader, el.keyword)
            for el in pydicom_reader
            if el.keyword not in ("", "PixelData")
        }

        patient_name = "".join(str(pydicom_reader.PatientName))
        parts = patient_name.split("^")
        row["patient_id"] = parts[0] if parts else ""
        row["doctor_id"] = parts[1] if len(parts) > 1 else ""

        pydicom_ok, sitk_ok = _try_open_dcm(pydicom_reader, sitk_reader, dc_path)
        row["pydicom_readable"] = pydicom_ok
        row["sitk_readable"] = sitk_ok
        row["oct_readable"] = pydicom_ok or sitk_ok

        if pydicom_ok or sitk_ok:
            sitk_reader.ReadImageInformation()
            image = sitk.ReadImage(str(dc_path))
            row["num_slices"] = image.GetDepth()
            row["img_size"] = image.GetSize()

        row["filename"] = Path(dc_path).name
        rows.append(row)

    return pd.DataFrame.from_records(rows)


# ------------------------------------------------------------------
# File anonymisation
# ------------------------------------------------------------------


def anonymize_file_paths(
    x_y_map: Dict[str, str],
    sep: str = "/",
    rename_files: bool = False,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Replace patient name tokens in file paths with integer identifiers.

    Extracts the first four underscore-delimited parts of each filename as
    the patient identifier, then substitutes them with a sequential integer.

    Args:
        x_y_map: Dict mapping OCT image paths to segmentation paths.
        sep: Path separator used to extract the filename component.
        rename_files: If True, physically rename files on disk as well.

    Returns:
        Tuple of (updated_map, replace_dict).
            updated_map: x_y_map with anonymised path strings.
            replace_dict: Mapping from original name strings to their
                integer replacements.
    """
    names = sorted({
        "_".join(p.split(sep)[-1].split("_")[:4])
        for p in x_y_map
    })
    replace_dict: Dict[str, int] = {name: idx for idx, name in enumerate(names)}

    updated_map: Dict[str, str] = {}
    for x_path, y_path in x_y_map.items():
        new_x, new_y = x_path, y_path
        for name, idx in replace_dict.items():
            new_x = new_x.replace(name, str(idx))
            new_y = new_y.replace(name, str(idx))
        updated_map[new_x] = new_y
        if rename_files:
            os.rename(x_path, new_x)
            os.rename(y_path, new_y)

    return updated_map, replace_dict
