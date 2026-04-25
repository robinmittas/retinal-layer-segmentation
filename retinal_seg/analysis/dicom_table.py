"""DICOM metadata extraction for Heidelberg Spectralis OCT files."""
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
from pydicom import dcmread

logger = logging.getLogger(__name__)


class DicomTable:
    """Parse and expose structured metadata from a Heidelberg Spectralis DICOM.

    Extracts patient demographics and per-frame OCT acquisition parameters
    (image positions, pixel spacing, frame boundary coordinates) into a
    tidy DataFrame accessible via record_lookup.

    Attributes:
        dicom_path: Path to the source .dcm file.
        record_lookup: DataFrame with one row per frame containing patient
            and OCT spatial metadata (patient_id, laterality, study_date,
            image_positions, stack_positions, x/y/z_scales, start/end coords).
        record_id: Unique record string "<patient_id>_<laterality>_<date>".
    """

    def __init__(self, dicom_path: str) -> None:
        """Parse the DICOM file and build the metadata lookup table.

        Args:
            dicom_path: Path to the .dcm file.
        """
        self.dicom_path = dicom_path
        self._dicom_file = dcmread(dicom_path)
        self.record_lookup: Optional[pd.DataFrame] = self._build_record_lookup()
        self.record_id: str = self._build_record_id()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_spectralis_macular(self) -> bool:
        """Check whether the DICOM is a 49-frame Heidelberg macular OCT volume.

        Returns:
            True if manufacturer, study description, series description, and
            B-scan count all match the expected Spectralis macular format.
        """
        try:
            return (
                self._dicom_file.Manufacturer == "Heidelberg Engineering"
                and self._dicom_file.StudyDescription == "Makula (OCT)"
                and self._dicom_file.SeriesDescription == "Volume IR"
                and self._dicom_file.pixel_array.shape[0] == 49
            )
        except Exception as exc:
            logger.warning("Could not verify DICOM format for %s: %s", self.dicom_path, exc)
            return False

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_record_id(self) -> str:
        if self.record_lookup is None:
            return "failed_to_fetch"
        try:
            pid = self.record_lookup.patient_id[0]
            lat = self.record_lookup.laterality[0]
            date = self.record_lookup.study_date[0]
            return f"{pid}_{lat}_{date}"
        except Exception:
            return "failed_to_fetch"

    def _get_oct_frame_data(self) -> dict:
        """Extract per-frame spatial metadata from PerFrameFunctionalGroupsSequence.

        Returns:
            Dict with lists for image_positions, stack_positions, x/y/z_scales,
            and y/x start/end coordinates.
        """
        frames = self._dicom_file.PerFrameFunctionalGroupsSequence
        pixel_spacing = (
            self._dicom_file.SharedFunctionalGroupsSequence[0]
            .PixelMeasuresSequence[0]
        )

        data: dict[str, list] = {k: [] for k in (
            "image_positions", "stack_positions",
            "x_scales", "y_scales", "z_scales",
            "y_starts", "x_starts", "y_ends", "x_ends",
        )}

        for frame in frames:
            data["image_positions"].append(
                frame.PlanePositionSequence[0].ImagePositionPatient
            )
            data["stack_positions"].append(
                frame.FrameContentSequence[0].InStackPositionNumber
            )
            data["x_scales"].append(float(pixel_spacing.PixelSpacing[1]))
            data["y_scales"].append(float(pixel_spacing.PixelSpacing[0]))
            data["z_scales"].append(float(pixel_spacing.SliceThickness))

            coords = frame.OphthalmicFrameLocationSequence[0].ReferenceCoordinates
            data["y_starts"].append(coords[0])
            data["x_starts"].append(coords[1])
            data["y_ends"].append(coords[2])
            data["x_ends"].append(coords[3])

        return data

    def _build_record_lookup(self) -> Optional[pd.DataFrame]:
        if self._dicom_file is None:
            return None

        patient = {
            "patient_id": re.findall(r"\d+", self._dicom_file.PatientID),
            "laterality": self._dicom_file.ImageLaterality,
            "study_date": self._dicom_file.StudyDate,
        }

        oct_data = self._get_oct_frame_data()
        return pd.concat(
            (pd.DataFrame(patient), pd.DataFrame(oct_data)), axis=1
        )
