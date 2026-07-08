"""Fits a pixel-position -> data-value mapping from a set of OCR'd tick
labels, using least-squares linear regression (or log10-space regression for
log-scaled axes). Ticks whose OCR text didn't parse as a number are dropped
before fitting."""
from dataclasses import dataclass

import numpy as np

from app.pipeline.ocr import TickLabel


@dataclass
class AxisCalibration:
    slope: float
    intercept: float
    log_scale: bool

    def pixel_to_value(self, pixel: float) -> float:
        raw = self.slope * pixel + self.intercept
        return 10 ** raw if self.log_scale else raw


def fit_axis_calibration(labels: list[TickLabel], log_scale: bool) -> AxisCalibration:
    readable = [label for label in labels if label.value is not None]
    if len(readable) < 2:
        raise ValueError(f"Need at least 2 readable tick labels to calibrate, got {len(readable)}")

    pixels = np.array([label.pixel_position for label in readable], dtype=float)
    values = np.array([label.value for label in readable], dtype=float)

    if log_scale:
        if np.any(values <= 0):
            raise ValueError("Log-scale axis cannot have zero or negative tick values")
        values = np.log10(values)

    slope, intercept = np.polyfit(pixels, values, deg=1)
    return AxisCalibration(slope=float(slope), intercept=float(intercept), log_scale=log_scale)
