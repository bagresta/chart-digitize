"""Fits a pixel-position -> data-value mapping from a set of OCR'd tick
labels, using least-squares linear regression (or log10-space regression for
log-scaled axes). Ticks whose OCR text didn't parse as a number are dropped
before fitting.

Single-outlier rejection
-------------------------
`np.polyfit` gives every tick equal weight, so a single bad value can
silently drag the whole calibration off. This is a real, observed failure
mode: manual browser verification of a Kaplan-Meier chart found Tesseract
misreading a "0.4" y-axis tick label as "2" (dropping the leading "0."
before the digit). That one corrupted point skewed the unweighted fit
enough to produce survival probabilities up to 170% off at other
timepoints, with no indication anything had gone wrong.

To guard against this, when there are 3 or more readable ticks we fit once
with all points, then check whether exactly one point's absolute residual
looks like a genuine misread rather than ordinary OCR/rendering jitter. We
require BOTH:
  1. The worst residual is much larger (>4x) than the median residual of
     the remaining points - i.e. it doesn't fit the pattern the other
     points establish.
  2. The worst residual is also large in absolute terms relative to the
     overall value range (>15% of it) - i.e. it's not merely a relatively
     large residual among otherwise tiny jitter, but an error big enough
     to matter.
Both conditions must hold before a point is dropped, and at most one point
is ever dropped. This keeps the heuristic conservative: normal sub-pixel
rendering jitter satisfies neither condition reliably (see
test_fit_calibration_does_not_reject_normal_jitter), while a genuinely
wrong tick value (like "0.4" misread as "2") satisfies both by a wide
margin (see test_fit_calibration_rejects_single_ocr_misread_outlier).
"""
from dataclasses import dataclass

import numpy as np

from app.pipeline.ocr import TickLabel

# A point's residual must be more than this many times the median residual
# of the other points to be considered structurally different from normal
# jitter.
_OUTLIER_RESIDUAL_RATIO = 4.0

# A point's residual must also exceed this fraction of the overall value
# range to be considered large enough to matter (guards against flagging a
# relatively-large-but-tiny residual amid otherwise minuscule jitter).
_OUTLIER_RANGE_FRACTION = 0.15


@dataclass
class AxisCalibration:
    slope: float
    intercept: float
    log_scale: bool

    def pixel_to_value(self, pixel: float) -> float:
        raw = self.slope * pixel + self.intercept
        return 10 ** raw if self.log_scale else raw


def _drop_single_outlier_index(pixels: np.ndarray, values: np.ndarray) -> int | None:
    """Return the index of a single clear outlier to drop, or None if the
    fit doesn't show one. See module docstring for the criteria."""
    if len(pixels) < 3:
        return None

    slope, intercept = np.polyfit(pixels, values, deg=1)
    residuals = np.abs(values - (slope * pixels + intercept))

    worst_index = int(np.argmax(residuals))
    worst_residual = residuals[worst_index]
    rest_residuals = np.delete(residuals, worst_index)
    median_rest = float(np.median(rest_residuals))
    value_range = float(values.max() - values.min())

    if median_rest <= 0 or value_range <= 0:
        return None

    ratio_condition = worst_residual > _OUTLIER_RESIDUAL_RATIO * median_rest
    magnitude_condition = worst_residual > _OUTLIER_RANGE_FRACTION * value_range

    if ratio_condition and magnitude_condition:
        return worst_index
    return None


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

    outlier_index = _drop_single_outlier_index(pixels, values)
    if outlier_index is not None:
        pixels = np.delete(pixels, outlier_index)
        values = np.delete(values, outlier_index)

    slope, intercept = np.polyfit(pixels, values, deg=1)
    return AxisCalibration(slope=float(slope), intercept=float(intercept), log_scale=log_scale)
