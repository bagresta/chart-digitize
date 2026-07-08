"""Draws detected data points back onto a copy of the original chart image,
for the user to visually sanity-check auto-extraction accuracy."""
import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration
from app.pipeline.geometry import PlotBox

_MARKER_RADIUS = 4
_MARKER_COLOR_BGR = (0, 0, 0)  # black outline, drawn under a white fill for contrast on any series color
_MARKER_OUTLINE_COLOR_BGR = (255, 255, 255)


def _value_to_pixel(value: float, calibration: AxisCalibration) -> int:
    # invert calibration.pixel_to_value: value = slope*pixel + intercept (or slope*pixel+intercept in log space)
    raw_value = np.log10(value) if calibration.log_scale else value
    pixel = (raw_value - calibration.intercept) / calibration.slope
    return int(round(pixel))


def draw_overlay(
    image: np.ndarray,
    box: PlotBox,
    x_calibration: AxisCalibration,
    y_calibration: AxisCalibration,
    series: list[dict],
) -> np.ndarray:
    overlay = image.copy()

    for s in series:
        for x_val, y_val in s["points"]:
            px = _value_to_pixel(x_val, x_calibration)
            py = _value_to_pixel(y_val, y_calibration)
            cv2.circle(overlay, (px, py), _MARKER_RADIUS + 1, _MARKER_OUTLINE_COLOR_BGR, -1)
            cv2.circle(overlay, (px, py), _MARKER_RADIUS, s.get("color_bgr", _MARKER_COLOR_BGR), -1)

    return overlay
