"""Isolates and traces individual series' pixels from the plot area, and
extracts (x, y) data points for line/step, scatter, and bar chart types."""
import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration
from app.pipeline.geometry import PlotBox


def isolate_series_mask(
    image: np.ndarray, box: PlotBox, target_color_bgr: tuple[int, int, int], tolerance: int = 40
) -> np.ndarray:
    region = image[box.top:box.bottom, box.left:box.right]
    target = np.array(target_color_bgr, dtype=np.int16)
    diff = np.abs(region.astype(np.int16) - target).sum(axis=2)
    mask = np.where(diff <= tolerance, 255, 0).astype(np.uint8)
    return mask
