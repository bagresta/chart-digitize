"""Detects the plot bounding box and axis tick pixel positions using classic
edge/line detection. Charts are assumed to have a visible left/bottom axis
spine (true for matplotlib default style and the vast majority of published
figures)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlotBox:
    left: int
    top: int
    right: int
    bottom: int


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_plot_box(image: np.ndarray) -> PlotBox:
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)

    h, w = gray.shape
    min_line_length = min(h, w) * 0.3
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=min_line_length, maxLineGap=10,
    )
    if lines is None:
        raise ValueError("No axis lines detected in image")

    # cv2.HoughLinesP normally returns shape (N, 1, 4), but some opencv
    # builds return (N, 4) directly; reshape to a flat (N, 4) either way.
    lines = lines.reshape(-1, 4)

    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines:
        if abs(y1 - y2) < 3 and abs(x1 - x2) > min_line_length:
            horizontals.append((y1 + y2) / 2)
        elif abs(x1 - x2) < 3 and abs(y1 - y2) > min_line_length:
            verticals.append((x1 + x2) / 2)

    if not horizontals or not verticals:
        raise ValueError("Could not find both horizontal and vertical axis lines")

    # bottom axis = lowest horizontal line, left axis = leftmost vertical line,
    # top/right bounds taken from the extreme opposite lines found (handles
    # boxed plots) or fall back to image edges (handles open-spine plots).
    bottom = int(max(horizontals))
    top_candidates = [y for y in horizontals if y < bottom - 10]
    top = int(min(top_candidates)) if top_candidates else int(h * 0.05)

    left = int(min(verticals))
    right_candidates = [x for x in verticals if x > left + 10]
    right = int(max(right_candidates)) if right_candidates else int(w * 0.95)

    return PlotBox(left=left, top=top, right=right, bottom=bottom)


def detect_tick_positions(image: np.ndarray, box: PlotBox, axis: str) -> list[int]:
    """Finds tick mark pixel positions along the given axis by looking for
    short perpendicular strokes just outside the plot box spine."""
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)

    if axis == "x":
        # look in a thin strip just below the bottom spine
        strip = edges[box.bottom + 1: box.bottom + 8, box.left:box.right]
        column_sums = strip.sum(axis=0)
        positions_local = _find_peak_centers(column_sums)
        return [box.left + p for p in positions_local]
    elif axis == "y":
        # look in a thin strip just left of the left spine
        strip = edges[box.top:box.bottom, max(0, box.left - 8):box.left - 1]
        row_sums = strip.sum(axis=1)
        positions_local = _find_peak_centers(row_sums)
        return [box.top + p for p in positions_local]
    else:
        raise ValueError(f"Unknown axis: {axis}")


def _find_peak_centers(signal: np.ndarray, min_gap: int = 5) -> list[int]:
    """Groups consecutive nonzero indices in a 1D signal into clusters and
    returns each cluster's center — used to turn a strip of tick-mark pixels
    into one position per tick."""
    nonzero = np.where(signal > 0)[0]
    if len(nonzero) == 0:
        return []

    clusters: list[list[int]] = [[int(nonzero[0])]]
    for idx in nonzero[1:]:
        if idx - clusters[-1][-1] <= min_gap:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])

    return [int(np.mean(cluster)) for cluster in clusters]
