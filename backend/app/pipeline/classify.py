"""Classifies chart type from plotted shape characteristics — no OCR or
labels needed. Order of checks matters: bar charts are ruled in first since
they have the most distinctive signature (large filled rectangular blobs
sharing a common baseline), then step-function detection distinguishes
Kaplan-Meier curves from ordinary line charts, then remaining charts are
split into line vs. scatter by how much of each column-of-pixels forms a
continuous stroke versus isolated blobs."""
from enum import Enum

import cv2
import numpy as np

from app.pipeline.geometry import PlotBox


class ChartType(str, Enum):
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    KAPLAN_MEIER = "kaplan_meier"


_BORDER_TRIM = 3


def _non_background_mask(region: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    # detect_plot_box's coordinates sit exactly on the axis spine lines, so
    # the crop's outer border is entirely spine pixels. Left uncropped, that
    # border fuses with anything touching it (e.g. bars touching the bottom
    # axis) into one giant contour tracing the whole box. Trim a few pixels
    # of border so shape analysis only sees the plotted data.
    trimmed = mask[_BORDER_TRIM:-_BORDER_TRIM, _BORDER_TRIM:-_BORDER_TRIM]
    return np.pad(trimmed, _BORDER_TRIM, mode="constant", constant_values=0)


def _is_bar_chart(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = mask.shape[0]
    bar_like = 0
    for contour in contours:
        x, y, w, cy = cv2.boundingRect(contour)
        area = w * cy
        if area < mask.size * 0.01:
            continue
        # bars are tall filled rectangles that touch (or nearly touch) the bottom
        fill_ratio = cv2.contourArea(contour) / area if area else 0
        touches_bottom = (y + cy) >= h - 5
        if fill_ratio > 0.85 and touches_bottom and cy > w:
            bar_like += 1
    return bar_like >= 1


def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Isolates the largest connected component, e.g. the main plotted
    stroke — this keeps small, spatially separate clutter (legend swatches,
    legend text, censoring-tick markers) from skewing per-column extent
    analysis, since those sit in their own disconnected blobs."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    # label 0 is background; pick the largest foreground component by area
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, mask, 0).astype(np.uint8)


def _has_step_pattern(mask: np.ndarray) -> bool:
    """Step functions alternate flat treads (long horizontal runs of
    foreground pixels within a single row) with short vertical risers —
    detected by counting rows that contain a long horizontal run. A
    per-column vertical-extent check was tried first but breaks down for
    multi-series KM charts, since step curves occupying overlapping y-ranges
    in the same columns merge into one connected component whose per-column
    extent spans both curves,
    masking the flat treads. Row-based horizontal-run detection is immune to
    that: a diagonal line (or scatter blobs) never produces a long
    horizontal run in any single row, no matter how many series overlap.
    Restricted to the largest connected component so an unrelated legend box
    sharing the same column range doesn't add spurious long runs."""
    stroke = _largest_component_mask(mask)

    long_runs = 0
    rows_with_data = 0
    for row in range(stroke.shape[0]):
        cols = np.where(stroke[row, :] > 0)[0]
        if len(cols) == 0:
            continue
        rows_with_data += 1
        splits = np.where(np.diff(cols) > 1)[0]
        runs = np.split(cols, splits + 1)
        max_run_len = max(len(r) for r in runs)
        if max_run_len > stroke.shape[1] * 0.08:
            long_runs += 1

    if rows_with_data == 0:
        return False

    return long_runs >= 3 and (long_runs / rows_with_data) > 0.05


def _is_scatter(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_blobs = [c for c in contours if cv2.contourArea(c) < mask.size * 0.02]
    # scatter: many small, disconnected blobs rather than one continuous stroke
    return len(small_blobs) >= 6


def classify_chart_type(image: np.ndarray, box: PlotBox) -> ChartType:
    region = image[box.top:box.bottom, box.left:box.right]
    mask = _non_background_mask(region)

    if _is_bar_chart(mask):
        return ChartType.BAR
    if _has_step_pattern(mask):
        return ChartType.KAPLAN_MEIER
    if _is_scatter(mask):
        return ChartType.SCATTER
    return ChartType.LINE
