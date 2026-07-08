"""Top-level orchestration: image bytes in, structured extraction result out.
Ties together geometry detection, OCR-based calibration (or a manual
override), chart-type classification, and per-series point extraction."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration, fit_axis_calibration
from app.pipeline.classify import ChartType, classify_chart_type
from app.pipeline.curves import (
    detect_censoring_marks,
    extract_bar_heights,
    extract_scatter_points,
    isolate_series_mask,
    trace_line_curve,
)
from app.pipeline.geometry import PlotBox, detect_plot_box, detect_tick_positions
from app.pipeline.legend import LegendEntry, detect_legend_entries
from app.pipeline.ocr import read_axis_tick_labels
from app.pipeline.overlay import draw_overlay


@dataclass
class SeriesResult:
    name: str
    color_bgr: tuple[int, int, int]
    points: list[tuple[float, float]]
    censoring_marks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class PipelineResult:
    chart_type: str
    series: list[SeriesResult]
    overlay_image_png: bytes
    x_axis_calibrated_from_ocr: bool
    y_axis_calibrated_from_ocr: bool
    x_reference_points: list[tuple[float, float]]  # (pixel, value) pairs used to fit x calibration
    y_reference_points: list[tuple[float, float]]  # (pixel, value) pairs used to fit y calibration


def _decode_image(image_bytes: bytes) -> np.ndarray:
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image — unsupported or corrupt file")
    return image


def _build_axis_calibration(
    image: np.ndarray, box: PlotBox, axis: str, manual_range: tuple[float, float] | None
) -> tuple[AxisCalibration, bool, list[tuple[float, float]]]:
    if manual_range is not None:
        low, high = manual_range
        # two-point calibration directly from user-provided min/max at the
        # known plot-box edges, bypassing OCR entirely
        if axis == "x":
            slope = (high - low) / (box.right - box.left)
            intercept = low - slope * box.left
            reference_points = [(float(box.left), low), (float(box.right), high)]
        else:
            slope = (low - high) / (box.bottom - box.top)
            intercept = high - slope * box.top
            reference_points = [(float(box.bottom), low), (float(box.top), high)]
        return AxisCalibration(slope=slope, intercept=intercept, log_scale=False), False, reference_points

    ticks = detect_tick_positions(image, box, axis=axis)
    labels = read_axis_tick_labels(image, box, ticks, axis=axis)
    calibration = fit_axis_calibration(labels, log_scale=False)
    reference_points = [
        (float(label.pixel_position), label.value) for label in labels if label.value is not None
    ]
    return calibration, True, reference_points


def _find_series_color_for_legend(legend_entries: list[LegendEntry]) -> list[tuple[str, tuple[int, int, int]]]:
    return [(entry.name, entry.color_bgr) for entry in legend_entries]


def _dominant_series_color(image: np.ndarray, box: PlotBox) -> tuple[int, int, int]:
    """Fallback when no legend is found: assumes a single series and picks
    the most common non-white, non-grayscale color in the plot area."""
    region = image[box.top:box.bottom, box.left:box.right].reshape(-1, 3)
    is_colorful = (region.max(axis=1).astype(int) - region.min(axis=1).astype(int)) > 20
    colorful_pixels = region[is_colorful]
    if len(colorful_pixels) == 0:
        raise ValueError("No colored curve/marker pixels found in plot area")
    colors, counts = np.unique(colorful_pixels, axis=0, return_counts=True)
    return tuple(int(c) for c in colors[np.argmax(counts)])


def run_pipeline(
    image_bytes: bytes,
    manual_x_range: tuple[float, float] | None = None,
    manual_y_range: tuple[float, float] | None = None,
) -> PipelineResult:
    image = _decode_image(image_bytes)
    box = detect_plot_box(image)
    chart_type = classify_chart_type(image, box)

    x_calibration, x_from_ocr, x_reference_points = _build_axis_calibration(image, box, "x", manual_x_range)
    y_calibration, y_from_ocr, y_reference_points = _build_axis_calibration(image, box, "y", manual_y_range)

    legend_entries = detect_legend_entries(image, box)
    named_colors = _find_series_color_for_legend(legend_entries)
    if not named_colors:
        named_colors = [("Series A", _dominant_series_color(image, box))]

    series_results: list[SeriesResult] = []
    for name, color_bgr in named_colors:
        mask = isolate_series_mask(image, box, color_bgr, tolerance=60)

        if chart_type == ChartType.BAR:
            points = extract_bar_heights(mask, box, x_calibration, y_calibration)
            censoring: list[tuple[float, float]] = []
        elif chart_type == ChartType.SCATTER:
            points = extract_scatter_points(mask, box, x_calibration, y_calibration)
            censoring = []
        elif chart_type == ChartType.KAPLAN_MEIER:
            points = trace_line_curve(mask, box, x_calibration, y_calibration, step=True)
            censoring = detect_censoring_marks(mask, box, x_calibration, y_calibration)
        else:  # LINE
            points = trace_line_curve(mask, box, x_calibration, y_calibration, step=False)
            censoring = []

        series_results.append(
            SeriesResult(name=name, color_bgr=color_bgr, points=points, censoring_marks=censoring)
        )

    overlay_image = draw_overlay(
        image, box, x_calibration, y_calibration,
        [{"name": s.name, "color_bgr": s.color_bgr, "points": s.points} for s in series_results],
    )
    success, overlay_encoded = cv2.imencode(".png", overlay_image)
    if not success:
        raise ValueError("Failed to encode overlay image")

    return PipelineResult(
        chart_type=chart_type.value,
        series=series_results,
        overlay_image_png=overlay_encoded.tobytes(),
        x_axis_calibrated_from_ocr=x_from_ocr,
        y_axis_calibrated_from_ocr=y_from_ocr,
        x_reference_points=x_reference_points,
        y_reference_points=y_reference_points,
    )
