import pytest

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import extract_bar_heights, extract_scatter_points, isolate_series_mask
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_bar_chart, make_scatter_chart


def _calibrate(image, box):
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_labels = read_axis_tick_labels(image, box, x_ticks, axis="x")
    y_labels = read_axis_tick_labels(image, box, y_ticks, axis="y")
    return (
        fit_axis_calibration(x_labels, log_scale=False),
        fit_axis_calibration(y_labels, log_scale=False),
    )


def test_extract_scatter_points_recovers_approximate_count():
    image, truth = make_scatter_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(128, 0, 128), tolerance=70)  # purple
    points = extract_scatter_points(mask, box, x_cal, y_cal)

    assert len(points) == pytest.approx(len(truth["series"][0]["points"]), abs=3)


def test_extract_bar_heights_recovers_known_heights():
    image, truth = make_bar_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(128, 128, 0), tolerance=70)  # teal
    bars = extract_bar_heights(mask, box, x_cal, y_cal)

    assert len(bars) == 4
    heights = sorted(y for _, y in bars)
    expected = sorted(y for _, y in truth["series"][0]["points"])
    for actual, exp in zip(heights, expected):
        assert actual == pytest.approx(exp, abs=3)
