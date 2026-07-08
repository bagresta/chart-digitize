import pytest

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import isolate_series_mask, trace_line_curve
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_km_chart, make_line_chart


def _calibrate(image, box):
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_labels = read_axis_tick_labels(image, box, x_ticks, axis="x")
    y_labels = read_axis_tick_labels(image, box, y_ticks, axis="y")
    return (
        fit_axis_calibration(x_labels, log_scale=False),
        fit_axis_calibration(y_labels, log_scale=False),
    )


def test_trace_line_curve_recovers_known_line():
    image, truth = make_line_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(255, 0, 0), tolerance=60)  # blue
    points = trace_line_curve(mask, box, x_cal, y_cal, step=False)

    assert len(points) > 5
    # y = 8x + 5 in the fixture; check a mid-range traced point is close
    mid_points = [p for p in points if 4 <= p[0] <= 6]
    assert mid_points
    x, y = mid_points[len(mid_points) // 2]
    assert y == pytest.approx(8 * x + 5, abs=5)


def test_trace_line_curve_step_mode_recovers_km_curve():
    image, truth = make_km_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=60)  # red
    points = trace_line_curve(mask, box, x_cal, y_cal, step=True)

    assert len(points) > 3
    # survival should be non-increasing across a KM curve
    y_values = [p[1] for p in points]
    assert all(y_values[i] >= y_values[i + 1] - 0.05 for i in range(len(y_values) - 1))

    # Regression guard: a step curve with a handful of flat plateaus should
    # produce roughly one point per plateau edge, not dozens. A prior bug in
    # the riser/plateau de-duplication logic ping-ponged between a flat
    # run's two boundary rows on every column of every plateau (since
    # `last_row` always sits exactly on one edge of a repeated run), which
    # inflated a curve that should trace to well under 20 points into
    # hundreds. Bound the point count by the number of distinct traced
    # y-levels (rounded to absorb anti-aliasing jitter of a pixel or two) —
    # a handful of points per distinct level is expected (each plateau's
    # two edges, plus a couple of riser points), but not dozens.
    distinct_levels = {round(y, 2) for y in y_values}
    assert len(points) <= len(distinct_levels) * 3
