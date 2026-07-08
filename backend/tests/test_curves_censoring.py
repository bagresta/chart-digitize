import pytest

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import detect_censoring_marks, isolate_series_mask
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_km_chart


def test_detect_censoring_marks_finds_plus_markers():
    image, truth = make_km_chart()
    box = detect_plot_box(image)
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_cal = fit_axis_calibration(read_axis_tick_labels(image, box, x_ticks, axis="x"), log_scale=False)
    y_cal = fit_axis_calibration(read_axis_tick_labels(image, box, y_ticks, axis="y"), log_scale=False)

    mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=60)  # red arm
    marks = detect_censoring_marks(mask, box, x_cal, y_cal)

    # fixture has 2 censoring points on the red arm, at t=5 and t=9
    assert len(marks) == 2
    times = sorted(x for x, _ in marks)
    assert times[0] == pytest.approx(5, abs=1)
    assert times[1] == pytest.approx(9, abs=1)
