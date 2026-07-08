import numpy as np

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from app.pipeline.overlay import draw_overlay
from tests.fixtures import make_line_chart


def test_draw_overlay_returns_modified_image_same_shape():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_cal = fit_axis_calibration(read_axis_tick_labels(image, box, x_ticks, axis="x"), log_scale=False)
    y_cal = fit_axis_calibration(read_axis_tick_labels(image, box, y_ticks, axis="y"), log_scale=False)

    series = [{"name": "Series A", "color_bgr": (255, 0, 0), "points": [(1.0, 13.0), (5.0, 45.0), (9.0, 77.0)]}]

    overlay_image = draw_overlay(image, box, x_cal, y_cal, series)

    assert overlay_image.shape == image.shape
    # overlay must actually change some pixels (markers drawn), not be a no-op copy
    assert not np.array_equal(overlay_image, image)
