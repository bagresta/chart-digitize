import numpy as np

from app.pipeline.curves import isolate_series_mask
from app.pipeline.geometry import detect_plot_box
from tests.fixtures import make_km_chart


def test_isolate_series_mask_separates_red_and_green_arms():
    image, _ = make_km_chart()
    box = detect_plot_box(image)

    # matplotlib anti-aliases line colors, so we need higher tolerance than pure RGB values
    red_mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=130)
    green_mask = isolate_series_mask(image, box, target_color_bgr=(0, 255, 0), tolerance=130)

    assert red_mask.sum() > 0
    assert green_mask.sum() > 0
    # the two masks should be almost entirely disjoint
    overlap = np.logical_and(red_mask > 0, green_mask > 0).sum()
    assert overlap < min(red_mask.sum(), green_mask.sum()) * 0.05 / 255
