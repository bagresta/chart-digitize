import numpy as np
from tests.fixtures import make_line_chart


def test_make_line_chart_returns_image_and_ground_truth():
    image, truth = make_line_chart()

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3  # H, W, 3 (BGR)
    assert truth["x_range"] == (0.0, 10.0)
    assert truth["y_range"] == (0.0, 100.0)
    assert len(truth["series"]) == 1
    assert truth["series"][0]["name"] == "Series A"
    assert len(truth["series"][0]["points"]) == 11
