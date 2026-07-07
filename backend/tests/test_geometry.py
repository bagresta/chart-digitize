from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from tests.fixtures import make_line_chart


def test_detect_plot_box_finds_axes_rectangle():
    image, _ = make_line_chart()
    box = detect_plot_box(image)

    h, w = image.shape[:2]
    assert 0 <= box.left < box.right <= w
    assert 0 <= box.top < box.bottom <= h
    # plot area should be a substantial, sane portion of the image
    assert (box.right - box.left) > w * 0.4
    assert (box.bottom - box.top) > h * 0.4


def test_detect_tick_positions_finds_expected_count_on_each_axis():
    image, truth = make_line_chart()
    box = detect_plot_box(image)

    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")

    # matplotlib default gives ~6 ticks per axis for these ranges; allow slack
    assert 4 <= len(x_ticks) <= 8
    assert 4 <= len(y_ticks) <= 8
    # x ticks should be sorted left-to-right, y ticks top-to-bottom
    assert x_ticks == sorted(x_ticks)
    assert y_ticks == sorted(y_ticks)
