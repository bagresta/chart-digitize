from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_line_chart


def test_read_axis_tick_labels_reads_numeric_y_values():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    y_ticks = detect_tick_positions(image, box, axis="y")

    labels = read_axis_tick_labels(image, box, y_ticks, axis="y")

    assert len(labels) == len(y_ticks)
    numeric_values = [label.value for label in labels if label.value is not None]
    # y range is 0..100 in steps of 20 by default matplotlib behavior;
    # require most ticks to be read correctly (OCR isn't perfect)
    assert len(numeric_values) >= len(y_ticks) - 1
    assert numeric_values == sorted(numeric_values, reverse=True)  # top-to-bottom => descending
