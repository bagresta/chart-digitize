from app.pipeline.classify import ChartType, classify_chart_type
from app.pipeline.geometry import detect_plot_box
from tests.fixtures import make_bar_chart, make_km_chart, make_line_chart, make_scatter_chart


def test_classify_line_chart():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.LINE


def test_classify_scatter_chart():
    image, _ = make_scatter_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.SCATTER


def test_classify_bar_chart():
    image, _ = make_bar_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.BAR


def test_classify_km_chart():
    image, _ = make_km_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.KAPLAN_MEIER
