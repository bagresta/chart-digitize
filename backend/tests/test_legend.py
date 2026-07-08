from app.pipeline.geometry import detect_plot_box
from app.pipeline.legend import detect_legend_entries
from tests.fixtures import make_km_chart, make_scatter_chart


def test_detect_legend_entries_finds_both_arms():
    image, truth = make_km_chart()
    box = detect_plot_box(image)

    entries = detect_legend_entries(image, box)

    assert len(entries) == 2
    names = {entry.name.strip() for entry in entries}
    assert "Arm A" in names
    assert "Arm B" in names


def test_detect_legend_entries_ignores_scatter_markers_with_no_real_legend():
    """Scatter markers are small saturated-color blobs that satisfy the same
    swatch-detection heuristics as real legend swatches, but they don't form
    a vertically-stacked, left-aligned column the way a real legend does —
    they're scattered across the whole plot area. make_scatter_chart() has
    no legend at all, so detection must return zero entries rather than
    mistaking a couple of coincidentally-aligned markers for a legend."""
    image, truth = make_scatter_chart()
    box = detect_plot_box(image)

    entries = detect_legend_entries(image, box)

    assert entries == []
