from app.pipeline.geometry import detect_plot_box
from app.pipeline.legend import detect_legend_entries
from tests.fixtures import make_km_chart


def test_detect_legend_entries_finds_both_arms():
    image, truth = make_km_chart()
    box = detect_plot_box(image)

    entries = detect_legend_entries(image, box)

    assert len(entries) == 2
    names = {entry.name.strip().lower() for entry in entries}
    assert any("arm a" in name for name in names)
    assert any("arm b" in name for name in names)
