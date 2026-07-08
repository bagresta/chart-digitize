import cv2
import pytest

from app.pipeline.pipeline import AxisCalibrationError, _dedupe_legend_entries_by_color, run_pipeline
from tests.fixtures import make_km_chart, make_line_chart, make_scatter_chart


def _encode(image):
    success, buf = cv2.imencode(".png", image)
    assert success
    return buf.tobytes()


def test_run_pipeline_on_single_series_line_chart():
    image, truth = make_line_chart()
    result = run_pipeline(_encode(image))

    assert result.chart_type == "line"
    assert len(result.series) == 1
    assert len(result.series[0].points) > 5
    mid_points = [p for p in result.series[0].points if 4 <= p[0] <= 6]
    assert mid_points
    mid = mid_points[len(mid_points) // 4]
    assert mid[1] == pytest.approx(8 * mid[0] + 5, abs=8)
    assert result.overlay_image_png  # non-empty bytes


def test_run_pipeline_on_km_chart_finds_both_arms():
    image, truth = make_km_chart()
    result = run_pipeline(_encode(image))

    assert result.chart_type == "kaplan_meier"
    assert len(result.series) == 2
    names = {s.name.strip().lower() for s in result.series}
    assert any("arm a" in n for n in names)
    assert any("arm b" in n for n in names)


def test_run_pipeline_with_manual_axis_override_skips_ocr_calibration():
    image, truth = make_line_chart()
    result = run_pipeline(
        _encode(image),
        manual_x_range=(0.0, 10.0),
        manual_y_range=(0.0, 100.0),
    )

    assert len(result.series) == 1
    assert len(result.series[0].points) > 5


def test_run_pipeline_raises_axis_calibration_error_when_ocr_finds_no_numbers(monkeypatch):
    from app.pipeline import pipeline as pipeline_module

    def _unreadable_labels(*args, **kwargs):
        from app.pipeline.ocr import TickLabel
        return [TickLabel(pixel_position=10, text="???", value=None)]

    monkeypatch.setattr(pipeline_module, "read_axis_tick_labels", _unreadable_labels)

    image, _ = make_line_chart()
    with pytest.raises(AxisCalibrationError):
        run_pipeline(_encode(image))


def test_run_pipeline_on_scatter_chart_produces_single_series():
    """Regression test for the confirmed Task 13 bug: detect_legend_entries
    used to produce two spurious legend entries (both sampling the same
    purple marker color, named after garbage OCR text) on charts with no
    real legend at all, splitting one true series into two fake ones. Now
    that legend.py rejects unaligned swatch candidates, the dominant-color
    fallback should kick in and yield exactly one series."""
    image, truth = make_scatter_chart()
    result = run_pipeline(_encode(image))

    assert result.chart_type == "scatter"
    assert len(result.series) == 1
    assert len(result.series[0].points) > 5


def test_dedupe_legend_entries_by_color_merges_near_identical_colors():
    # Two entries with a color distance of 20 (well within the dedup
    # tolerance) should collapse to one, keeping the more plausible name.
    named_colors = [
        ("ee", (128, 0, 128)),
        ("Sn", (130, 0, 135)),
    ]
    result = _dedupe_legend_entries_by_color(named_colors)
    assert result == [("ee", (128, 0, 128))]


def test_dedupe_legend_entries_by_color_prefers_more_plausible_name():
    named_colors = [
        ("Sn", (128, 0, 128)),
        ("Series A", (130, 0, 132)),
    ]
    result = _dedupe_legend_entries_by_color(named_colors)
    assert result == [("Series A", (130, 0, 132))]


def test_dedupe_legend_entries_by_color_keeps_genuinely_different_colors():
    named_colors = [
        ("Arm A", (0, 0, 255)),  # red
        ("Arm B", (0, 128, 0)),  # green
    ]
    result = _dedupe_legend_entries_by_color(named_colors)
    assert result == named_colors
