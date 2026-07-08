import cv2
import pytest

from app.pipeline.pipeline import run_pipeline
from tests.fixtures import make_km_chart, make_line_chart


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
