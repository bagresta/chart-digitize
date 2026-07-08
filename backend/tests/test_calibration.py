import pytest

from app.pipeline.calibration import AxisCalibration, fit_axis_calibration
from app.pipeline.ocr import TickLabel


def test_fit_linear_calibration_maps_pixels_to_values():
    # 5 ticks evenly spaced in pixels, matching evenly spaced values 0..100
    labels = [
        TickLabel(pixel_position=300, text="0", value=0.0),
        TickLabel(pixel_position=225, text="25", value=25.0),
        TickLabel(pixel_position=150, text="50", value=50.0),
        TickLabel(pixel_position=75, text="75", value=75.0),
        TickLabel(pixel_position=0, text="100", value=100.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=False)

    assert calibration.pixel_to_value(300) == pytest.approx(0.0, abs=0.5)
    assert calibration.pixel_to_value(150) == pytest.approx(50.0, abs=0.5)
    assert calibration.pixel_to_value(0) == pytest.approx(100.0, abs=0.5)


def test_fit_calibration_ignores_unreadable_ticks():
    labels = [
        TickLabel(pixel_position=300, text="0", value=0.0),
        TickLabel(pixel_position=225, text="", value=None),  # OCR failed on this one
        TickLabel(pixel_position=150, text="50", value=50.0),
        TickLabel(pixel_position=0, text="100", value=100.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=False)

    assert calibration.pixel_to_value(150) == pytest.approx(50.0, abs=0.5)


def test_fit_calibration_raises_with_fewer_than_two_readable_ticks():
    labels = [TickLabel(pixel_position=300, text="0", value=0.0)]

    with pytest.raises(ValueError, match="at least 2"):
        fit_axis_calibration(labels, log_scale=False)


def test_fit_log_calibration():
    labels = [
        TickLabel(pixel_position=300, text="1", value=1.0),
        TickLabel(pixel_position=200, text="10", value=10.0),
        TickLabel(pixel_position=100, text="100", value=100.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=True)

    assert calibration.pixel_to_value(300) == pytest.approx(1.0, rel=0.05)
    assert calibration.pixel_to_value(200) == pytest.approx(10.0, rel=0.05)
    assert calibration.pixel_to_value(100) == pytest.approx(100.0, rel=0.05)
