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


def test_fit_calibration_rejects_single_ocr_misread_outlier():
    # Reproduces the real-world bug found via manual browser verification:
    # a Kaplan-Meier y-axis with ticks 1.0, 0.8, 0.6, 0.4, 0.2, 0.0 at evenly
    # spaced pixel positions, where Tesseract misread "0.4" as "2" for the
    # tick at pixel 233. An unweighted least-squares fit over all 6 points
    # let that single bad value drag the whole calibration off, producing
    # systematically wrong survival probabilities (up to 170% error) at
    # every other timepoint. The fitter should detect this one clear
    # outlier and refit without it.
    labels = [
        TickLabel(pixel_position=48, text="1", value=1.0),
        TickLabel(pixel_position=110, text="0.8", value=0.8),
        TickLabel(pixel_position=171, text="0.6", value=0.6),
        TickLabel(pixel_position=233, text="2", value=2.0),  # misread of "0.4"
        TickLabel(pixel_position=294, text="0.2", value=0.2),
        TickLabel(pixel_position=355, text="0", value=0.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=False)

    # Should match the fit obtained from the 5 correct points, not the
    # corrupted one.
    assert calibration.pixel_to_value(233) == pytest.approx(0.4, abs=0.05)
    assert calibration.pixel_to_value(48) == pytest.approx(1.0, abs=0.05)
    assert calibration.pixel_to_value(355) == pytest.approx(0.0, abs=0.05)


def test_fit_calibration_does_not_reject_normal_jitter():
    # Realistic tick positions are never perfectly linear in pixel space
    # (anti-aliasing, sub-pixel rendering, OCR bounding-box noise). None of
    # these deviations should be mistaken for a genuine outlier and dropped.
    labels = [
        TickLabel(pixel_position=49, text="1", value=1.0),
        TickLabel(pixel_position=109, text="0.8", value=0.8),
        TickLabel(pixel_position=173, text="0.6", value=0.6),
        TickLabel(pixel_position=230, text="0.4", value=0.4),
        TickLabel(pixel_position=296, text="0.2", value=0.2),
        TickLabel(pixel_position=354, text="0", value=0.0),
    ]

    calibration = fit_axis_calibration(labels, log_scale=False)

    # All 6 points should still be used - verify against the full-data
    # least-squares fit rather than a 5-point fit.
    import numpy as np

    pixels = np.array([label.pixel_position for label in labels], dtype=float)
    values = np.array([label.value for label in labels], dtype=float)
    expected_slope, expected_intercept = np.polyfit(pixels, values, deg=1)

    assert calibration.slope == pytest.approx(expected_slope)
    assert calibration.intercept == pytest.approx(expected_intercept)
