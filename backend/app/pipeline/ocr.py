"""Reads axis tick label text/numbers, axis titles, and legend text from a
chart image using Tesseract OCR. Each region is cropped and upscaled before
OCR to maximize accuracy on small chart label text."""
import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from app.pipeline.geometry import PlotBox

_UPSCALE_FACTOR = 3
_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


@dataclass
class TickLabel:
    pixel_position: int
    text: str
    value: float | None


def _preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(
        gray, None, fx=_UPSCALE_FACTOR, fy=_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC
    )
    _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _parse_number(text: str) -> float | None:
    text = text.strip().replace(",", "")
    match = _NUMBER_PATTERN.search(text)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def read_axis_tick_labels(
    image: np.ndarray, box: PlotBox, tick_positions: list[int], axis: str
) -> list[TickLabel]:
    labels: list[TickLabel] = []
    h, w = image.shape[:2]

    for pos in tick_positions:
        if axis == "x":
            crop = image[box.bottom + 5: min(h, box.bottom + 35), max(0, pos - 30): pos + 30]
            config = "--psm 7"
        elif axis == "y":
            crop = image[max(0, pos - 12): pos + 12, max(0, box.left - 60): max(0, box.left - 5)]
            config = "--psm 7"
        else:
            raise ValueError(f"Unknown axis: {axis}")

        if crop.size == 0:
            labels.append(TickLabel(pixel_position=pos, text="", value=None))
            continue

        processed = _preprocess_for_ocr(crop)
        text = pytesseract.image_to_string(processed, config=config).strip()
        labels.append(TickLabel(pixel_position=pos, text=text, value=_parse_number(text)))

    return labels


def read_axis_title(image: np.ndarray, box: PlotBox, axis: str) -> str:
    h, w = image.shape[:2]
    if axis == "x":
        # the x title sits below the tick label row, near the bottom of the image
        crop = image[box.bottom + 22: h, box.left:box.right]
        if crop.size == 0:
            return ""
        processed = _preprocess_for_ocr(crop)
        return pytesseract.image_to_string(processed, config="--psm 7").strip()
    elif axis == "y":
        # the y title is the leftmost (rotated) text, further left than the
        # tick number labels which sit closer to the spine
        crop = image[box.top:box.bottom, 0: max(0, box.left - 35)]
        if crop.size == 0:
            return ""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # matplotlib draws the rotated y-axis title text such that a naive
        # counterclockwise rotation (the "obvious" inverse) comes out upside
        # down/mirrored; a clockwise rotation is what reads correctly here.
        # Empirically verified against the fixture image — do not "fix" this
        # back to ROTATE_90_COUNTERCLOCKWISE.
        rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        upscaled = cv2.resize(rotated, None, fx=_UPSCALE_FACTOR, fy=_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(binary, config="--psm 7").strip()
    else:
        raise ValueError(f"Unknown axis: {axis}")
