"""Finds legend entries by looking for small solid-color swatches followed
immediately (to the right) by a run of text, then OCRs that text. Legends in
matplotlib-style figures are drawn as a tight box of swatch+label rows, so
scanning for that pattern generalizes reasonably well beyond matplotlib too."""
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from app.pipeline.geometry import PlotBox

_MIN_SWATCH_AREA = 20
_MAX_SWATCH_AREA = 3000
# matplotlib legend swatches for line plots are drawn as thin wide line
# samples (e.g. ~31x3 px), not square patches, so the aspect range must
# tolerate very wide/short rectangles in addition to square-ish ones.
_SWATCH_ASPECT_RANGE = (0.1, 15.0)
_TEXT_CROP_VPAD = 10
_TEXT_CROP_WIDTH = 150
_TEXT_UPSCALE_FACTOR = 10


@dataclass
class LegendEntry:
    name: str
    color_bgr: tuple[int, int, int]
    swatch_position: tuple[int, int]  # (x, y) center, for matching to curve colors


def _is_near_grayscale(color_bgr: tuple[int, int, int], tolerance: int = 12) -> bool:
    b, g, r = color_bgr
    return max(b, g, r) - min(b, g, r) < tolerance


def detect_legend_entries(image: np.ndarray, box: PlotBox) -> list[LegendEntry]:
    # Search inside the plot box for small saturated-color rectangular blobs
    # (candidate legend swatches) — legends are typically drawn inside or
    # just outside the axes.
    region = image[box.top:box.bottom, box.left:box.right]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Detect saturated colors (S>=80, V>=60) across all hues (H spans the
    # full 0-179 OpenCV range) — this picks out vivid swatch colors while
    # excluding grayscale/black axis and text pixels.
    saturated_mask = cv2.inRange(hsv, (0, 80, 60), (179, 255, 255))

    contours, _ = cv2.findContours(saturated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    entries: list[LegendEntry] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not (_MIN_SWATCH_AREA <= area <= _MAX_SWATCH_AREA):
            continue
        aspect = w / h if h else 0
        if not (_SWATCH_ASPECT_RANGE[0] <= aspect <= _SWATCH_ASPECT_RANGE[1]):
            continue

        swatch_color = region[y + h // 2, x + w // 2].tolist()
        if _is_near_grayscale(tuple(swatch_color)):
            continue  # likely text or axis artifact, not a color swatch

        # legend swatches for line-plot handles are very short (a few px
        # tall), so pad generously above/below to capture the full label
        # text height rather than just the swatch's own bounding box.
        text_crop = region[max(0, y - _TEXT_CROP_VPAD): y + h + _TEXT_CROP_VPAD,
                            x + w + 3: min(region.shape[1], x + w + _TEXT_CROP_WIDTH)]
        if text_crop.size == 0:
            continue

        gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
        # A large upscale factor (vs. the 3x used elsewhere in the pipeline)
        # is needed here: legend label text is rendered very small, and at
        # lower upscale factors Tesseract tends to merge adjacent glyphs
        # (e.g. "Arm A" -> "ArmA"), losing the inter-word space.
        upscaled = cv2.resize(gray, None, fx=_TEXT_UPSCALE_FACTOR, fy=_TEXT_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Tesseract sometimes picks up the legend box's border as a stray
        # trailing "|" glyph; strip it (and any space left in its place) so
        # downstream consumers get an exact series name, not an OCR artifact.
        text = pytesseract.image_to_string(binary, config="--psm 7").strip().rstrip("|").strip()

        if not text:
            continue

        entries.append(
            LegendEntry(
                name=text,
                color_bgr=tuple(swatch_color),
                swatch_position=(box.left + x + w // 2, box.top + y + h // 2),
            )
        )

    return entries
