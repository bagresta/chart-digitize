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
# Real matplotlib legends draw every swatch's handle patch flush against the
# same left edge, so genuine swatches land at (empirically) *exactly* the
# same x pixel — verified at 0px drift across upper-right/lower-left/center
# legend placements. Individual scatter/line data markers, by contrast, are
# scattered across the whole plot width and only ever land within a few
# pixels of each other by pure chance. On this project's own scatter-chart
# fixture (15 random points), the closest two unrelated markers land 9px
# apart (confirmed by brute-force testing tolerances 0-15px against this
# fixture) — so the column-alignment tolerance must stay below that to
# avoid mistaking marker noise for an aligned legend column. 8px keeps a
# safe margin under the 9px false-grouping threshold on this fixture while
# giving more slack than a tighter value for real-world (non-synthetic,
# JPEG/screenshot) chart images, where swatch-centroid detection may drift
# further than on a clean vector-rendered matplotlib PNG. Note: this
# heuristic is tuned against synthetic fixtures only — legend alignment
# drift on real published figures (scanned/photographed, or rendered by
# tools other than matplotlib) hasn't been validated and may need
# retuning once tested against real chart images.
_SWATCH_COLUMN_X_TOLERANCE = 8


@dataclass
class LegendEntry:
    name: str
    color_bgr: tuple[int, int, int]
    swatch_position: tuple[int, int]  # (x, y) center, for matching to curve colors


def _is_near_grayscale(color_bgr: tuple[int, int, int], tolerance: int = 12) -> bool:
    b, g, r = color_bgr
    return max(b, g, r) - min(b, g, r) < tolerance


def _filter_to_largest_aligned_column(entries: list["LegendEntry"]) -> list["LegendEntry"]:
    """Rejects swatch candidates that don't form a coherent legend column.

    Real legends draw every swatch's handle patch at the same x position
    (a left-aligned, vertically-stacked column); individual scatter/line
    data markers are scattered across the plot area and essentially never
    align in x together. Group candidates by x-position (small tolerance,
    to allow for anti-aliasing/rounding noise) and keep only the largest
    group.

    A single, ungrouped candidate is a weak signal on its own — it's
    consistent with either a genuine single-entry legend or a lone
    false-positive swatch-like blob (e.g. a scatter marker) — so this
    deliberately biases toward NOT inventing a legend that isn't clearly
    there: only groups of 2+ aligned candidates are accepted as a legend.
    Ties for largest group size are also rejected for the same reason —
    multiple equally-sized, mutually-misaligned columns aren't a coherent
    single legend.
    """
    if not entries:
        return []

    groups: list[list["LegendEntry"]] = []
    for entry in sorted(entries, key=lambda e: e.swatch_position[0]):
        x = entry.swatch_position[0]
        if groups and x - groups[-1][-1].swatch_position[0] <= _SWATCH_COLUMN_X_TOLERANCE:
            groups[-1].append(entry)
        else:
            groups.append([entry])

    largest_size = max(len(g) for g in groups)
    if largest_size < 2:
        return []  # no group of 2+ aligned candidates — no coherent legend column

    largest_groups = [g for g in groups if len(g) == largest_size]
    if len(largest_groups) != 1:
        return []  # tie between multiple equally-sized, misaligned columns — ambiguous

    return largest_groups[0]


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

    return _filter_to_largest_aligned_column(entries)
