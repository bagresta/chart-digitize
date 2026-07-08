"""Isolates and traces individual series' pixels from the plot area, and
extracts (x, y) data points for line/step, scatter, and bar chart types."""
import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration
from app.pipeline.geometry import PlotBox


def isolate_series_mask(
    image: np.ndarray, box: PlotBox, target_color_bgr: tuple[int, int, int], tolerance: int = 40
) -> np.ndarray:
    region = image[box.top:box.bottom, box.left:box.right]
    target = np.array(target_color_bgr, dtype=np.int16)
    diff = np.abs(region.astype(np.int16) - target).sum(axis=2)
    mask = np.where(diff <= tolerance, 255, 0).astype(np.uint8)
    return mask


def _cluster_row_runs(rows: np.ndarray, max_gap: int = 2) -> list[np.ndarray]:
    """Groups a sorted array of row indices into contiguous runs, allowing a
    small gap (default 2px) between consecutive rows so anti-aliased strokes
    that are broken up by a pixel or two are still treated as one run."""
    runs: list[list[int]] = [[int(rows[0])]]
    for row in rows[1:]:
        if int(row) - runs[-1][-1] <= max_gap:
            runs[-1].append(int(row))
        else:
            runs.append([int(row)])
    return [np.array(run) for run in runs]


def trace_line_curve(
    mask: np.ndarray,
    box: PlotBox,
    x_calibration: AxisCalibration,
    y_calibration: AxisCalibration,
    step: bool,
) -> list[tuple[float, float]]:
    """For each pixel-column containing series pixels, takes the vertical
    center (line mode) or, for step-function curves, keeps every distinct
    flat run's y-level so vertical jumps are preserved as separate points.

    Real published figures routinely place a legend swatch, censoring tick
    mark, or other annotation in the exact same color as the traced curve,
    and those can overlap the curve's column range — this isn't a synthetic
    edge case, it's normal for Kaplan-Meier figures in particular. When a
    column's foreground pixels aren't a single contiguous blob,
    `isolate_series_mask` can't tell curve pixels apart from this
    same-colored clutter by color alone, so two continuity heuristics are
    applied here based on the most recently traced point's row
    (`last_row`), on the assumption that the true curve moves continuously
    column-to-column while unrelated clutter sits at a disconnected row:

    1. Row-run selection: pixels in a column are first grouped into
       contiguous runs (allowing a small gap for anti-aliasing). If a
       column has more than one run, the run closest to `last_row` is kept
       and the others (e.g. a legend swatch or censoring "+" mark sitting
       far away) are discarded. The very first column with any data has no
       prior point to anchor to, so it falls back to the run with the most
       pixels, since a genuine line/step stroke is typically much thicker
       than a small swatch or tick mark.
    2. Riser/plateau de-duplication (step mode only): a step's vertical
       jump is drawn as a diagonal, anti-aliased edge that itself spans 2-3
       pixel columns, each contributing a run covering nearly the whole
       jump; and a flat plateau repeats essentially the same run across
       every column it spans. Either way, once a column's run has been
       processed, later columns whose run is (within anti-aliasing noise)
       the same span contribute nothing new and are skipped outright —
       tracked via `last_run`, the previous column's full (top, bottom)
       span, not just a single row. This is deliberately not a check
       against `last_row` alone: on a flat plateau, the two boundary rows
       (top and bottom) are the only values ever emitted, so `last_row`
       always sits exactly on one edge of every later identical run, and a
       same-vs-different-span comparison is needed to recognize "nothing
       changed" instead of misreading the repeat as fresh movement (which
       would otherwise ping-pong between the plateau's two edges forever).
       When a column's run *does* differ from the previous one — a genuine
       riser or a new plateau — only the endpoint that extends beyond
       `last_row` (the previously traced frontier) is emitted, so a
       multi-column riser doesn't restate ground already covered by an
       earlier column in the same jump.
    """
    points_px: list[tuple[int, int]] = []
    last_row: int | None = None
    last_run: tuple[int, int] | None = None

    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) == 0:
            continue

        runs = _cluster_row_runs(rows)
        if len(runs) > 1:
            if last_row is None:
                run = max(runs, key=len)
            else:
                run = min(runs, key=lambda r: min(abs(int(r.min()) - last_row), abs(int(r.max()) - last_row)))
        else:
            run = runs[0]

        if step:
            # a step curve can have two y-levels in the same column at a
            # jump; record the extremes rather than averaging them away.
            top, bottom = int(run.min()), int(run.max())
            if last_run is None:
                points_px.append((col, top))
                if bottom != top:
                    points_px.append((col, bottom))
                last_row = bottom
                last_run = (top, bottom)
                continue

            prev_top, prev_bottom = last_run
            if abs(top - prev_top) <= 2 and abs(bottom - prev_bottom) <= 2:
                # same span as the previous data column (within
                # anti-aliasing noise): a plateau continuing, or a riser
                # column repeating — nothing new to emit either way.
                last_run = (top, bottom)
                continue

            if last_row <= top:
                # extending downward (row increasing) past what we've seen
                points_px.append((col, bottom))
                last_row = bottom
            elif last_row >= bottom:
                # extending upward (row decreasing) past what we've seen
                points_px.append((col, top))
                last_row = top
            else:
                # last_row falls inside this differing run's span (e.g. a
                # riser column widening mid-transition); advance to
                # whichever extreme is farther as the new frontier.
                new_last = bottom if (bottom - last_row) >= (last_row - top) else top
                points_px.append((col, new_last))
                last_row = new_last
            last_run = (top, bottom)
        else:
            row = int(run.mean())
            points_px.append((col, row))
            last_row = row

    points_px.sort(key=lambda p: p[0])

    points_data = [
        (
            x_calibration.pixel_to_value(box.left + col),
            y_calibration.pixel_to_value(box.top + row),
        )
        for col, row in points_px
    ]
    return points_data


def extract_scatter_points(
    mask: np.ndarray, box: PlotBox, x_calibration: AxisCalibration, y_calibration: AxisCalibration
) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points_data = []
    for contour in contours:
        if cv2.contourArea(contour) < 3:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        points_data.append(
            (
                x_calibration.pixel_to_value(box.left + cx),
                y_calibration.pixel_to_value(box.top + cy),
            )
        )
    return points_data


def detect_censoring_marks(
    mask: np.ndarray, box: PlotBox, x_calibration: AxisCalibration, y_calibration: AxisCalibration
) -> list[tuple[float, float]]:
    """Finds Kaplan-Meier censoring tick marks ('+' or '|' glyphs plotted in
    the same color as the survival curve).

    A naive `cv2.findContours(mask, cv2.RETR_EXTERNAL, ...)` over the whole
    mask does *not* reliably isolate these as their own small contours: a
    censoring mark is frequently plotted directly on the curve's stroke, so
    its pixels are edge- or corner-adjacent to the curve's pixels and get
    swallowed into the same external contour as the (much larger) curve
    itself. On this project's own KM fixture, exactly that happens for the
    censoring mark at t=9 — its vertical stroke sits immediately below the
    curve's plateau row with no gap, so `findContours` merges mark and curve
    into a single giant contour, while the t=5 mark (which happens to have a
    few blank rows between it and the curve) survives as an independent
    contour. Relying on that coincidence would make detection fragile.

    Instead, this reuses the same per-column row-run continuity logic that
    `trace_line_curve` uses to follow the curve (via `_cluster_row_runs`),
    tracks which run in each column is "the curve" (closest to the
    previous column's chosen run, precisely mirroring `trace_line_curve`'s
    own selection so both functions agree on what counts as curve), and
    paints only those chosen runs into a `curve_mask`. Subtracting
    `curve_mask` from the full series mask leaves behind only the material
    the curve tracer *didn't* claim: censoring ticks, legend swatches, and
    similar same-colored clutter — even where a mark was touching the curve,
    since only the specific run pixels the tracer selected are removed, not
    the whole connected component.

    A censoring "+" mark's own two strokes (horizontal and vertical) are
    thin and can still land in adjacent-but-not-touching row-run fragments
    after subtraction (e.g. the t=9 mark's horizontal bar and vertical stem
    end up a few blank rows apart once the shared curve pixels are removed).
    A small dilation re-merges same-mark fragments into one blob before
    `findContours` runs, without merging distinct marks or the legend swatch
    (verified against this fixture: a 7x7 kernel merges the t=9 mark's two
    fragments into one blob while keeping it and the legend swatch
    separate). Centroids are then computed from the *original* (undilated)
    remainder pixels so the reported position isn't biased by the dilation.

    Finally, blobs are filtered to plausible mark shapes: small, and close
    to square (a "+" or "|" glyph) rather than the long thin strip typical
    of a legend color swatch. A minimum *actual* (undilated) pixel count is
    also required: the diagonal, anti-aliased edge of an ordinary step
    riser can itself split into two row-runs for a column or two (the same
    ambiguity `trace_line_curve` resolves via continuity), leaving a
    stray 1-2px fragment in `remainder` that has nothing to do with
    censoring. On this fixture, genuine "+" marks contribute ~24-25 real
    pixels versus ~2 for riser anti-aliasing noise, so a small pixel-count
    floor (well below a real mark, comfortably above noise) filters these
    out without needing exact tuning.
    """
    curve_mask = np.zeros_like(mask)
    last_row: int | None = None
    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        runs = _cluster_row_runs(rows)
        if len(runs) > 1:
            if last_row is None:
                run = max(runs, key=len)
            else:
                run = min(runs, key=lambda r: min(abs(int(r.min()) - last_row), abs(int(r.max()) - last_row)))
        else:
            run = runs[0]
        curve_mask[run.min():run.max() + 1, col] = 255
        last_row = int(run.mean())

    remainder = cv2.bitwise_and(mask, cv2.bitwise_not(curve_mask))
    if not np.any(remainder):
        return []

    kernel = np.ones((7, 7), np.uint8)
    merged = cv2.dilate(remainder, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < 6 or area > 400:
            continue
        aspect = cw / ch if ch else 0
        if not (0.4 <= aspect <= 2.5):
            continue  # skip long thin strips (legend swatches etc.)

        # centroid from the original (undilated) remainder pixels so the
        # dilation used for merging doesn't skew the reported position.
        orig_rows, orig_cols = np.where(remainder[y:y + ch, x:x + cw] > 0)
        if len(orig_rows) < 8:
            continue  # too few real pixels to be a mark; likely riser anti-aliasing noise
        center_x_px = box.left + x + orig_cols.mean()
        center_y_px = box.top + y + orig_rows.mean()
        marks.append(
            (
                x_calibration.pixel_to_value(center_x_px),
                y_calibration.pixel_to_value(center_y_px),
            )
        )
    return marks


def extract_bar_heights(
    mask: np.ndarray, box: PlotBox, x_calibration: AxisCalibration, y_calibration: AxisCalibration
) -> list[tuple[float, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < mask.size * 0.005:
            continue
        center_x_px = box.left + x + w / 2
        top_y_px = box.top + y  # top edge of the bar = its value
        bars.append(
            (
                x_calibration.pixel_to_value(center_x_px),
                y_calibration.pixel_to_value(top_y_px),
            )
        )
    bars.sort(key=lambda p: p[0])
    return bars
