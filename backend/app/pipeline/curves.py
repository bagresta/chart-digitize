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
    2. Riser de-duplication (step mode only): a step's vertical jump is
       drawn as a diagonal, anti-aliased edge that itself spans 2-3 pixel
       columns, each contributing a run covering nearly the whole jump. If
       every such column's full extent were emitted, the second and later
       columns would restate ground the first one already covered and then
       flip back toward it, appearing as a spurious increase. So each
       column's run is clipped to only the direction beyond `last_row`
       (its previous frontier) before emitting a point, and a run that
       has `last_row` strictly inside it (i.e. contributes nothing new
       beyond what's already been traced) is skipped entirely.
    """
    points_px: list[tuple[int, int]] = []
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

        if step:
            # a step curve can have two y-levels in the same column at a
            # jump; record the extremes rather than averaging them away.
            # A diagonal (anti-aliased) riser can itself span 2-3 adjacent
            # pixel columns, each contributing a run that covers most of the
            # jump's row range. Emitting both endpoints of every such column
            # would restate ground already covered by a prior column and
            # then immediately "reverse" back toward it, looking like a
            # false direction change. So the run is clipped to only the
            # portion beyond the last traced row (i.e. genuinely new ground)
            # before its extremes are emitted; anything already covered by
            # an earlier column in this same riser is skipped.
            top, bottom = int(run.min()), int(run.max())
            if last_row is None:
                points_px.append((col, top))
                if bottom != top:
                    points_px.append((col, bottom))
                last_row = bottom
            elif top < last_row < bottom:
                # last_row already falls strictly inside this run: this
                # column's run is just the same riser we already traced
                # continuing to show up (its anti-aliased edges shift by a
                # pixel or two per column), not new movement. Nothing to add.
                continue
            elif last_row <= top:
                # extending downward (row increasing) past what we've seen
                points_px.append((col, bottom))
                last_row = bottom
            else:  # last_row >= bottom
                # extending upward (row decreasing) past what we've seen
                points_px.append((col, top))
                last_row = top
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
