# Chart Digitizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a hosted web app that automatically digitizes data points from uploaded chart images (line/scatter/bar/Kaplan-Meier), with a visual drag-to-correct review step and CSV/Excel export.

**Architecture:** Python FastAPI backend (OpenCV geometry detection + Tesseract OCR + color-based curve tracing, stateless, deployed to Render as a Docker service) and a React+TypeScript frontend (Vite, Konva canvas for the review/edit step, deployed to Vercel). No database, no external LLM API. See `docs/superpowers/specs/2026-07-07-chart-digitizer-design.md` for the full design rationale.

**Tech Stack:** Python 3.11, FastAPI, opencv-python-headless, pytesseract + system `tesseract-ocr`, numpy, openpyxl, itsdangerous (session cookies), pytest, matplotlib (test fixture generation) · React 18, TypeScript, Vite, react-konva, Vitest + React Testing Library

---

## File Structure

**Backend** (`backend/`):
- `backend/app/main.py` — FastAPI app, route registration, CORS
- `backend/app/auth.py` — password check, signed session cookie issue/verify, auth dependency
- `backend/app/pipeline/geometry.py` — plot box + axis line + tick pixel-position detection
- `backend/app/pipeline/ocr.py` — axis tick text, axis titles, legend text extraction via Tesseract
- `backend/app/pipeline/calibration.py` — pixel↔data transform fitting (linear + log)
- `backend/app/pipeline/classify.py` — chart-type heuristic (line/scatter/bar/KM)
- `backend/app/pipeline/curves.py` — per-series color mask isolation + point/curve extraction per chart type
- `backend/app/pipeline/overlay.py` — draws detected points/curves back onto the image for QC
- `backend/app/pipeline/pipeline.py` — orchestrates the full extraction, returns structured result
- `backend/app/export.py` — CSV/Excel generation from corrected series data
- `backend/app/models.py` — Pydantic request/response models
- `backend/requirements.txt`, `backend/Dockerfile`
- `backend/tests/fixtures.py` — matplotlib-based synthetic chart image generator used across tests
- `backend/tests/test_*.py` — one test file per pipeline module

**Frontend** (`frontend/`):
- `frontend/src/pages/Login.tsx`
- `frontend/src/pages/Upload.tsx`
- `frontend/src/pages/Review.tsx`
- `frontend/src/components/ChartCanvas.tsx` — Konva image + draggable point/axis-marker overlay
- `frontend/src/components/DataTable.tsx`
- `frontend/src/api.ts` — typed fetch wrappers for backend endpoints
- `frontend/src/types.ts`
- `frontend/src/App.tsx`, `frontend/src/main.tsx`

---

## Task 1: Backend scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Test: `backend/tests/test_main.py`

- [ ] **Step 1: Create the directory structure and requirements file**

```bash
mkdir -p backend/app/pipeline backend/tests
```

`backend/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
opencv-python-headless==4.10.0.84
numpy==1.26.4
pytesseract==0.3.13
Pillow==10.4.0
openpyxl==3.1.5
itsdangerous==2.2.0
python-multipart==0.0.9
pytest==8.3.3
httpx==0.27.2
matplotlib==3.9.2
```

- [ ] **Step 2: Write the failing test for a health check endpoint**

`backend/tests/test_main.py`:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 3: Run test to verify it fails**

Run (from `backend/`): `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'` or import error, since `app/main.py` doesn't exist yet.

- [ ] **Step 4: Create empty package init and the FastAPI app**

`backend/app/__init__.py`: (empty file)

`backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Chart Digitizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 5: Install dependencies and run test to verify it passes**

Run: `pip install -r backend/requirements.txt`
Run: `cd backend && pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/app/__init__.py backend/app/main.py backend/tests/test_main.py
git commit -m "feat: scaffold FastAPI backend with health check"
```

---

## Task 2: Synthetic chart test-fixture generator

Every later pipeline task needs known-ground-truth chart images to test against. Build this once, reuse everywhere.

**Files:**
- Create: `backend/tests/fixtures.py`
- Test: `backend/tests/test_fixtures.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_fixtures.py`:
```python
import numpy as np
from tests.fixtures import make_line_chart


def test_make_line_chart_returns_image_and_ground_truth():
    image, truth = make_line_chart()

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3  # H, W, 3 (BGR)
    assert truth["x_range"] == (0.0, 10.0)
    assert truth["y_range"] == (0.0, 100.0)
    assert len(truth["series"]) == 1
    assert truth["series"][0]["name"] == "Series A"
    assert len(truth["series"][0]["points"]) == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fixtures'`

- [ ] **Step 3: Write the fixture generator**

`backend/tests/fixtures.py`:
```python
"""Generates synthetic chart images with known ground-truth data for tests."""
import io

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_bgr_array(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    file_bytes = np.frombuffer(buf.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def make_line_chart():
    x = np.linspace(0, 10, 11)
    y = x * 8 + 5  # simple known line: y = 8x + 5, range 5..85

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, color="blue", linewidth=2)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Value")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.0, 10.0),
        "y_range": (0.0, 100.0),
        "series": [
            {"name": "Series A", "color": "blue", "points": list(zip(x.tolist(), y.tolist()))}
        ],
    }
    return image, truth


def make_km_chart():
    """Two-arm step-function survival curve with censoring ticks."""
    t = np.array([0, 2, 4, 6, 8, 10, 12])
    surv_a = np.array([1.0, 0.95, 0.85, 0.70, 0.60, 0.55, 0.50])
    surv_b = np.array([1.0, 0.90, 0.70, 0.50, 0.35, 0.25, 0.20])
    censor_t_a = np.array([5, 9])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(t, surv_a, where="post", color="red", linewidth=2, label="Arm A")
    ax.step(t, surv_b, where="post", color="green", linewidth=2, label="Arm B")
    ax.plot(censor_t_a, np.interp(censor_t_a, t, surv_a), "+", color="red", markersize=10)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="upper right")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.0, 12.0),
        "y_range": (0.0, 1.0),
        "series": [
            {"name": "Arm A", "color": "red", "points": list(zip(t.tolist(), surv_a.tolist()))},
            {"name": "Arm B", "color": "green", "points": list(zip(t.tolist(), surv_b.tolist()))},
        ],
    }
    return image, truth


def make_scatter_chart():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, size=15)
    y = 3 * x + rng.normal(0, 1, size=15)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="purple")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 40)
    ax.set_xlabel("Dose (mg)")
    ax.set_ylabel("Response")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.0, 10.0),
        "y_range": (0.0, 40.0),
        "series": [{"name": "Series A", "color": "purple", "points": list(zip(x.tolist(), y.tolist()))}],
    }
    return image, truth


def make_bar_chart():
    categories_x = np.array([1, 2, 3, 4])
    heights = np.array([12.0, 25.0, 18.0, 30.0])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories_x, heights, color="teal")
    ax.set_ylim(0, 40)
    ax.set_xlabel("Group")
    ax.set_ylabel("Count")

    image = _fig_to_bgr_array(fig)
    truth = {
        "x_range": (0.5, 4.5),
        "y_range": (0.0, 40.0),
        "series": [
            {"name": "Series A", "color": "teal", "points": list(zip(categories_x.tolist(), heights.tolist()))}
        ],
    }
    return image, truth
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_fixtures.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/tests/fixtures.py backend/tests/test_fixtures.py
git commit -m "test: add synthetic chart image generator for pipeline tests"
```

---

## Task 3: Plot box and axis tick pixel-position detection

**Files:**
- Create: `backend/app/pipeline/__init__.py`
- Create: `backend/app/pipeline/geometry.py`
- Test: `backend/tests/test_geometry.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_geometry.py`:
```python
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from tests.fixtures import make_line_chart


def test_detect_plot_box_finds_axes_rectangle():
    image, _ = make_line_chart()
    box = detect_plot_box(image)

    h, w = image.shape[:2]
    assert 0 <= box.left < box.right <= w
    assert 0 <= box.top < box.bottom <= h
    # plot area should be a substantial, sane portion of the image
    assert (box.right - box.left) > w * 0.4
    assert (box.bottom - box.top) > h * 0.4


def test_detect_tick_positions_finds_expected_count_on_each_axis():
    image, truth = make_line_chart()
    box = detect_plot_box(image)

    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")

    # matplotlib default gives ~6 ticks per axis for these ranges; allow slack
    assert 4 <= len(x_ticks) <= 8
    assert 4 <= len(y_ticks) <= 8
    # x ticks should be sorted left-to-right, y ticks top-to-bottom
    assert x_ticks == sorted(x_ticks)
    assert y_ticks == sorted(y_ticks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline'`

- [ ] **Step 3: Implement plot box and tick detection**

`backend/app/pipeline/__init__.py`: (empty file)

`backend/app/pipeline/geometry.py`:
```python
"""Detects the plot bounding box and axis tick pixel positions using classic
edge/line detection. Charts are assumed to have a visible left/bottom axis
spine (true for matplotlib default style and the vast majority of published
figures)."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlotBox:
    left: int
    top: int
    right: int
    bottom: int


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_plot_box(image: np.ndarray) -> PlotBox:
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)

    h, w = gray.shape
    min_line_length = min(h, w) * 0.3
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=min_line_length, maxLineGap=10,
    )
    if lines is None:
        raise ValueError("No axis lines detected in image")

    horizontals, verticals = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(y1 - y2) < 3 and abs(x1 - x2) > min_line_length:
            horizontals.append((y1 + y2) / 2)
        elif abs(x1 - x2) < 3 and abs(y1 - y2) > min_line_length:
            verticals.append((x1 + x2) / 2)

    if not horizontals or not verticals:
        raise ValueError("Could not find both horizontal and vertical axis lines")

    # bottom axis = lowest horizontal line, left axis = leftmost vertical line,
    # top/right bounds taken from the extreme opposite lines found (handles
    # boxed plots) or fall back to image edges (handles open-spine plots).
    bottom = int(max(horizontals))
    top_candidates = [y for y in horizontals if y < bottom - 10]
    top = int(min(top_candidates)) if top_candidates else int(h * 0.05)

    left = int(min(verticals))
    right_candidates = [x for x in verticals if x > left + 10]
    right = int(max(right_candidates)) if right_candidates else int(w * 0.95)

    return PlotBox(left=left, top=top, right=right, bottom=bottom)


def detect_tick_positions(image: np.ndarray, box: PlotBox, axis: str) -> list[int]:
    """Finds tick mark pixel positions along the given axis by looking for
    short perpendicular strokes just outside the plot box spine."""
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)

    if axis == "x":
        # look in a thin strip just below the bottom spine
        strip = edges[box.bottom + 1: box.bottom + 8, box.left:box.right]
        column_sums = strip.sum(axis=0)
        positions_local = _find_peak_centers(column_sums)
        return [box.left + p for p in positions_local]
    elif axis == "y":
        # look in a thin strip just left of the left spine
        strip = edges[box.top:box.bottom, max(0, box.left - 8):box.left - 1]
        row_sums = strip.sum(axis=1)
        positions_local = _find_peak_centers(row_sums)
        return [box.top + p for p in positions_local]
    else:
        raise ValueError(f"Unknown axis: {axis}")


def _find_peak_centers(signal: np.ndarray, min_gap: int = 5) -> list[int]:
    """Groups consecutive nonzero indices in a 1D signal into clusters and
    returns each cluster's center — used to turn a strip of tick-mark pixels
    into one position per tick."""
    nonzero = np.where(signal > 0)[0]
    if len(nonzero) == 0:
        return []

    clusters: list[list[int]] = [[int(nonzero[0])]]
    for idx in nonzero[1:]:
        if idx - clusters[-1][-1] <= min_gap:
            clusters[-1].append(int(idx))
        else:
            clusters.append([int(idx)])

    return [int(np.mean(cluster)) for cluster in clusters]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_geometry.py -v`
Expected: PASS. If tick counts fall outside the 4-8 range, print `x_ticks`/`y_ticks` and adjust `min_gap` in `_find_peak_centers` or the strip thickness (`+8`) — matplotlib tick mark length varies slightly with DPI.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/__init__.py backend/app/pipeline/geometry.py backend/tests/test_geometry.py
git commit -m "feat: detect plot box and axis tick pixel positions"
```

---

## Task 4: OCR of axis tick labels and titles

**Prerequisite:** the `tesseract` binary must be installed locally to run these tests (`choco install tesseract` on Windows, or `apt-get install tesseract-ocr` on Linux/CI). This is separate from the `pytesseract` pip package, which is just a wrapper.

**Files:**
- Create: `backend/app/pipeline/ocr.py`
- Test: `backend/tests/test_ocr.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_ocr.py`:
```python
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_line_chart


def test_read_axis_tick_labels_reads_numeric_y_values():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    y_ticks = detect_tick_positions(image, box, axis="y")

    labels = read_axis_tick_labels(image, box, y_ticks, axis="y")

    assert len(labels) == len(y_ticks)
    numeric_values = [label.value for label in labels if label.value is not None]
    # y range is 0..100 in steps of 20 by default matplotlib behavior;
    # require most ticks to be read correctly (OCR isn't perfect)
    assert len(numeric_values) >= len(y_ticks) - 1
    assert numeric_values == sorted(numeric_values, reverse=True)  # top-to-bottom => descending
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.ocr'`

- [ ] **Step 3: Implement OCR reading**

`backend/app/pipeline/ocr.py`:
```python
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
        crop = image[min(h, box.bottom + 35): min(h, box.bottom + 65), box.left:box.right]
        processed = _preprocess_for_ocr(crop)
        return pytesseract.image_to_string(processed, config="--psm 7").strip()
    elif axis == "y":
        crop = image[box.top:box.bottom, max(0, box.left - 95): max(0, box.left - 60)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
        upscaled = cv2.resize(rotated, None, fx=_UPSCALE_FACTOR, fy=_UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(binary, config="--psm 7").strip()
    else:
        raise ValueError(f"Unknown axis: {axis}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_ocr.py -v`
Expected: PASS. OCR accuracy is sensitive to crop padding — if it fails, print `labels` to see the raw `text` Tesseract returned per tick and adjust the crop box size/offset in `read_axis_tick_labels`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/ocr.py backend/tests/test_ocr.py
git commit -m "feat: OCR axis tick labels and titles via Tesseract"
```

---

## Task 5: Legend detection (series name + color)

**Files:**
- Create: `backend/app/pipeline/legend.py`
- Test: `backend/tests/test_legend.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_legend.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_legend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.legend'`

- [ ] **Step 3: Implement legend detection**

`backend/app/pipeline/legend.py`:
```python
"""Finds legend entries by looking for small solid-color swatches followed
immediately (to the right) by a run of text, then OCRs that text. Legends in
matplotlib-style figures are drawn as a tight box of swatch+label rows, so
scanning for that pattern generalizes reasonably well beyond matplotlib too."""
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from app.pipeline.geometry import PlotBox

_MIN_SWATCH_AREA = 40
_MAX_SWATCH_AREA = 2000
_SWATCH_ASPECT_RANGE = (0.3, 3.0)  # roughly square-ish to wide rectangle


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

        text_crop = region[max(0, y - 4): y + h + 4, x + w + 3: min(region.shape[1], x + w + 150)]
        if text_crop.size == 0:
            continue

        gray = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(binary, config="--psm 7").strip()

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_legend.py -v`
Expected: PASS. If swatches aren't found, print `saturated_mask` contour count and loosen `_MIN_SWATCH_AREA`/`_SWATCH_ASPECT_RANGE`; if OCR text is empty, widen the `text_crop` width.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/legend.py backend/tests/test_legend.py
git commit -m "feat: detect legend entries (color swatch + series name)"
```

---

## Task 6: Pixel-to-data calibration (linear + log)

**Files:**
- Create: `backend/app/pipeline/calibration.py`
- Test: `backend/tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_calibration.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.calibration'`

- [ ] **Step 3: Implement calibration fitting**

`backend/app/pipeline/calibration.py`:
```python
"""Fits a pixel-position -> data-value mapping from a set of OCR'd tick
labels, using least-squares linear regression (or log10-space regression for
log-scaled axes). Ticks whose OCR text didn't parse as a number are dropped
before fitting."""
from dataclasses import dataclass

import numpy as np

from app.pipeline.ocr import TickLabel


@dataclass
class AxisCalibration:
    slope: float
    intercept: float
    log_scale: bool

    def pixel_to_value(self, pixel: float) -> float:
        raw = self.slope * pixel + self.intercept
        return 10 ** raw if self.log_scale else raw


def fit_axis_calibration(labels: list[TickLabel], log_scale: bool) -> AxisCalibration:
    readable = [label for label in labels if label.value is not None]
    if len(readable) < 2:
        raise ValueError(f"Need at least 2 readable tick labels to calibrate, got {len(readable)}")

    pixels = np.array([label.pixel_position for label in readable], dtype=float)
    values = np.array([label.value for label in readable], dtype=float)

    if log_scale:
        if np.any(values <= 0):
            raise ValueError("Log-scale axis cannot have zero or negative tick values")
        values = np.log10(values)

    slope, intercept = np.polyfit(pixels, values, deg=1)
    return AxisCalibration(slope=float(slope), intercept=float(intercept), log_scale=log_scale)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/calibration.py backend/tests/test_calibration.py
git commit -m "feat: fit pixel-to-data axis calibration (linear and log scale)"
```

---

## Task 7: Chart-type classification heuristic

**Files:**
- Create: `backend/app/pipeline/classify.py`
- Test: `backend/tests/test_classify.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_classify.py`:
```python
from app.pipeline.classify import ChartType, classify_chart_type
from app.pipeline.geometry import detect_plot_box
from tests.fixtures import make_bar_chart, make_km_chart, make_line_chart, make_scatter_chart


def test_classify_line_chart():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.LINE


def test_classify_scatter_chart():
    image, _ = make_scatter_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.SCATTER


def test_classify_bar_chart():
    image, _ = make_bar_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.BAR


def test_classify_km_chart():
    image, _ = make_km_chart()
    box = detect_plot_box(image)
    assert classify_chart_type(image, box) == ChartType.KAPLAN_MEIER
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.classify'`

- [ ] **Step 3: Implement classification heuristic**

`backend/app/pipeline/classify.py`:
```python
"""Classifies chart type from plotted shape characteristics — no OCR or
labels needed. Order of checks matters: bar charts are ruled in first since
they have the most distinctive signature (large filled rectangular blobs
sharing a common baseline), then step-function detection distinguishes
Kaplan-Meier curves from ordinary line charts, then remaining charts are
split into line vs. scatter by how much of each column-of-pixels forms a
continuous stroke versus isolated blobs."""
from enum import Enum

import cv2
import numpy as np

from app.pipeline.geometry import PlotBox


class ChartType(str, Enum):
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    KAPLAN_MEIER = "kaplan_meier"


def _non_background_mask(region: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    return mask


def _is_bar_chart(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = mask.shape[0]
    bar_like = 0
    for contour in contours:
        x, y, w, cy = cv2.boundingRect(contour)
        area = w * cy
        if area < mask.size * 0.01:
            continue
        # bars are tall filled rectangles that touch (or nearly touch) the bottom
        fill_ratio = cv2.contourArea(contour) / area if area else 0
        touches_bottom = (y + cy) >= h - 5
        if fill_ratio > 0.85 and touches_bottom and cy > w:
            bar_like += 1
    return bar_like >= 1


def _has_step_pattern(mask: np.ndarray) -> bool:
    """Step functions have many horizontal runs connected by short vertical
    jumps — approximate by counting columns where the stroke's vertical
    extent is near-zero (flat run) versus columns with a tall vertical jump."""
    column_extents = []
    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        column_extents.append(rows.max() - rows.min())

    if not column_extents:
        return False

    extents = np.array(column_extents)
    flat_columns = np.sum(extents < 3)
    jump_columns = np.sum(extents > mask.shape[0] * 0.05)
    return flat_columns > len(extents) * 0.5 and jump_columns > 0


def _is_scatter(mask: np.ndarray) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_blobs = [c for c in contours if cv2.contourArea(c) < mask.size * 0.02]
    # scatter: many small, disconnected blobs rather than one continuous stroke
    return len(small_blobs) >= 6


def classify_chart_type(image: np.ndarray, box: PlotBox) -> ChartType:
    region = image[box.top:box.bottom, box.left:box.right]
    mask = _non_background_mask(region)

    if _is_bar_chart(mask):
        return ChartType.BAR
    if _has_step_pattern(mask):
        return ChartType.KAPLAN_MEIER
    if _is_scatter(mask):
        return ChartType.SCATTER
    return ChartType.LINE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_classify.py -v`
Expected: PASS. If a chart type is misclassified, print the intermediate `mask` stats (`flat_columns`, `jump_columns`, blob count) for that fixture and tune the thresholds — these are heuristics, not exact rules, and thresholds may need adjusting once tested against real (non-synthetic) chart images later in Task 15.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/classify.py backend/tests/test_classify.py
git commit -m "feat: classify chart type (line/scatter/bar/Kaplan-Meier) from shape heuristics"
```

---

## Task 8: Series color-mask isolation

Every extraction routine (line tracing, scatter centroids, bar extents) needs to first isolate "just this series' pixels" from the plot area. Build that once as a shared building block.

**Files:**
- Create: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_mask.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_mask.py`:
```python
import numpy as np

from app.pipeline.curves import isolate_series_mask
from app.pipeline.geometry import detect_plot_box
from tests.fixtures import make_km_chart


def test_isolate_series_mask_separates_red_and_green_arms():
    image, _ = make_km_chart()
    box = detect_plot_box(image)

    red_mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=40)
    green_mask = isolate_series_mask(image, box, target_color_bgr=(0, 255, 0), tolerance=40)

    assert red_mask.sum() > 0
    assert green_mask.sum() > 0
    # the two masks should be almost entirely disjoint
    overlap = np.logical_and(red_mask > 0, green_mask > 0).sum()
    assert overlap < min(red_mask.sum(), green_mask.sum()) * 0.05 / 255
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_mask.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.curves'`

- [ ] **Step 3: Implement color-mask isolation**

`backend/app/pipeline/curves.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_mask.py -v`
Expected: PASS. If overlap is too high, lower `tolerance`; if a mask sum is 0, the fixture's line color didn't match `target_color_bgr` closely enough — print `region[region.shape[0]//2]` to sample actual pixel colors along the curve.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_mask.py
git commit -m "feat: isolate per-series pixel mask by target color"
```

---

## Task 9: Line and Kaplan-Meier step-curve tracing

**Files:**
- Modify: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_line.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_line.py`:
```python
import pytest

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import isolate_series_mask, trace_line_curve
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_km_chart, make_line_chart


def _calibrate(image, box):
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_labels = read_axis_tick_labels(image, box, x_ticks, axis="x")
    y_labels = read_axis_tick_labels(image, box, y_ticks, axis="y")
    return (
        fit_axis_calibration(x_labels, log_scale=False),
        fit_axis_calibration(y_labels, log_scale=False),
    )


def test_trace_line_curve_recovers_known_line():
    image, truth = make_line_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(255, 0, 0), tolerance=60)  # blue
    points = trace_line_curve(mask, box, x_cal, y_cal, step=False)

    assert len(points) > 5
    # y = 8x + 5 in the fixture; check a mid-range traced point is close
    mid_points = [p for p in points if 4 <= p[0] <= 6]
    assert mid_points
    x, y = mid_points[len(mid_points) // 2]
    assert y == pytest.approx(8 * x + 5, abs=5)


def test_trace_line_curve_step_mode_recovers_km_curve():
    image, truth = make_km_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=60)  # red
    points = trace_line_curve(mask, box, x_cal, y_cal, step=True)

    assert len(points) > 3
    # survival should be non-increasing across a KM curve
    y_values = [p[1] for p in points]
    assert all(y_values[i] >= y_values[i + 1] - 0.05 for i in range(len(y_values) - 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_line.py -v`
Expected: FAIL — `ImportError: cannot import name 'trace_line_curve'`

- [ ] **Step 3: Implement line/step tracing**

Append to `backend/app/pipeline/curves.py`:
```python
def trace_line_curve(
    mask: np.ndarray,
    box: PlotBox,
    x_calibration: AxisCalibration,
    y_calibration: AxisCalibration,
    step: bool,
) -> list[tuple[float, float]]:
    """For each pixel-column containing series pixels, takes the vertical
    center (line mode) or, for step-function curves, keeps every distinct
    flat run's y-level so vertical jumps are preserved as separate points."""
    points_px: list[tuple[int, int]] = []

    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        if step:
            # a step curve can have two y-levels in the same column at a
            # jump; record the extremes rather than averaging them away
            points_px.append((col, int(rows.min())))
            if rows.max() != rows.min():
                points_px.append((col, int(rows.max())))
        else:
            points_px.append((col, int(rows.mean())))

    points_px.sort(key=lambda p: p[0])

    points_data = [
        (
            x_calibration.pixel_to_value(box.left + col),
            y_calibration.pixel_to_value(box.top + row),
        )
        for col, row in points_px
    ]
    return points_data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_line.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_line.py
git commit -m "feat: trace line and step-function (Kaplan-Meier) curves to data points"
```

---

## Task 10: Scatter centroid and bar extent extraction

**Files:**
- Modify: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_scatter_bar.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_scatter_bar.py`:
```python
import pytest

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import extract_bar_heights, extract_scatter_points, isolate_series_mask
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_bar_chart, make_scatter_chart


def _calibrate(image, box):
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_labels = read_axis_tick_labels(image, box, x_ticks, axis="x")
    y_labels = read_axis_tick_labels(image, box, y_ticks, axis="y")
    return (
        fit_axis_calibration(x_labels, log_scale=False),
        fit_axis_calibration(y_labels, log_scale=False),
    )


def test_extract_scatter_points_recovers_approximate_count():
    image, truth = make_scatter_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(128, 0, 128), tolerance=70)  # purple
    points = extract_scatter_points(mask, box, x_cal, y_cal)

    assert len(points) == pytest.approx(len(truth["series"][0]["points"]), abs=3)


def test_extract_bar_heights_recovers_known_heights():
    image, truth = make_bar_chart()
    box = detect_plot_box(image)
    x_cal, y_cal = _calibrate(image, box)

    mask = isolate_series_mask(image, box, target_color_bgr=(128, 128, 0), tolerance=70)  # teal
    bars = extract_bar_heights(mask, box, x_cal, y_cal)

    assert len(bars) == 4
    heights = sorted(y for _, y in bars)
    expected = sorted(y for _, y in truth["series"][0]["points"])
    for actual, exp in zip(heights, expected):
        assert actual == pytest.approx(exp, abs=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_scatter_bar.py -v`
Expected: FAIL — `ImportError: cannot import name 'extract_scatter_points'`

- [ ] **Step 3: Implement scatter and bar extraction**

Append to `backend/app/pipeline/curves.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_scatter_bar.py -v`
Expected: PASS. If bar count is wrong, adjacent bars may be merging in the mask — check the fixture's bar spacing vs. `tolerance` used in `isolate_series_mask`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_scatter_bar.py
git commit -m "feat: extract scatter point centroids and bar heights"
```

---

## Task 11: Kaplan-Meier censoring tick detection

**Files:**
- Modify: `backend/app/pipeline/curves.py`
- Test: `backend/tests/test_curves_censoring.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_curves_censoring.py`:
```python
from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.curves import detect_censoring_marks, isolate_series_mask
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from tests.fixtures import make_km_chart


def test_detect_censoring_marks_finds_plus_markers():
    image, truth = make_km_chart()
    box = detect_plot_box(image)
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_cal = fit_axis_calibration(read_axis_tick_labels(image, box, x_ticks, axis="x"), log_scale=False)
    y_cal = fit_axis_calibration(read_axis_tick_labels(image, box, y_ticks, axis="y"), log_scale=False)

    mask = isolate_series_mask(image, box, target_color_bgr=(0, 0, 255), tolerance=60)  # red arm
    marks = detect_censoring_marks(mask, box, x_cal, y_cal)

    # fixture has 2 censoring points on the red arm, at t=5 and t=9
    assert len(marks) == 2
    times = sorted(x for x, _ in marks)
    assert times[0] == pytest.approx(5, abs=1)
    assert times[1] == pytest.approx(9, abs=1)
```

Add `import pytest` at the top of the test file alongside the other imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_curves_censoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_censoring_marks'`

- [ ] **Step 3: Implement censoring mark detection**

Append to `backend/app/pipeline/curves.py`:
```python
def detect_censoring_marks(
    mask: np.ndarray, box: PlotBox, x_calibration: AxisCalibration, y_calibration: AxisCalibration
) -> list[tuple[float, float]]:
    """Censoring ticks (typically '+' or '|' markers) are small isolated
    blobs disconnected from the main step-curve stroke, roughly
    square/plus-shaped rather than long thin horizontal/vertical runs. This
    finds small blobs whose bounding box is close to square and clearly
    smaller than a full step-run segment."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marks = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 6 or area > 400:
            continue
        aspect = w / h if h else 0
        if not (0.4 <= aspect <= 2.5):
            continue  # skip long thin strokes (part of the main curve line)

        center_x_px = box.left + x + w / 2
        center_y_px = box.top + y + h / 2
        marks.append(
            (
                x_calibration.pixel_to_value(center_x_px),
                y_calibration.pixel_to_value(center_y_px),
            )
        )
    return marks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_curves_censoring.py -v`
Expected: PASS. If marks are missed or the main curve stroke is picked up as a false mark, tune the `area`/`aspect` thresholds against this fixture's actual marker pixel size.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/curves.py backend/tests/test_curves_censoring.py
git commit -m "feat: detect Kaplan-Meier censoring tick marks"
```

---

## Task 12: Overlay image generation

**Files:**
- Create: `backend/app/pipeline/overlay.py`
- Test: `backend/tests/test_overlay.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_overlay.py`:
```python
import numpy as np

from app.pipeline.calibration import fit_axis_calibration
from app.pipeline.geometry import detect_plot_box, detect_tick_positions
from app.pipeline.ocr import read_axis_tick_labels
from app.pipeline.overlay import draw_overlay
from tests.fixtures import make_line_chart


def test_draw_overlay_returns_modified_image_same_shape():
    image, _ = make_line_chart()
    box = detect_plot_box(image)
    x_ticks = detect_tick_positions(image, box, axis="x")
    y_ticks = detect_tick_positions(image, box, axis="y")
    x_cal = fit_axis_calibration(read_axis_tick_labels(image, box, x_ticks, axis="x"), log_scale=False)
    y_cal = fit_axis_calibration(read_axis_tick_labels(image, box, y_ticks, axis="y"), log_scale=False)

    series = [{"name": "Series A", "color_bgr": (255, 0, 0), "points": [(1.0, 13.0), (5.0, 45.0), (9.0, 77.0)]}]

    overlay_image = draw_overlay(image, box, x_cal, y_cal, series)

    assert overlay_image.shape == image.shape
    # overlay must actually change some pixels (markers drawn), not be a no-op copy
    assert not np.array_equal(overlay_image, image)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_overlay.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline.overlay'`

- [ ] **Step 3: Implement overlay drawing**

`backend/app/pipeline/overlay.py`:
```python
"""Draws detected data points back onto a copy of the original chart image,
for the user to visually sanity-check auto-extraction accuracy."""
import cv2
import numpy as np

from app.pipeline.calibration import AxisCalibration
from app.pipeline.geometry import PlotBox

_MARKER_RADIUS = 4
_MARKER_COLOR_BGR = (0, 0, 0)  # black outline, drawn under a white fill for contrast on any series color
_MARKER_OUTLINE_COLOR_BGR = (255, 255, 255)


def _value_to_pixel(value: float, calibration: AxisCalibration, origin: int) -> int:
    # invert calibration.pixel_to_value: value = slope*pixel + intercept (or slope*pixel+intercept in log space)
    raw_value = np.log10(value) if calibration.log_scale else value
    pixel = (raw_value - calibration.intercept) / calibration.slope
    return int(round(pixel))


def draw_overlay(
    image: np.ndarray,
    box: PlotBox,
    x_calibration: AxisCalibration,
    y_calibration: AxisCalibration,
    series: list[dict],
) -> np.ndarray:
    overlay = image.copy()

    for s in series:
        for x_val, y_val in s["points"]:
            px = _value_to_pixel(x_val, x_calibration, box.left)
            py = _value_to_pixel(y_val, y_calibration, box.top)
            cv2.circle(overlay, (px, py), _MARKER_RADIUS + 1, _MARKER_OUTLINE_COLOR_BGR, -1)
            cv2.circle(overlay, (px, py), _MARKER_RADIUS, s.get("color_bgr", _MARKER_COLOR_BGR), -1)

    return overlay
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_overlay.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/overlay.py backend/tests/test_overlay.py
git commit -m "feat: draw detected points overlay onto original chart image"
```

---
