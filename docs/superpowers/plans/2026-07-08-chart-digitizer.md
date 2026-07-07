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
